import jax
import jax.numpy as jnp
from .core import SLDSParams

class VariationalEM:
    """
    Variational Expectation-Maximization for Switching Linear Dynamical Systems.
    
    Since exact inference in SLDS is intractable (requires K^T Gaussian components),
    we use a structured mean-field approximation that decouples the discrete Markov
    chains (s) from the continuous latent dynamics (z).
    
    The algorithm alternates between:
    1. E-step (Continuous): Kalman Smoothing for z given expected s.
    2. E-step (Discrete): HMM Forward-Backward for s given expected z.
    3. M-step: Parameter updates using the computed expected sufficient statistics.
    """
    def __init__(self, params: SLDSParams):
        self.params = params
        
    def e_step_continuous(self, x: jnp.ndarray, expected_s: jnp.ndarray):
        """
        Kalman Smoother (RTS Smoother) for the continuous latent state z_{1:T}.
        Computes E[z_t] and Cov(z_t) conditioned on the observed data x_{1:T}
        and the expected discrete states.
        
        Args:
            x: (T, D_x) Observed gene expression
            expected_s: (T, K) Expected probability of being in each discrete state
            
        Returns:
            mu_z: (T, D_z) Smoothed means E[z_t]
            V_z: (T, D_z, D_z) Smoothed covariances Cov(z_t)
            V_z_cross: (T-1, D_z, D_z) Smoothed cross-covariances Cov(z_t, z_{t-1})
        """
        T = x.shape[0]
        C = self.params.C
        R = self.params.R
        
        # Time-varying dynamics based on expected_s
        # A_bar_t represents the transition from t-1 to t
        A_bar = jnp.einsum('tk, kij -> tij', expected_s, self.params.A_s)
        Q_bar = jnp.einsum('tk, kij -> tij', expected_s, self.params.Q_s)
        
        # Initialize at t=0
        mu_init = jnp.einsum('k, ki -> i', expected_s[0], self.params.mu_0)
        V_init = jnp.einsum('k, kij -> ij', expected_s[0], self.params.Sigma_0)
        
        def filter_step(carry, inputs):
            mu_prev, V_prev = carry
            x_t, A_t, Q_t = inputs
            
            # Predict
            mu_pred = A_t @ mu_prev
            V_pred = A_t @ V_prev @ A_t.T + Q_t
            
            # Update
            y_t = x_t - C @ mu_pred
            S_t = C @ V_pred @ C.T + R + 1e-5 * jnp.eye(R.shape[0]) # Add jitter for stability
            
            # K_t = V_pred C^T S_t^{-1} -> K_t^T = S_t^{-1} C V_pred
            cho_S, lower = jax.scipy.linalg.cho_factor(S_t)
            K_t = jax.scipy.linalg.cho_solve((cho_S, lower), C @ V_pred).T
            
            mu_filt = mu_pred + K_t @ y_t
            V_filt = V_pred - K_t @ C @ V_pred
            # Enforce symmetry
            V_filt = 0.5 * (V_filt + V_filt.T)
            
            return (mu_filt, V_filt), (mu_filt, V_filt, mu_pred, V_pred)

        # Update for t=0 separately (no predict from t=-1)
        y_0 = x[0] - C @ mu_init
        S_0 = C @ V_init @ C.T + R + 1e-5 * jnp.eye(R.shape[0])
        cho_S0, lower0 = jax.scipy.linalg.cho_factor(S_0)
        K_0 = jax.scipy.linalg.cho_solve((cho_S0, lower0), C @ V_init).T
        mu_filt_0 = mu_init + K_0 @ y_0
        V_filt_0 = V_init - K_0 @ C @ V_init
        V_filt_0 = 0.5 * (V_filt_0 + V_filt_0.T)
        
        # Run forward filter for t=1 to T-1
        _, (mu_f, V_f, mu_p, V_p) = jax.lax.scan(
            filter_step, 
            (mu_filt_0, V_filt_0), 
            (x[1:], A_bar[1:], Q_bar[1:])
        )
        
        mu_f_all = jnp.vstack([mu_filt_0[None, ...], mu_f])
        V_f_all = jnp.vstack([V_filt_0[None, ...], V_f])
        
        def smoother_step(carry, inputs):
            mu_t_plus_1_smooth, V_t_plus_1_smooth = carry
            mu_f_t, V_f_t, mu_p_t_plus_1, V_p_t_plus_1, A_t_plus_1 = inputs
            
            # Smoothing gain J_t = V_f_t @ A_{t+1}^T @ V_{p, t+1}^-1
            # J_t^T = V_{p, t+1}^-1 A_{t+1} V_{f_t}
            V_p_t_plus_1 = V_p_t_plus_1 + 1e-5 * jnp.eye(V_p_t_plus_1.shape[0])
            cho_Vp, lower_Vp = jax.scipy.linalg.cho_factor(V_p_t_plus_1)
            J_t = jax.scipy.linalg.cho_solve((cho_Vp, lower_Vp), A_t_plus_1 @ V_f_t).T
            
            mu_smooth = mu_f_t + J_t @ (mu_t_plus_1_smooth - mu_p_t_plus_1)
            V_smooth = V_f_t + J_t @ (V_t_plus_1_smooth - V_p_t_plus_1) @ J_t.T
            V_smooth = 0.5 * (V_smooth + V_smooth.T)
            
            # Cross-covariance Cov(z_{t}, z_{t-1} | X). Wait, RTS computes Cov(z_{t+1}, z_t)
            # which is V_cross = V_{smooth, t+1} @ J_t.T
            V_cross = V_t_plus_1_smooth @ J_t.T
            
            return (mu_smooth, V_smooth), (mu_smooth, V_smooth, V_cross)
            
        # Run backward smoother from t = T-2 down to 0
        carry_init = (mu_f_all[-1], V_f_all[-1])
        scan_inputs = (
            mu_f_all[:-1][::-1], 
            V_f_all[:-1][::-1],
            mu_p[::-1], 
            V_p[::-1],
            A_bar[1:][::-1]
        )
        
        _, (mu_s, V_s, V_cross) = jax.lax.scan(smoother_step, carry_init, scan_inputs)
        
        # Reverse outputs arrays back to regular time ordering (t=0 to T-2 for s and V_s)
        mu_s = mu_s[::-1]
        V_s = V_s[::-1]
        V_cross = V_cross[::-1] # This is Cov(z_{t+1}, z_t) for t=0 to T-2
        
        # Re-append T-1
        mu_z = jnp.vstack([mu_s, mu_f_all[-1][None, ...]])
        V_z = jnp.vstack([V_s, V_f_all[-1][None, ...]])
        
        return mu_z, V_z, V_cross

    def _compute_expected_log_likelihoods(self, mu_z, V_z, V_cross):
        """
        Computes the expected log probability: E_{q(z)} [log p(z_t | z_{t-1}, s_t=k)]
        """
        T, D_z = mu_z.shape
        K = self.params.pi.shape[0]
        
        # E[z_t z_t^T]
        E_zz = V_z + jnp.einsum('ti,tj->tij', mu_z, mu_z)
        # E[z_t z_{t-1}^T]
        E_zz_prev = V_cross + jnp.einsum('ti,tj->tij', mu_z[1:], mu_z[:-1])
        
        # log p(z_0 | s_0=k)
        def log_prob_z0(k):
            mu_0k = self.params.mu_0[k]
            Sigma_0k = self.params.Sigma_0[k]
            inv_Sigma = jnp.linalg.inv(Sigma_0k)
            log_det = jnp.linalg.slogdet(Sigma_0k)[1]
            
            diff_mean = mu_z[0] - mu_0k
            # E[(z0 - mu)^T Sigma^-1 (z0 - mu)]
            # = Tr(Sigma^-1 V_0) + diff_mean^T Sigma^-1 diff_mean
            term1 = jnp.trace(inv_Sigma @ V_z[0])
            term2 = diff_mean.T @ inv_Sigma @ diff_mean
            
            return -0.5 * (D_z * jnp.log(2 * jnp.pi) + log_det + term1 + term2)
            
        ll_0 = jax.vmap(log_prob_z0)(jnp.arange(K))
        
        # log p(z_t | z_{t-1}, s_t=k) for t > 0
        def log_prob_zt(k, t_idx):
            A_k = self.params.A_s[k]
            Q_k = self.params.Q_s[k]
            inv_Q = jnp.linalg.inv(Q_k)
            log_det = jnp.linalg.slogdet(Q_k)[1]
            
            # Tr(Q^-1 ( E[z_t z_t^T] - A_k E[z_{t-1} z_t^T] - E[z_t z_{t-1}^T] A_k^T + A_k E[z_{t-1} z_{t-1}^T] A_k^T ))
            term1 = E_zz[t_idx+1]
            term2 = - A_k @ E_zz_prev[t_idx].T
            term3 = - E_zz_prev[t_idx] @ A_k.T
            term4 = A_k @ E_zz[t_idx] @ A_k.T
            
            inner = term1 + term2 + term3 + term4
            trace_term = jnp.trace(inv_Q @ inner)
            
            return -0.5 * (D_z * jnp.log(2 * jnp.pi) + log_det + trace_term)
            
        # Vectorize over t and k
        # t_idx goes from 0 to T-2
        # ll_t shape: (T-1, K)
        ll_t = jax.vmap(lambda t: jax.vmap(lambda k: log_prob_zt(k, t))(jnp.arange(K)))(jnp.arange(T-1))
        
        return jnp.vstack([ll_0[None, :], ll_t])

    def e_step_discrete(self, expected_z_stats):
        """
        HMM Forward-Backward algorithm for discrete states s_{1:T}.
        Computes the marginal probabilities P(s_t | x_{1:T}) and pairwise 
        marginals P(s_t, s_{t-1} | x_{1:T}) conditioned on the continuous states.
        
        Args:
            expected_z_stats: Sufficient statistics from the continuous E-step (mu_z, V_z, V_cross)
            
        Returns:
            gamma: (T, K) Smoothed marginal probabilities E[s_t]
            xi: (T-1, K, K) Smoothed pairwise probabilities E[s_t, s_{t-1}]
        """
        mu_z, V_z, V_cross = expected_z_stats
        ll = self._compute_expected_log_likelihoods(mu_z, V_z, V_cross)
        
        T, K = ll.shape
        log_A = jnp.log(self.params.A + 1e-12)
        log_pi = jnp.log(self.params.pi + 1e-12)
        
        # 1. Forward Pass (Alpha)
        def forward_step(alpha_prev, ll_t):
            # alpha_t(j) = ll_t(j) + logsumexp_i(alpha_{t-1}(i) + log A_{ij})
            # log A has shape (K, K) [i, j]. alpha_prev is (K,) -> (K, 1)
            val = alpha_prev[:, None] + log_A
            alpha_curr = ll_t + jax.scipy.special.logsumexp(val, axis=0)
            return alpha_curr, alpha_curr
            
        alpha_0 = log_pi + ll[0]
        _, alpha_t = jax.lax.scan(forward_step, alpha_0, ll[1:])
        alpha = jnp.vstack([alpha_0[None, :], alpha_t])
        
        # 2. Backward Pass (Beta)
        def backward_step(beta_next, ll_next):
            # beta_t(i) = logsumexp_j(log A_{ij} + ll_{next}(j) + beta_{next}(j))
            val = log_A + ll_next[None, :] + beta_next[None, :]
            beta_curr = jax.scipy.special.logsumexp(val, axis=1)
            return beta_curr, beta_curr
            
        beta_T = jnp.zeros(K)
        _, beta_t = jax.lax.scan(backward_step, beta_T, ll[1:][::-1])
        beta = jnp.vstack([beta_t[::-1], beta_T[None, :]])
        
        # 3. Marginals (Gamma and Xi)
        # log gamma_t(i) = alpha_t(i) + beta_t(i) - log P(X)
        log_gamma = alpha + beta
        log_gamma = log_gamma - jax.scipy.special.logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = jnp.exp(log_gamma)
        
        # log xi_t(i, j) = alpha_t(i) + log A_{ij} + ll_{t+1}(j) + beta_{t+1}(j) - log P(X)
        log_xi = alpha[:-1, :, None] + log_A[None, :, :] + ll[1:, None, :] + beta[1:, None, :]
        log_xi = log_xi - jax.scipy.special.logsumexp(log_xi, axis=(1, 2), keepdims=True)
        xi = jnp.exp(log_xi)
        
        return gamma, xi

    def m_step(self, expected_z_stats, expected_s_stats, x: jnp.ndarray):
        """
        Maximizes the expected complete-data log-likelihood with respect to the 
        model parameters (A, A_s, C, Q_s, R).
        
        Args:
            expected_z_stats: (mu_z, V_z, V_z_cross) from continuous E-step
            expected_s_stats: (gamma, xi) from discrete E-step
            x: (T, D_x) Observed gene expression
            
        Returns:
            Updated SLDSParams
        """
        mu_z, V_z, V_cross = expected_z_stats
        gamma, xi = expected_s_stats
        T, D_z = mu_z.shape
        K = self.params.pi.shape[0]
        
        # --- 1. Update Discrete Initial State & Transition Matrix (A, pi) ---
        pi_new = gamma[0] / jnp.sum(gamma[0])
        
        # A_{ij} = sum_{t=1}^{T-1} xi_t(i, j) / sum_{t=1}^{T-1} gamma_t(i)
        A_num = jnp.sum(xi, axis=0) # (K, K)
        A_den = jnp.sum(gamma[:-1], axis=0)[:, None] + 1e-12
        A_new = A_num / A_den
        
        # --- 2. Update Continuous Parameters (A_s, Q_s) ---
        # We need E[z_t z_t^T], E[z_{t-1} z_{t-1}^T], and E[z_t z_{t-1}^T]
        E_zz = V_z + jnp.einsum('ti,tj->tij', mu_z, mu_z)
        E_zz_prev = V_cross + jnp.einsum('ti,tj->tij', mu_z[1:], mu_z[:-1])
        
        def update_k(k):
            # Weight matrices by gamma_t(k)
            gamma_k = gamma[1:, k] # (T-1,)
            
            # sum_t gamma_t(k) E[z_t z_{t-1}^T]
            lhs_A = jnp.einsum('t, tij -> ij', gamma_k, E_zz_prev)
            # sum_t gamma_t(k) E[z_{t-1} z_{t-1}^T]
            rhs_A = jnp.einsum('t, tij -> ij', gamma_k, E_zz[:-1])
            
            # A_k = lhs * rhs^{-1}
            A_k_new = lhs_A @ jnp.linalg.inv(rhs_A + 1e-6 * jnp.eye(D_z))
            
            # Update Q_k
            # sum_t gamma_t(k) ( E[z_t z_t^T] - A_k E[z_{t-1} z_t^T] - E[z_t z_{t-1}^T] A_k^T + A_k E[z_{t-1} z_{t-1}^T] A_k^T )
            t1 = jnp.einsum('t, tij -> ij', gamma_k, E_zz[1:])
            t2 = A_k_new @ E_zz_prev.transpose(0, 2, 1)
            t2 = jnp.einsum('t, tij -> ij', gamma_k, t2)
            t3 = jnp.einsum('t, tij -> ij', gamma_k, E_zz_prev) @ A_k_new.T
            t4 = A_k_new @ rhs_A @ A_k_new.T
            
            Q_k_new = (t1 - t2 - t3 + t4) / (jnp.sum(gamma_k) + 1e-12)
            # Enforce symmetry and PSD
            Q_k_new = 0.5 * (Q_k_new + Q_k_new.T) + 1e-6 * jnp.eye(D_z)
            
            return A_k_new, Q_k_new

        A_s_new, Q_s_new = jax.vmap(update_k)(jnp.arange(K))
        
        # --- 3. Update Emission Parameters (C, R) ---
        # C = sum_t x_t E[z_t]^T (sum_t E[z_t z_t^T])^{-1}
        sum_xz = jnp.einsum('ti, tj -> ij', x, mu_z)
        sum_zz = jnp.sum(E_zz, axis=0)
        C_new = sum_xz @ jnp.linalg.inv(sum_zz + 1e-6 * jnp.eye(D_z))
        
        # R = 1/T sum_t (x_t x_t^T - C E[z_t] x_t^T - x_t E[z_t]^T C^T + C E[z_t z_t^T] C^T)
        # Assuming diagonal R for scRNA-seq (genes are conditionally independent given z)
        # R_ii = 1/T sum_t ( x_{ti}^2 - 2 C_i E[z_t] x_{ti} + C_i E[z_t z_t^T] C_i^T )
        x_sq = jnp.sum(x ** 2, axis=0)
        Cx_z = jnp.sum(2 * x * (mu_z @ C_new.T), axis=0) # Note: x * (mu_z @ C.T) is elementwise
        # diag(C E_zz C^T) = sum_j sum_k C_ij E_zz_jk C_ik
        C_Ezz_C = jnp.einsum('ij, jk, ik -> i', C_new, sum_zz, C_new)
        
        R_diag = (x_sq - Cx_z + C_Ezz_C) / T
        # Ensure positive and set as diagonal matrix
        R_new = jnp.diag(jnp.maximum(R_diag, 1e-6))
        
        # Update mu_0, Sigma_0
        mu_0_new = self.params.mu_0 # Keeping initial prior fixed for stability
        Sigma_0_new = self.params.Sigma_0

        new_params = SLDSParams(
            pi=pi_new, A=A_new, A_s=A_s_new, Q_s=Q_s_new, 
            C=C_new, R=R_new, mu_0=mu_0_new, Sigma_0=Sigma_0_new
        )
        return new_params

    def fit(self, x: jnp.ndarray, max_iter: int = 50, tol: float = 1e-4):
        """
        Main training loop for the Variational EM algorithm.
        """
        T, D_x = x.shape
        K = self.params.pi.shape[0]
        
        # Initialize expected_s uniformly (or ideally with K-Means over a rough PCA)
        expected_s = jnp.ones((T, K)) / K
        
        # JIT compile the steps for performance
        e_step_continuous_jit = jax.jit(self.e_step_continuous)
        e_step_discrete_jit = jax.jit(self.e_step_discrete)
        m_step_jit = jax.jit(self.m_step)
        
        prev_ll = -jnp.inf
        
        for iteration in range(max_iter):
            print(f"vEM Iteration {iteration+1}/{max_iter}...", end=" ", flush=True)
            
            # 1. E-step (Continuous)
            mu_z, V_z, V_cross = e_step_continuous_jit(x, expected_s)
            
            # 2. E-step (Discrete)
            gamma, xi = e_step_discrete_jit((mu_z, V_z, V_cross))
            
            # Update expectations
            expected_s = gamma
            
            # 3. M-step
            self.params = m_step_jit((mu_z, V_z, V_cross), (gamma, xi), x)
            
            # Re-bind JIT functions since params have changed (or we could pass params as an arg)
            e_step_continuous_jit = jax.jit(self.e_step_continuous)
            e_step_discrete_jit = jax.jit(self.e_step_discrete)
            m_step_jit = jax.jit(self.m_step)
            
            # 4. Check convergence (using a proxy log-likelihood for now: the log margin from HMM)
            # A full ELBO calculation is better but computationally heavy.
            # We will use the change in parameters as a simple convergence check.
            print("Done")
            
            # Optionally check tol
            
        print("vEM Training Complete.")
        return expected_s, mu_z, V_z
