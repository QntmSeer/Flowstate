import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from stateflow.sde_core import NeuralSLDS
from stateflow.sde_inference import ContinuousDiscreteKalmanFilter

class HybridSDEFit:
    """
    Fits the Neural SLDS model.
    Uses classical EM counts for the discrete transition matrix A,
    but uses gradient descent (Optax) over the filter likelihood for 
    the continuous SDE drift and diffusion parameters.
    """
    def __init__(self, sde_model: NeuralSLDS, C: jnp.ndarray, R: jnp.ndarray, K: int, D_z: int):
        self.sde_model = sde_model
        self.C = C
        self.R = R
        self.K = K
        self.D_z = D_z
        self.pi = jnp.ones(K) / K
        self.A = jnp.eye(K) * 0.9 + 0.1 / K
        
    def _hmm_e_step(self, filter_lls):
        """
        Standard discrete Forward-Backward given log-likelihoods from the continuous filter.
        filter_lls: (T, K)
        Returns: gamma (T, K), xi (T-1, K, K)
        """
        T = filter_lls.shape[0]
        log_A = jnp.log(self.A + 1e-12)
        log_pi = jnp.log(self.pi + 1e-12)
        
        # Forward pass
        def forward_step(alpha_prev, ll_t):
            val = alpha_prev[:, None] + log_A
            alpha_curr = ll_t + jax.scipy.special.logsumexp(val, axis=0)
            return alpha_curr, alpha_curr
            
        alpha_0 = log_pi + filter_lls[0]
        _, alpha_t = jax.lax.scan(forward_step, alpha_0, filter_lls[1:])
        alpha = jnp.vstack([alpha_0[None, :], alpha_t])
        
        # Backward pass
        def backward_step(beta_next, ll_next):
            val = beta_next + ll_next + log_A
            beta_curr = jax.scipy.special.logsumexp(val, axis=1)
            return beta_curr, beta_curr
            
        beta_T = jnp.zeros(self.K)
        _, beta_t = jax.lax.scan(backward_step, beta_T, filter_lls[1:], reverse=True)
        beta = jnp.vstack([beta_t, beta_T[None, :]])
        
        # Marginals
        log_gamma = alpha + beta
        log_Z = jax.scipy.special.logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = jnp.exp(log_gamma - log_Z)
        
        # Pairwise
        log_xi = alpha[:-1, :, None] + log_A[None, :, :] + filter_lls[1:, None, :] + beta[1:, None, :]
        xi = jnp.exp(log_xi - log_Z[:-1, :, None])
        
        return gamma, xi
        
    def _loss_fn(self, params, static, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray):
        """
        The negative log-likelihood of the observations given the predicted filter moments.
        We differentiate through the Continuous-Discrete Kalman Filter!
        """
        model = eqx.combine(params, static)
        filter = ContinuousDiscreteKalmanFilter(model, self.C, self.R)
        
        # For simplicity in testing, we write a quick likelihood evaluation here.
        # We approximate the marginalized log probability.
        
        mu_0 = jnp.zeros(self.D_z)
        P_0 = jnp.eye(self.D_z)
        
        # Call filter
        mu_filt, P_filt = filter.filter(xs, ts, expected_s, mu_0, P_0)
        
        # Compute L2 loss against observations 
        # (For true MLE we would use the Gaussian PDF via prediction residuals, but MSE is a robust start for drift fitting)
        preds = jax.vmap(lambda m: self.C @ m)(mu_filt)
        mse_loss = jnp.mean((xs - preds)**2)
        
        return mse_loss

    def fit(self, xs: jnp.ndarray, ts: jnp.ndarray, max_iter: int = 5):
        """
        Alternates HMM Discrete E-step and Optax Continuous gradient steps.
        """
        optimizer = optax.adam(1e-2)
        params, static = eqx.partition(self.sde_model, eqx.is_array)
        opt_state = optimizer.init(params)
        
        # Wrap loss with explicit jax.value_and_grad
        val_and_grad = jax.value_and_grad(self._loss_fn)
        
        # Initialize expected_s uniformly
        T = xs.shape[0]
        expected_s = jnp.ones((T, self.K)) / self.K
        
        print("Starting Hybrid Neural SDE Fit...")
        for iter in range(max_iter):
            # 1. OPTAX GRADIENT STEP FOR SDE PARAMETERS
            for grad_step in range(3):
                loss, grads = val_and_grad(params, static, xs, ts, expected_s)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
            
            self.sde_model = eqx.combine(params, static)
            
            # 2. DISCRETE E-STEP (HMM)
            # Re-evaluate filter to get rough likelihoods per state
            # (In practice, we compute log N(x_t | C mu_pred^k, S^k))
            # We use MSE as a proxy for likelihood assignment for simplicity
            filter = ContinuousDiscreteKalmanFilter(self.sde_model, self.C, self.R)
            mu_filt, _ = filter.filter(xs, ts, expected_s, jnp.zeros(self.D_z), jnp.eye(self.D_z))
            
            # MSE per state prediction
            preds = jax.vmap(lambda m: self.C @ m)(mu_filt) # True multi-modal requires parallel filters, using smoothed proxy
            
            # As a shortcut for this demo, we assume the prediction errors inform state likelihood
            # In a rigorous implementation, we run K parallel filters
            err = jnp.sum((xs - preds)**2, axis=1)
            # Pseudo-likelihoods 
            pseudo_lls = jnp.tile(-0.5 * err[:, None], (1, self.K)) 
            # Add state specific bias just to prevent singularity in demo
            pseudo_lls += jax.random.normal(jax.random.PRNGKey(iter), pseudo_lls.shape) * 0.1
            
            gamma, xi = self._hmm_e_step(pseudo_lls)
            expected_s = gamma
            
            # 3. DISCRETE M-STEP
            self.pi = gamma[0] / jnp.sum(gamma[0])
            A_num = jnp.sum(xi, axis=0)
            A_den = jnp.sum(gamma[:-1], axis=0)[:, None] + 1e-12
            self.A = A_num / A_den
            
            print(f"Iter {iter+1}/{max_iter} | Loss: {loss:.4f}")
            
        return expected_s, self.sde_model
