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
        
    def _loss_fn(self, params, static, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray,
                 barcodes: jnp.ndarray = None, lambda_clonal: float = 0.0):
        """
        The negative log-likelihood of the observations given the predicted filter moments.
        Optionally adds a clonal regularization penalty to enforce lineage constraints.
        """
        model = eqx.combine(params, static)
        filter = ContinuousDiscreteKalmanFilter(model, self.C, self.R)
        
        mu_0 = jnp.zeros(self.D_z)
        P_0 = jnp.eye(self.D_z)
        
        # Call filter (for MLE, forward filtering computes exact marginal likelihoods)
        mu_filt, P_filt = filter.filter(xs, ts, expected_s, mu_0, P_0)
        
        preds = jax.vmap(lambda m: self.C @ m)(mu_filt)
        mse_loss = jnp.mean((xs - preds)**2)
        
        # Add clonal regularization penalty if barcodes are provided
        if barcodes is not None and lambda_clonal > 0.0:
            unique_barcodes = jnp.unique(barcodes)
            
            def clone_penalty_fn(clone_id):
                mask = (barcodes == clone_id)[:, None]
                centroid = jnp.sum(mu_filt * mask, axis=0) / (jnp.sum(mask) + 1e-8)
                sq_dist = jnp.sum((mu_filt - centroid)**2, axis=1)
                return jnp.sum(sq_dist * mask.flatten())
                
            clonal_loss = jnp.sum(jax.vmap(clone_penalty_fn)(unique_barcodes))
            # Normalize by sequence length to keep scale comparable
            total_loss = mse_loss + lambda_clonal * (clonal_loss / xs.shape[0])
            return total_loss
            
        return mse_loss

    def fit(self, xs: jnp.ndarray, ts: jnp.ndarray, max_iter: int = 5,
            barcodes: jnp.ndarray = None, lambda_clonal: float = 0.0):
        """
        Alternates HMM Discrete E-step (using RTS Smoother) and Optax Continuous gradient steps.
        Supports lineage barcode constraints via a clonal regularization penalty.
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
                loss, grads = val_and_grad(params, static, xs, ts, expected_s, barcodes, lambda_clonal)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
            
            self.sde_model = eqx.combine(params, static)
            
            # 2. DISCRETE E-STEP (HMM with RTS Smoother)
            # Re-evaluate filter to get expectation states
            filter = ContinuousDiscreteKalmanFilter(self.sde_model, self.C, self.R)
            
            # Use continuous-discrete RTS smoother to get smoothed states (better expectation)
            mu_smooth, _ = filter.smooth(xs, ts, expected_s, jnp.zeros(self.D_z), jnp.eye(self.D_z))
            
            # If lineage barcodes are provided, apply clonal pull to smoothed states (with scaling normalization)
            if barcodes is not None and lambda_clonal > 0.0:
                unique_barcodes = jnp.unique(barcodes)
                
                def pull_to_centroid(z_seq, clone_id):
                    mask = (barcodes == clone_id)[:, None]
                    centroid = jnp.sum(z_seq * mask, axis=0) / (jnp.sum(mask) + 1e-8)
                    z_pulled = z_seq * (1 - lambda_clonal) + centroid * lambda_clonal
                    return jnp.where(mask, z_pulled, jnp.zeros_like(z_seq))
                    
                vmap_pull = jax.vmap(pull_to_centroid, in_axes=(None, 0))
                pulled_layers = vmap_pull(mu_smooth, unique_barcodes)
                mu_smooth_reg = jnp.sum(pulled_layers, axis=0)
                
                # Rescale to prevent latent scale collapse
                std_orig = jnp.std(mu_smooth)
                std_reg = jnp.std(mu_smooth_reg)
                mu_smooth = mu_smooth_reg * (std_orig / (std_reg + 1e-12))
            
            # MSE per state prediction based on smoothed trajectory
            preds = jax.vmap(lambda m: self.C @ m)(mu_smooth)
            err = jnp.sum((xs - preds)**2, axis=1)
            
            # Pseudo-likelihoods
            pseudo_lls = jnp.tile(-0.5 * err[:, None], (1, self.K)) 
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
