import jax
import jax.numpy as jnp
from typing import NamedTuple

class SLDSParams(NamedTuple):
    """
    Parameters for a single-cell Linear Gaussian Switching Latent Dynamical System.
    
    K: number of discrete states (e.g., cell types or transcriptional programs)
    D_z: dimensionality of the continuous latent space (e.g., low-dim regulatory factors)
    D_x: dimensionality of the observed space (e.g., gene expression)
    """
    pi: jnp.ndarray       # (K,) Initial discrete state probabilities
    A: jnp.ndarray        # (K, K) Markov transition matrix (row-stochastic)
    A_s: jnp.ndarray      # (K, D_z, D_z) State-dependent continuous dynamics matrices
    Q_s: jnp.ndarray      # (K, D_z, D_z) State-dependent process noise covariances
    C: jnp.ndarray        # (D_x, D_z) Emission mapping from latent to observed
    R: jnp.ndarray        # (D_x, D_x) Emission noise covariance
    mu_0: jnp.ndarray     # (K, D_z) Initial continuous state mean (per state)
    Sigma_0: jnp.ndarray  # (K, D_z, D_z) Initial continuous state covariance (per state)


class StateFlowModel:
    """
    Core mathematical framework for modeling cellular dynamics using an SLDS.
    """
    def __init__(self, K: int, D_z: int, D_x: int, seed: int = 42):
        self.K = K
        self.D_z = D_z
        self.D_x = D_x
        self.key = jax.random.PRNGKey(seed)
        self.params = self._initialize_params()
        
    def _initialize_params(self) -> SLDSParams:
        """
        Initializes the SLDS parameters with plausible biological priors for synthesis.
        For example:
        - Metastable transition matrices (high diagonal)
        - Stable continuous dynamics (eigenvalues < 1)
        """
        k1, k2, k3, k4, self.key = jax.random.split(self.key, 5)
        
        # 1. Initial State Dist (Assume most cells start in state 0, e.g., 'Stem')
        pi = jnp.zeros(self.K)
        pi = pi.at[0].set(0.9)
        pi = pi.at[1:].set(0.1 / (self.K - 1))
        
        # 2. Transition Matrix (Metastable states: high probability of staying)
        A_raw = jax.random.dirichlet(k1, alpha=jnp.ones(self.K), shape=(self.K,))
        A_raw = A_raw + jnp.eye(self.K) * 5.0 # Boost self-transition
        A = A_raw / jnp.sum(A_raw, axis=1, keepdims=True)
        
        # 3. Continuous Dynamics (Stable or slightly contracting)
        # Random matrices, scaled down to ensure stability (eigenvalues < 1)
        A_s_raw = jax.random.normal(k2, shape=(self.K, self.D_z, self.D_z))
        A_s = A_s_raw / jnp.linalg.norm(A_s_raw, axis=(1, 2), keepdims=True) * 0.9 + 0.05 * jnp.eye(self.D_z)
        
        # 4. Process Noise (State-dependent stochasticity)
        Q_s = jnp.zeros((self.K, self.D_z, self.D_z))
        for i in range(self.K):
            k3, sub_k = jax.random.split(k3)
            L = jax.random.normal(sub_k, shape=(self.D_z, self.D_z)) * 0.1
            Q_s = Q_s.at[i].set(L @ L.T + 1e-3 * jnp.eye(self.D_z)) # Ensure PSD
            
        # 5. Emission Matrix (Mapping from regulatory latent space to gene expression space)
        C = jax.random.normal(k4, shape=(self.D_x, self.D_z)) * 0.5
        
        # 6. Emission Noise (Diagonal covariance for independent gene noise)
        R = jnp.eye(self.D_x) * 0.1
        
        # 7. Initial Continuous State (Centered at 0)
        mu_0 = jnp.zeros((self.K, self.D_z))
        
        # 8. Initial Continuous Covariance (Identity)
        Sigma_0 = jnp.tile(jnp.eye(self.D_z), (self.K, 1, 1))

        return SLDSParams(
            pi=pi, A=A, A_s=A_s, Q_s=Q_s, C=C, R=R, mu_0=mu_0, Sigma_0=Sigma_0
        )
