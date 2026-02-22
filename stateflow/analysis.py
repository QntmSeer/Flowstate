import jax
import jax.numpy as jnp
from .core import SLDSParams

class StabilityAnalyzer:
    """
    Mathematical tools for extracting biological interpretations from a fitted SLDS model.
    """
    def __init__(self, params: SLDSParams):
        self.params = params
        self.K = params.pi.shape[0]
        self.D_z = params.mu_0.shape[1]

    def compute_stationary_distribution(self):
        """
        Solves for the stationary distribution pi_inf of the Markov transition matrix A.
        pi_inf = pi_inf * A  =>  pi_inf * (I - A) = 0
        """
        A = self.params.A
        # Compute eigenvalues and left eigenvectors
        eigenvalues, eigenvectors = jnp.linalg.eig(A.T)
        
        # Find the eigenvector corresponding to eigenvalue 1 (index where lambda-1 is min)
        idx = jnp.argmin(jnp.abs(eigenvalues - 1.0))
        
        # Stationary distribution is the real part of this eigenvector, normalized to sum to 1
        pi_inf = jnp.real(eigenvectors[:, idx])
        pi_inf = pi_inf / jnp.sum(pi_inf)
        
        return pi_inf

    def compute_continuous_stability(self):
        """
        Analyzes the eigenvalues of the state-dependent continuous dynamics matrices A_s.
        - If all eigenvalues < 1, the state is a stable biological attractor.
        - If any eigenvalue > 1, the state is unstable/transient.
        
        Returns:
            eigenvalues: (K, D_z) complex eigenvalues for each state
            max_eigenvalue_magnitudes: (K,) the spectral radius (max magnitude) for each state
        """
        A_s = self.params.A_s
        
        def get_eigs(A_k):
            eigs = jnp.linalg.eigvals(A_k)
            return eigs
            
        eigenvalues = jax.vmap(get_eigs)(A_s)
        magnitudes = jnp.abs(eigenvalues)
        max_magnitudes = jnp.max(magnitudes, axis=1)
        
        return eigenvalues, max_magnitudes

    def interpret_model(self):
        """
        Prints out the biological interpretation of the fitted model parameters.
        """
        pi_inf = self.compute_stationary_distribution()
        eigs, max_mags = self.compute_continuous_stability()
        
        print("\n=== Model Biological Interpretation ===")
        print("1. Long-Term Population Fate (Stationary Distribution)")
        print(f"pi_inf: {pi_inf}")

        print("\n2. Metastable State Stability (Attractor Basin Analysis)")
        for k in range(self.K):
            status = "Stable Attractor" if max_mags[k] < 1.0 else "Unstable/Transient"
            print(f"State {k} (Spectral Radius: {max_mags[k]:.3f}) -> {status}")
            
        print("\n3. Markov Transition Matrix")
        # Print rounded matrix for readable output
        with jnp.printoptions(precision=3, suppress=True):
            print(self.params.A)
        print("=======================================\n")
