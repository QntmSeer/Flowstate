import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

class SwitchableDrift(eqx.Module):
    """
    Parametrizes the continuous drift vector field f(z, s).
    For each discrete state s (k in 1..K), we have a different linear or non-linear drift.
    Currently implements a linear drift per state: f(z, k) = A_k z + b_k
    """
    A_s: jnp.ndarray
    b_s: jnp.ndarray

    def __init__(self, K, D_z, key):
        keys = jax.random.split(key, 2)
        # Random initializations near stable origins
        self.A_s = jax.random.normal(keys[0], (K, D_z, D_z)) * 0.1 - jnp.eye(D_z) * 0.5
        self.b_s = jax.random.normal(keys[1], (K, D_z)) * 0.1

    def __call__(self, t, z, args):
        """
        args contains the current active discrete program (one-hot encoded or integer)
        For differentiability and expected statistics, we can take a softmax-weighted sum
        over the state drifts if `expected_s` is passed instead of a hard assignment.
        """
        s_prob = args  # Expecting shape (K,) probabilities
        
        # Drift if in state k: A_k @ z + b_k
        def per_state_drift(k):
            return self.A_s[k] @ z + self.b_s[k]
        
        # Weighted mixture of drifts based on state probabilities
        drifts = jax.vmap(per_state_drift)(jnp.arange(self.A_s.shape[0]))
        expected_drift = jnp.average(drifts, weights=s_prob, axis=0)
        
        return expected_drift

class SwitchableDiffusion(eqx.Module):
    """
    Parametrizes the stochastic diffusion matrix g(z, s).
    Implements constant within-state noise Q_k.
    """
    Q_s_chol: jnp.ndarray # Cholesky factor for positive semi-definiteness
    
    def __init__(self, K, D_z, key):
        # Initialize as diagonal process noise
        self.Q_s_chol = jax.random.uniform(key, (K, D_z)) * 0.1 + 0.1

    def __call__(self, t, z, args):
        s_prob = args # (K,)
        # Diagonal noise scaled by state probabilities
        def per_state_diffusion(k):
            return jnp.diag(self.Q_s_chol[k])
            
        diffusions = jax.vmap(per_state_diffusion)(jnp.arange(self.Q_s_chol.shape[0]))
        # We square the weights because they are variances? Actually, for linear expected value:
        expected_diff = jnp.average(diffusions, weights=s_prob, axis=0)
        
        return expected_diff

class NeuralSLDS(eqx.Module):
    """
    A full Continuous-Time Switching Linear Dynamical System using diffrax.
    """
    drift: SwitchableDrift
    diffusion: SwitchableDiffusion
    K: int
    D_z: int

    def __init__(self, K, D_z, key):
        k1, k2 = jax.random.split(key)
        self.K = K
        self.D_z = D_z
        self.drift = SwitchableDrift(K, D_z, k1)
        self.diffusion = SwitchableDiffusion(K, D_z, k2)

    def simulate_path(self, z0, ts, s_path, key):
        """
        Simulate a single rigorous forward path using SDE solvers.
        ts: array of timepoints to evaluate
        s_path: function s(t) that returns the state probabilities at any time t.
        """
        # Define the SDE terms
        def drift_func(t, y, args):
            # Evaluate the discrete state path at time t
            s_t = s_path(t) 
            return self.drift(t, y, s_t)
            
        def diffusion_func(t, y, args):
            s_t = s_path(t)
            return self.diffusion(t, y, s_t)

        drift = diffrax.ODETerm(drift_func)
        diffusion = diffrax.ControlTerm(diffusion_func, diffrax.VirtualBrownianTree(ts[0], ts[-1], tol=1e-3, shape=(self.D_z,), key=key))
        terms = diffrax.MultiTerm(drift, diffusion)

        # Solve it using Euler-Maruyama (standard for SDEs)
        solver = diffrax.Euler()
        dt0 = 0.05
        
        saveat = diffrax.SaveAt(ts=ts)
        
        sol = diffrax.diffeqsolve(
            terms, solver, ts[0], ts[-1], dt0, z0, saveat=saveat
        )
        
        return sol.ys
