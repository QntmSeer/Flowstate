import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

class SwitchableDrift(eqx.Module):
    """
    Parametrizes the continuous drift vector field f(z, s).
    For each discrete state s (k in 1..K), we have a different non-linear drift.
    Implements an Equinox MLP per state.
    """
    mlps: tuple
    K: int
    D_z: int

    def __init__(self, K, D_z, key):
        self.K = K
        self.D_z = D_z
        keys = jax.random.split(key, K)
        # Initialize an MLP for each of the K states.
        # We use a smooth activation function (softplus) so derivatives/Jacobians are stable.
        self.mlps = tuple([
            eqx.nn.MLP(
                in_size=D_z,
                out_size=D_z,
                width_size=64,
                depth=2,
                activation=jax.nn.softplus,
                key=keys[k]
            )
            for k in range(K)
        ])

    def __call__(self, t, z, args):
        """
        args contains the current active discrete program probabilities (K,)
        """
        s_prob = args  # Expecting shape (K,) probabilities
        
        # Evaluate each state-specific MLP on z
        drifts = jnp.stack([mlp(z) for mlp in self.mlps])  # (K, D_z)
        expected_drift = jnp.average(drifts, weights=s_prob, axis=0)
        
        return expected_drift

    def get_jacobians(self, z):
        """
        Computes the Jacobian matrices df_k/dz at z for all states k.
        Returns: (K, D_z, D_z)
        """
        jacs = jnp.stack([jax.jacfwd(mlp)(z) for mlp in self.mlps])
        return jacs


class SwitchableDiffusion(eqx.Module):
    """
    Parametrizes the stochastic diffusion matrix g(z, s) using MLPs.
    Outputs a diagonal noise matrix where diagonals are strictly positive.
    """
    mlps: tuple
    K: int
    D_z: int
    
    def __init__(self, K, D_z, key):
        self.K = K
        self.D_z = D_z
        keys = jax.random.split(key, K)
        # Initialize an MLP for each of the K states.
        # The output represents the diagonal of the diffusion matrix.
        self.mlps = tuple([
            eqx.nn.MLP(
                in_size=D_z,
                out_size=D_z,
                width_size=64,
                depth=2,
                activation=jax.nn.softplus,
                key=keys[k]
            )
            for k in range(K)
        ])

    def __call__(self, t, z, args):
        s_prob = args # (K,)
        
        # Evaluate each state-specific diffusion MLP on z using static list comprehension
        diags = jnp.stack([jax.nn.softplus(mlp(z)) for mlp in self.mlps]) # (K, D_z)
        diffusions = jax.vmap(jnp.diag)(diags) # (K, D_z, D_z)
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
