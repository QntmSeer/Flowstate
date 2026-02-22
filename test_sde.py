import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
from stateflow.sde_core import NeuralSLDS

def test_sde_forward():
    print("Initializing Neural SDE...")
    K = 3
    D_z = 2
    key = jax.random.PRNGKey(42)
    sde_model = NeuralSLDS(K, D_z, key)
    
    # Let's create a synthetic discrete path s_path(t)
    # E.g., at t=0 to 1, state is [1, 0, 0]
    #      at t=1 to 2, state is [0, 1, 0]
    #      at t=2 to 3, state is [0, 0, 1]
    def s_path(t):
        p0 = jnp.where(t < 1.0, 1.0, 0.0)
        p1 = jnp.where((t >= 1.0) & (t < 2.0), 1.0, 0.0)
        p2 = jnp.where(t >= 2.0, 1.0, 0.0)
        return jnp.array([p0, p1, p2])

    ts = jnp.linspace(0.0, 3.0, 300)
    z0 = jnp.array([1.0, 1.0])
    
    print("Simulating path...")
    
    # We need to structure the VirtualBrownianTree according to D_z
    def _run(key):
        def drift_func(t, y, args):
            return sde_model.drift(t, y, s_path(t))

        def diffusion_func(t, y, args):
            # return shape (D_z, D_z)
            return sde_model.diffusion(t, y, s_path(t))

        drift = diffrax.ODETerm(drift_func)
        diffusion = diffrax.ControlTerm(diffusion_func, diffrax.VirtualBrownianTree(ts[0], ts[-1], tol=1e-3, shape=(D_z,), key=key))
        terms = diffrax.MultiTerm(drift, diffusion)

        solver = diffrax.Euler()
        dt0 = 0.01

        sol = diffrax.diffeqsolve(
            terms, solver, ts[0], ts[-1], dt0, z0, saveat=diffrax.SaveAt(ts=ts)
        )
        return sol.ys
        
    ys = _run(jax.random.PRNGKey(123))
    
    # Also get multiple samples
    multi_run = jax.vmap(_run)(jax.random.split(jax.random.PRNGKey(999), 10))
    
    print("Plotting continuous time trajectory...")
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.plot(ts, multi_run[i, :, 0], alpha=0.3, color='blue')
        plt.plot(ts, multi_run[i, :, 1], alpha=0.3, color='red')
    
    plt.plot(ts, ys[:, 0], color='darkblue', linewidth=2, label='Dim 0')
    plt.plot(ts, ys[:, 1], color='darkred', linewidth=2, label='Dim 1')
    
    # Draw state boundaries
    plt.axvline(1.0, color='black', linestyle='--', label='State 0 -> State 1')
    plt.axvline(2.0, color='black', linestyle=':', label='State 1 -> State 2')
    
    plt.legend()
    plt.title("Neural SDE Forward Simulation (Switching Drifts)")
    plt.xlabel("Time (t)")
    plt.ylabel("Latent Gene Status $z_t$")
    plt.tight_layout()
    plt.savefig("test_sde.png")
    print("Saved test_sde.png")

if __name__ == "__main__":
    test_sde_forward()
