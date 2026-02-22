import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from stateflow.sde_core import NeuralSLDS
from stateflow.sde_inference import ContinuousDiscreteKalmanFilter

def test_sde_inference():
    print("Testing Continuous-Discrete Kalman Filter...")
    
    K = 2
    D_z = 2
    D_x = 5
    T = 100
    
    key = jax.random.PRNGKey(42)
    sde_model = NeuralSLDS(K, D_z, key)
    
    # Let's create an observation matrix C and R
    C = jnp.eye(D_x, D_z) * np.array([1.0, 1.0, 0.0, 0.0, 0.0])[:, None] # Only observe first 2 dims
    C = jax.random.normal(key, (D_x, D_z))
    R = jnp.eye(D_x) * 0.1
    
    # 1. Generate some continuous truth trajectory
    def s_path(t):
        p0 = jnp.where(t < 5.0, 1.0, 0.0)
        p1 = jnp.where(t >= 5.0, 1.0, 0.0)
        return jnp.array([p0, p1])

    ts = jnp.linspace(0.0, 10.0, T)
    z0 = jnp.array([2.0, -2.0])
    
    print("Simulating ground truth SDE...")
    z_true = sde_model.simulate_path(z0, ts, s_path, key)
    
    # 2. Add emission noise to generate observations
    x_obs = jax.vmap(lambda z: C @ z)(z_true)
    x_obs += jax.random.normal(jax.random.PRNGKey(123), x_obs.shape) * jnp.sqrt(0.1)
    
    # 3. Filter with our Continuous Discrete Filter
    print("Running Filter over discrete observations...")
    expected_s = jax.vmap(s_path)(ts) # We assume discrete states are known for testing the filter
    
    filter = ContinuousDiscreteKalmanFilter(sde_model, C, R)
    mu_0 = jnp.zeros(D_z)
    P_0 = jnp.eye(D_z)
    
    # Need to vmap / scan step
    # Wait, the class filter() function is missing import numpy as np and jax.lax inside logic.
    # We will test the class directly.
    mu_filt, P_filt = filter.filter(x_obs, ts, expected_s, mu_0, P_0)
    
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # Plot true
    plt.plot(ts, z_true[:, 0], 'k-', label='True z_0')
    plt.plot(ts, z_true[:, 1], 'k--', label='True z_1')
    
    # Plot Filtered
    plt.plot(ts, mu_filt[:, 0], 'r-', alpha=0.8, label='Filtered mu_0')
    plt.plot(ts, mu_filt[:, 1], 'b--', alpha=0.8, label='Filtered mu_1')
    
    # Uncertainty
    std_0 = jnp.sqrt(P_filt[:, 0, 0])
    plt.fill_between(ts, mu_filt[:, 0] - 2*std_0, mu_filt[:, 0] + 2*std_0, color='r', alpha=0.2)
    
    std_1 = jnp.sqrt(P_filt[:, 1, 1])
    plt.fill_between(ts, mu_filt[:, 1] - 2*std_1, mu_filt[:, 1] + 2*std_1, color='b', alpha=0.2)
    
    plt.axvline(5.0, color='gray', linestyle=':', label='State Switch')
    plt.legend()
    plt.title("Continuous-Discrete Kalman Filter on Neural SDE")
    plt.xlabel("Continuous Pseudotime (t)")
    plt.ylabel("Latent State (z)")
    
    plt.tight_layout()
    plt.savefig('test_sde_inference.png')
    print("Saved test_sde_inference.png")

if __name__ == "__main__":
    import numpy as np
    test_sde_inference()
