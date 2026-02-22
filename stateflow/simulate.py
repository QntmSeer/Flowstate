import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from .core import StateFlowModel, SLDSParams

def simulate_trajectory(params: SLDSParams, T: int, key: jax.random.PRNGKey):
    """
    Simulates a single cell trajectory over T time steps.
    """
    keys = jax.random.split(key, T)
    
    def step_fn(carry, t_key):
        s_prev, z_prev = carry
        k_s, k_z, k_x = jax.random.split(t_key, 3)
        
        # 1. Discrete state transition
        # Sample next state s_t given s_{t-1} and transition matrix A
        p_trans = params.A[s_prev]
        s_curr = jax.random.choice(k_s, params.A.shape[0], p=p_trans)
        
        # 2. Continuous state evolution
        # z_t = A_s * z_{t-1} + noise
        mean_z = params.A_s[s_curr] @ z_prev
        cov_z = params.Q_s[s_curr]
        # Equivalent to custom multivariate normal sample
        L = jnp.linalg.cholesky(cov_z)
        z_curr = mean_z + L @ jax.random.normal(k_z, shape=(z_prev.shape[0],))
        
        # 3. Observation emission
        # x_t = C * z_t + noise
        mean_x = params.C @ z_curr
        L_r = jnp.linalg.cholesky(params.R)
        x_curr = mean_x + L_r @ jax.random.normal(k_x, shape=(mean_x.shape[0],))
        
        carry_next = (s_curr, z_curr)
        output = (s_curr, z_curr, x_curr)
        return carry_next, output

    # Initialization
    k_init_s, k_init_z = jax.random.split(keys[0], 2)
    s_0 = jax.random.choice(k_init_s, params.pi.shape[0], p=params.pi)
    L_0 = jnp.linalg.cholesky(params.Sigma_0[s_0])
    z_0 = params.mu_0[s_0] + L_0 @ jax.random.normal(k_init_z, shape=(params.mu_0.shape[1],))
    
    # Run scan
    _, (s_traj, z_traj, x_traj) = jax.lax.scan(step_fn, (s_0, z_0), keys[1:])
    
    # Prepend initial state
    # Actually, we usually just care about the output from t=1 to T, but let's just use the scan output.
    return s_traj, z_traj, x_traj

def plot_synthetic_data(s_traj, z_traj, x_traj):
    """
    Plot the generated states and emissions to verify correctness.
    """
    T = s_traj.shape[0]
    time = jnp.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2, 3]})
    
    # Plot 1: Discrete States
    axes[0].step(time, s_traj, where='post', color='black')
    axes[0].set_yticks(jnp.unique(s_traj))
    axes[0].set_ylabel('Markov State ($s_t$)')
    axes[0].set_title('Discrete State Transitions')
    
    # Plot 2: Latent trajectory (first 3 dims)
    for i in range(min(3, z_traj.shape[1])):
        axes[1].plot(time, z_traj[:, i], label=f'Latent Dim {i}')
    axes[1].set_ylabel('Latent Value ($z_t$)')
    axes[1].legend()
    axes[1].set_title('Continuous Process Evolution')
    
    # Plot 3: Gene Expression Heatmap
    sns.heatmap(x_traj.T, ax=axes[2], cmap='viridis', cbar=True)
    axes[2].set_ylabel('Genes ($x_t$)')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Simulated Gene Expression Emission')
    
    plt.tight_layout()
    plt.savefig('synthetic_trajectory.png')
    print("Saved 'synthetic_trajectory.png'")

if __name__ == "__main__":
    K = 3      # 3 distinct cellular programs
    D_z = 5    # 5 latent regulatory pathways
    D_x = 50   # 50 observed genes
    T = 200    # 200 time steps
    
    print(f"Initializing SLDS Model (K={K}, D_z={D_z}, D_x={D_x})...")
    model = StateFlowModel(K=K, D_z=D_z, D_x=D_x, seed=123)
    
    print(f"Simulating a trajectory of length {T}...")
    s_traj, z_traj, x_traj = simulate_trajectory(model.params, T, jax.random.PRNGKey(42))
    
    print("Plotting synthetic data...")
    plot_synthetic_data(s_traj, z_traj, x_traj)
    print("Done!")
