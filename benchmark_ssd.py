import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from stateflow.sde_core import NeuralSLDS
from stateflow.sde_inference import ContinuousDiscreteKalmanFilter, SpeculativeSpeculativeKalmanFilter
import equinox as eqx

def generate_benchmark_data(T=200, D_z=2, D_x=5, seed=123):
    """
    Simulates a ground-truth trajectory using the Nonlinear Neural SDE model
    and adds measurement noise to create observations.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    
    sde_model = NeuralSLDS(K=2, D_z=D_z, key=k1)
    
    # Emission mapping and observation noise
    C = jax.random.normal(k2, (D_x, D_z)) * 0.5
    R = jnp.eye(D_x) * 0.1
    
    # Setup timepoints and discrete states
    ts = jnp.sort(jax.random.uniform(k3, (T,)) * 10.0)
    
    def s_path(t):
        p0 = jnp.where(t < 5.0, 1.0, 0.0)
        p1 = jnp.where(t >= 5.0, 1.0, 0.0)
        return jnp.array([p0, p1])
        
    expected_s = jax.vmap(s_path)(ts)
    
    # Simulate ground-truth continuous path
    z_true = sde_model.simulate_path(jnp.array([1.0, -1.0]), ts, s_path, jax.random.PRNGKey(888))
    
    # Emit noisy observations
    x_obs = jax.vmap(lambda z: C @ z)(z_true)
    x_obs += jax.random.normal(jax.random.PRNGKey(777), x_obs.shape) * jnp.sqrt(0.1)
    
    return sde_model, C, R, x_obs, ts, expected_s, z_true

def main():
    print("=== STARTING SSD-SDE BENCHMARK ===")
    T = 200
    D_z = 2
    D_x = 5
    
    sde_model, C, R, x_obs, ts, expected_s, z_true = generate_benchmark_data(T, D_z, D_x)
    mu_0 = jnp.zeros(D_z)
    P_0 = jnp.eye(D_z)
    
    # Initialize both filters
    # We set tolerance to 0.1 for speculative acceptance threshold
    std_filter = ContinuousDiscreteKalmanFilter(sde_model, C, R)
    ssd_filter = SpeculativeSpeculativeKalmanFilter(sde_model, C, R, tolerance=0.1)
    
    # Define standalone filter functions decorated with eqx.filter_jit
    @eqx.filter_jit
    def run_std_filter(f_obj, x, t, s, m0, p0):
        return f_obj.filter(x, t, s, m0, p0)
        
    @eqx.filter_jit
    def run_ssd_filter(f_obj, x, t, s, m0, p0):
        return f_obj.filter(x, t, s, m0, p0)
    
    # JIT-compile the filters to ensure we are benchmarking compiled performance
    print("Compiling Standard Filter...")
    t_start = time.time()
    mu_std, P_std = run_std_filter(std_filter, x_obs, ts, expected_s, mu_0, P_0)
    jax.block_until_ready((mu_std, P_std))
    print(f"Standard Filter Compiled in: {time.time() - t_start:.2f} seconds")
    
    print("Compiling Speculative Speculative Filter...")
    t_start = time.time()
    mu_ssd, P_ssd, accepts = run_ssd_filter(ssd_filter, x_obs, ts, expected_s, mu_0, P_0)
    jax.block_until_ready((mu_ssd, P_ssd, accepts))
    print(f"Speculative Speculative Filter Compiled in: {time.time() - t_start:.2f} seconds")
    
    # Benchmark 1: Latency of Standard Filter
    runs = 100
    print(f"Running Standard Filter benchmark ({runs} runs)...")
    t_start = time.time()
    for _ in range(runs):
        _ = run_std_filter(std_filter, x_obs, ts, expected_s, mu_0, P_0)
    jax.block_until_ready(_)
    time_std = (time.time() - t_start) / runs * 1000  # Convert to milliseconds
    print(f"Standard Filter Mean Latency: {time_std:.3f} ms")
    
    # Benchmark 2: Latency of Speculative Speculative Filter
    print(f"Running Speculative Speculative Filter benchmark ({runs} runs)...")
    t_start = time.time()
    for _ in range(runs):
        _ = run_ssd_filter(ssd_filter, x_obs, ts, expected_s, mu_0, P_0)
    jax.block_until_ready(_)
    time_ssd = (time.time() - t_start) / runs * 1000  # Convert to milliseconds
    print(f"Speculative Speculative Filter Mean Latency: {time_ssd:.3f} ms")
    
    # Benchmark 3: Accuracy comparison (RMSE)
    rmse_std = jnp.sqrt(jnp.mean((z_true - mu_std)**2))
    rmse_ssd = jnp.sqrt(jnp.mean((z_true - mu_ssd)**2))
    accept_rate = jnp.mean(accepts) * 100.0
    
    print("\n=== BENCHMARK RESULTS SUMMARY ===")
    print(f"Standard EKF Mean Latency : {time_std:.3f} ms")
    print(f"SSD-EKF Mean Latency      : {time_ssd:.3f} ms")
    speedup = (time_std - time_ssd) / time_std * 100.0
    print(f"Latency Reduction (Speedup): {speedup:.1f}%")
    print(f"Standard EKF Trajectory RMSE: {rmse_std:.4f}")
    print(f"SSD-EKF Trajectory RMSE     : {rmse_ssd:.4f}")
    print(f"SSD Speculation Acceptance  : {accept_rate:.1f}%")
    print("==================================\n")
    
    # Plotting comparison
    print("Generating performance plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Latency and Accuracy Comparison
    labels = ['Standard CD-EKF', 'Speculative (SSD-EKF)']
    latencies = [time_std, time_ssd]
    
    bars = axes[0].bar(labels, latencies, color=['#1f77b4', '#d62728'], width=0.5)
    axes[0].set_ylabel('Execution Latency (ms)', fontsize=12)
    axes[0].set_title('Inference Speed Comparison', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
    # Add text box for metrics
    metrics_text = (
        f"Speedup: {speedup:.1f}%\n"
        f"Acceptance Rate: {accept_rate:.1f}%\n"
        f"Standard RMSE: {rmse_std:.4f}\n"
        f"SSD-EKF RMSE: {rmse_ssd:.4f}"
    )
    axes[0].text(0.05, 0.05, metrics_text, transform=axes[0].transAxes, fontsize=12,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
    # Plot 2: Latent Coordinate Trajectories comparison (Dim 0)
    axes[1].plot(ts, z_true[:, 0], 'k-', linewidth=2.5, label='Ground Truth $z_0$')
    axes[1].plot(ts, mu_std[:, 0], 'b--', linewidth=2.0, label='Standard CD-EKF')
    axes[1].plot(ts, mu_ssd[:, 0], 'r:', linewidth=2.0, label='Speculative SSD-EKF')
    
    # Highlight accepted vs rejected steps
    for idx in range(1, len(ts)):
        if accepts[idx]:
            # Light green background for accepted speculative steps
            axes[1].axvspan(ts[idx-1], ts[idx], color='lightgreen', alpha=0.15)
        else:
            # Light red background for rejected speculative steps
            axes[1].axvspan(ts[idx-1], ts[idx], color='salmon', alpha=0.1)
            
    axes[1].set_title('Trajectory Reconstruction & Speculation Verification\n(Green = Speculation Accepted, Red = Fallback)', fontsize=14)
    axes[1].set_xlabel('Time (t)', fontsize=12)
    axes[1].set_ylabel('Latent Value $z_0$', fontsize=12)
    axes[1].legend(loc='upper left', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/benchmark_ssd.png', dpi=150)
    print("Saved assets/benchmark_ssd.png successfully!")

if __name__ == "__main__":
    main()
