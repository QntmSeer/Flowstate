import jax
import jax.numpy as jnp
from stateflow.core import StateFlowModel, SLDSParams
from stateflow.simulate import simulate_trajectory
from stateflow.inference import VariationalEM
from stateflow.analysis import StabilityAnalyzer
import matplotlib.pyplot as plt

def main():
    # 1. Initialize True Model & Simulate
    print("Setting up simulation...")
    K, D_z, D_x = 3, 5, 50
    T = 200
    
    true_model = StateFlowModel(K=K, D_z=D_z, D_x=D_x, seed=42)
    s_traj, z_traj, x_traj = simulate_trajectory(true_model.params, T, jax.random.PRNGKey(123))
    
    # 2. Setup Inference Model with exact true parameters initially just to test continuous E-step
    print("Testing continuous E-step (Kalman Smoother)...")
    vem = VariationalEM(true_model.params)
    
    # Normally we don't know the true states, but let's provide the exact true states as expected_s
    # essentially doing a one-hot encoding for the expected states
    expected_s = jax.nn.one_hot(s_traj, K)
    
    # JIT compile the continuous E-step to ensure no JAX shape/type errors
    e_step_jit = jax.jit(vem.e_step_continuous)
    
    print("Running Kalman Smoother...")
    mu_z, V_z, V_z_cross = e_step_jit(x_traj, expected_s)
    
    T_actual = x_traj.shape[0]
    print(f"mu_z shape: {mu_z.shape} (Expected: {T_actual, D_z})")
    print(f"V_z shape: {V_z.shape} (Expected: {T_actual, D_z, D_z})")
    print(f"V_z_cross shape: {V_z_cross.shape} (Expected: {T_actual-1, D_z, D_z})")
    
    # Plot real z_traj vs inferred mu_z
    fig, axes = plt.subplots(D_z, 1, figsize=(10, 10), sharex=True)
    for i in range(D_z):
        axes[i].plot(z_traj[:, i], label='True z', color='black', alpha=0.6)
        axes[i].plot(mu_z[:, i], label='Smoothed mu', color='red', linestyle='--')
        
        # Add 95% confidence intervals
        std_z = jnp.sqrt(V_z[:, i, i])
        axes[i].fill_between(jnp.arange(T_actual), mu_z[:, i] - 2*std_z, mu_z[:, i] + 2*std_z, color='red', alpha=0.2)
        
        axes[i].set_ylabel(f'Dim {i}')
        if i == 0:
            axes[i].legend()
            axes[i].set_title('True Latent vs Smoothed Latent Trajectory')
            
    plt.tight_layout()
    plt.savefig('test_kalman_smoother.png')
    print("Saved test_kalman_smoother.png")
    plt.close()

    # 3. Test Discrete E-step (HMM Forward-Backward)
    print("Testing discrete E-step (HMM Forward-Backward)...")
    e_step_discrete_jit = jax.jit(vem.e_step_discrete)
    
    print("Running Forward-Backward Smoother...")
    expected_z_stats = (mu_z, V_z, V_z_cross)
    gamma, xi = e_step_discrete_jit(expected_z_stats)
    
    print(f"gamma shape: {gamma.shape} (Expected: {T_actual, K})")
    print(f"xi shape: {xi.shape} (Expected: {T_actual-1, K, K})")
    
    # Plot true states vs inferred probabilities
    fig, axes = plt.subplots(K, 1, figsize=(10, 6), sharex=True)
    for k in range(K):
        # True state dummy indicator
        true_k = (s_traj == k).astype(float)
        axes[k].fill_between(jnp.arange(T_actual), 0, true_k, color='black', alpha=0.3, label='True State')
        axes[k].plot(gamma[:, k], label=f'Inferred P(s_t = {k})', color='blue', linewidth=2)
        axes[k].set_ylabel(f'State {k}')
        if k == 0:
            axes[k].legend()
            axes[k].set_title('Inferred Discrete Program Probabilities (Gamma)')
    axes[-1].set_xlabel('Time Step')
            
    # 4. Test Full vEM Loop
    print("\n--- Testing Full Variational EM Loop (5 iterations) ---")
    
    # Initialize a new model with slightly perturbed true parameters to see if it learns
    # (Just taking the true model and adding noise to A, C)
    perturbed_params = SLDSParams(
        pi=true_model.params.pi,
        A=true_model.params.A * 0.9 + 0.1 * jnp.ones((K, K))/K, # add noise to A
        A_s=true_model.params.A_s,
        Q_s=true_model.params.Q_s,
        C=true_model.params.C + jax.random.normal(jax.random.PRNGKey(999), true_model.params.C.shape) * 0.1,
        R=true_model.params.R,
        mu_0=true_model.params.mu_0,
        Sigma_0=true_model.params.Sigma_0
    )
    
    vem_full = VariationalEM(perturbed_params)
    expected_s, mu_z_fit, V_z_fit = vem_full.fit(x_traj, max_iter=5)
    
    # Plot final fitted inferred probabilities
    fig, axes = plt.subplots(K, 1, figsize=(10, 6), sharex=True)
    for k in range(K):
        true_k = (s_traj == k).astype(float)
        axes[k].fill_between(jnp.arange(T_actual), 0, true_k, color='black', alpha=0.3, label='True State')
        axes[k].plot(expected_s[:, k], label=f'vEM P(s_t = {k})', color='green', linewidth=2)
        axes[k].set_ylabel(f'State {k}')
        if k == 0:
            axes[k].legend()
            axes[k].set_title('vEM Inferred Programs After 5 Iterations')
    axes[-1].set_xlabel('Time Step')
            
    plt.tight_layout()
    plt.savefig('test_vem_fit.png')
    print("Saved test_vem_fit.png")
    
    # 5. Extract Biological interpretation of the true underlying model
    analyzer = StabilityAnalyzer(true_model.params)
    analyzer.interpret_model()

if __name__ == "__main__":
    main()
