import jax
import jax.numpy as jnp
from stateflow.core import SLDSParams
from stateflow.simulate import simulate_trajectory
import matplotlib.pyplot as plt

def generate_bifurcating_topology(T=500, D_z=3, D_x=20, seed=42):
    """
    Generates a synthetic dataset with a strict Bifurcating Topology:
    State 0 (Stem) -> transitions to -> State 1 (Fate A)
                                     -> State 2 (Fate B)
    State 1 -> Absorbing
    State 2 -> Absorbing
    """
    key = jax.random.PRNGKey(seed)
    # K=3 for Bifurcation (Root + 2 branches)
    K = 3
    
    # 1. Markov Transition Matrix (A)
    # State 0 goes to 1 or 2. States 1 and 2 are absorbing (1.0 self-transition).
    A = jnp.array([
        [0.85, 0.075, 0.075], # Stem cell slowly differentiates into A or B
        [0.0,  1.0,   0.0],   # Fate A is terminal
        [0.0,  0.0,   1.0]    # Fate B is terminal
    ])
    
    # Intial state must be Stem
    pi = jnp.array([1.0, 0.0, 0.0])
    
    # 2. Continuous parameters
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # A_s: Continuous dynamics. We make them stable attractors (eigs < 1)
    # Stem: stays near origin
    # Fate A: drifts towards an attractor in +z direction
    # Fate B: drifts towards an attractor in -z direction
    A_s = jnp.array([
        0.9 * jnp.eye(D_z),
        0.5 * jnp.eye(D_z) + jax.random.normal(k1, (D_z, D_z))*0.1,
        0.5 * jnp.eye(D_z) + jax.random.normal(k2, (D_z, D_z))*0.1
    ])
    
    # We will encode the actual drift target in the emission or mean, but SLDS handles this 
    # via the process noise / initial mean distributions usually. 
    # For a true bifurcation in SLDS, the emission matrix C maps the continuous dimensions 
    # to gene space. As long as V_z has distinct dynamics, PCA will show the branch.
    Q_s = jnp.tile(jnp.eye(D_z)*0.05, (K, 1, 1))
    
    C = jax.random.normal(k3, (D_x, D_z))
    R = jnp.eye(D_x) * 0.1
    
    # To ensure distinct branches, let's offset the mean of the continuous states initially
    mu_0 = jnp.array([
        [0.0, 0.0, 0.0],   # Stem starts at origin
        [5.0, 5.0, 0.0],   # Fate A basin
        [-5.0, -5.0, 0.0]  # Fate B basin
    ])
    if D_z != 3:
        mu_0 = jnp.zeros((K, D_z))
        mu_0 = mu_0.at[1, 0:2].set(5.0)
        mu_0 = mu_0.at[2, 0:2].set(-5.0)
        
    Sigma_0 = jnp.tile(jnp.eye(D_z), (K, 1, 1))
    
    params = SLDSParams(pi, A, A_s, Q_s, C, R, mu_0, Sigma_0)
    
    # Simulate
    k_sim = jax.random.PRNGKey(seed + 1)
    s_traj, z_traj, x_traj = simulate_trajectory(params, T, k_sim)
    
    return params, s_traj, z_traj, x_traj

if __name__ == "__main__":
    params, s, z, x = generate_bifurcating_topology()
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=s, cmap='viridis')
    plt.colorbar(scatter, label='True Discrete State (0=Stem, 1=FateA, 2=FateB)')
    plt.title("Ground Truth Bifurcating Topology (PCA of Genes)")
    plt.savefig('benchmark_bifurcation_gt.png')
    print("Saved ground truth bifurcation plot.")
