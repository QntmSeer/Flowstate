import scanpy as sc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx

# Import our package modules
from stateflow.sde_core import NeuralSLDS
from stateflow.sde_em import HybridSDEFit

def main():
    print("1. Loading Moignard 2015 Hematopoiesis qPCR Dataset...")
    adata = sc.datasets.moignard15()
    
    # Cast expression data to float32 and impute NaNs
    adata.X = adata.X.astype(np.float32)
    if np.isnan(adata.X).any():
        print("Dataset contains NaNs. Imputing with zeros...")
        adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    # Scale data to N(0, 1) for numerical stability
    sc.pp.scale(adata)
    
    # Filter genes with zero variance
    sc.pp.filter_genes(adata, min_cells=5)
    
    # Compute PCA
    print("Computing PCA...")
    try:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=5)
    except Exception:
        sc.tl.pca(adata, svd_solver='randomized', n_comps=5)
        
    # Order cells by PC1 as a proxy for developmental pseudotime
    time_order = np.argsort(adata.obsm['X_pca'][:, 0])
    
    # We will use the top 3 PCs as our observations (D_x = 3)
    x_obs = jnp.array(adata.obsm['X_pca'][time_order, :3])
    T_cells, D_x = x_obs.shape
    
    # Set time points as ordered PC1 coordinates (normalized to range [0, 10])
    ts = adata.obsm['X_pca'][time_order, 0]
    ts = (ts - ts.min()) / (ts.max() - ts.min()) * 10.0
    ts = jnp.array(ts)
    
    K = 5   # 5 metastable cell programs (representing finer developmental stages)
    D_z = 2 # 2D latent space to easily visualize the learned vector fields
    
    print(f"Initializing Neural SDE Model (K={K}, D_z={D_z}, D_x={D_x})...")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Set up model and initial parameters
    sde_model = NeuralSLDS(K, D_z, k1)
    
    # Initialize emission mapping C (3 -> 2) and noise R
    C = jax.random.normal(k2, (D_x, D_z)) * 0.5
    R = jnp.eye(D_x) * 0.5
    
    # Initialize fitter
    fitter = HybridSDEFit(sde_model, C, R, K, D_z)
    
    # Run EM + Optax gradient loop
    max_iter = 5
    print(f"Fitting Nonlinear Neural SDE (EM/Optax) for {max_iter} iterations...")
    expected_s, fitted_model = fitter.fit(x_obs, ts, max_iter=max_iter)
    
    # Get final inferred states
    inferred_states = np.argmax(expected_s, axis=1)
    
    print("Plotting results & learned vector fields...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: PCA of cell trajectories colored by inferred macro-states
    pca_ordered = adata.obsm['X_pca'][time_order]
    scatter = axes[0].scatter(pca_ordered[:, 0], pca_ordered[:, 1], c=inferred_states, cmap='viridis', alpha=0.7, s=15)
    axes[0].set_title("Moignard qPCR Trajectory\n(Colored by Inferred Macro-States)", fontsize=14)
    axes[0].set_xlabel("PC 1 (Pseudotime Proxy)", fontsize=12)
    axes[0].set_ylabel("PC 2", fontsize=12)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("Inferred State ID", fontsize=11)
    
    # Plot 2: Discovered Nonlinear Dynamical Vector Fields in Latent Space
    # Create 2D grid in latent space
    grid_lim = 2.0
    x_grid = np.linspace(-grid_lim, grid_lim, 20)
    y_grid = np.linspace(-grid_lim, grid_lim, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1) # (400, 2)
    grid_jnp = jnp.array(grid_points)
    
    # Evaluate drift vector field for each discrete state k
    # Use RGBA tuples for colors to set transparency (alpha=0.6) directly, ensuring Matplotlib compatibility
    colors = [
        (1.0, 0.0, 0.0, 0.6),    # State 0: Primitive Streak
        (0.8, 0.5, 0.0, 0.6),    # State 1: Mesoderm
        (0.0, 0.6, 0.0, 0.6),    # State 2: Hemogenic Endothelium
        (0.0, 0.7, 0.7, 0.6),    # State 3: Vascular Endothelial
        (0.0, 0.0, 1.0, 0.6)     # State 4: Blood Progenitor
    ]
    state_names = [
        "Primitive Streak Flow", 
        "Mesoderm Flow", 
        "Hemogenic Endothelium Flow", 
        "Vascular Endothelial Flow", 
        "Blood Progenitor Flow"
    ]
    
    # Plot streamplots/quivers for each state's MLP
    for k in range(K):
        mlp = fitted_model.drift.mlps[k]
        # Evaluate mlp on all grid points
        drifts = jax.vmap(mlp)(grid_jnp) # (400, 2)
        U = np.array(drifts[:, 0]).reshape(X.shape)
        V = np.array(drifts[:, 1]).reshape(Y.shape)
        
        # Plot streamplot representing the flow field
        axes[1].streamplot(X, Y, U, V, color=colors[k], density=0.8, arrowsize=1.2, linewidth=1.0)
        # Plot a single representative line for legend
        axes[1].plot([], [], color=colors[k][:3], label=f"State {k}: {state_names[k]}")
        
    axes[1].set_title("Discovered Nonlinear Latent Dynamics\n(Vector Fields/Flows per Cell State)", fontsize=14)
    axes[1].set_xlabel("Latent Coordinate $z_0$", fontsize=12)
    axes[1].set_ylabel("Latent Coordinate $z_1$", fontsize=12)
    axes[1].set_xlim(-grid_lim, grid_lim)
    axes[1].set_ylim(-grid_lim, grid_lim)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_data_sde.png', dpi=150)
    print("Saved real_data_sde.png successfully!")

if __name__ == "__main__":
    main()
