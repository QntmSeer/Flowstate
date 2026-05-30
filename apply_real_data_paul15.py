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
    print("1. Loading Paul 2015 Myeloid Progenitor Dataset...")
    # This dataset contains ~2730 cells continuously differentiating
    adata = sc.datasets.paul15()
    
    # Preprocessing
    print("Preprocessing data...")
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    print("Computing PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    
    # Order cells by PC1 as a proxy for developmental pseudotime
    time_order = np.argsort(adata.obsm['X_pca'][:, 0])
    
    # We will use the top 3 PCs as our observations (D_x = 3)
    x_obs = jnp.array(adata.obsm['X_pca'][time_order, :3])
    T_cells, D_x = x_obs.shape
    
    # Set pseudotime range from 0 to 10
    ts = adata.obsm['X_pca'][time_order, 0]
    ts = (ts - ts.min()) / (ts.max() - ts.min()) * 10.0
    ts = jnp.array(ts)
    
    K = 3   # 3 cell programs: CMP (Stem), MEP (Erythroid), GMP (Myeloid/Granulocyte)
    D_z = 2 # 2D latent space to easily visualize the learned vector fields
    
    print(f"Initializing Neural SDE Model (K={K}, D_z={D_z}, D_x={D_x})...")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Set up model and initial parameters (using width=64 depth=2 MLPs)
    sde_model = NeuralSLDS(K, D_z, k1)
    
    # Initialize emission mapping C (3 -> 2) and noise R
    C = jax.random.normal(k2, (D_x, D_z)) * 0.5
    R = jnp.eye(D_x) * 0.5
    
    # Initialize fitter
    fitter = HybridSDEFit(sde_model, C, R, K, D_z)
    
    # Run EM + Optax gradient loop (using optimized fixed-step solver)
    max_iter = 5
    print(f"Fitting Nonlinear Neural SDE (EM/Optax) on Paul15 for {max_iter} iterations...")
    expected_s, fitted_model = fitter.fit(x_obs, ts, max_iter=max_iter)
    
    # Get final inferred states
    inferred_states = np.argmax(expected_s, axis=1)
    
    print("Computing UMAP for visualization...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)
    umap_coords = adata.obsm['X_umap']
    
    # Unorder the inferred states back to original cell indices
    original_inferred = np.zeros(T_cells, dtype=int)
    original_inferred[time_order] = inferred_states
    
    print("Plotting results & learned vector fields...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: UMAP colored by inferred states
    scatter = axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1], c=original_inferred, cmap='viridis', alpha=0.8, s=15)
    axes[0].set_title("Paul 2015 Myeloid Bifurcation\n(UMAP colored by SDE Inferred Macro-States)", fontsize=14)
    axes[0].set_xlabel("UMAP 1", fontsize=12)
    axes[0].set_ylabel("UMAP 2", fontsize=12)
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("Inferred State ID", fontsize=11)
    
    # Plot 2: Discovered Nonlinear Dynamical Vector Fields in Latent Space
    grid_lim = 2.0
    x_grid = np.linspace(-grid_lim, grid_lim, 20)
    y_grid = np.linspace(-grid_lim, grid_lim, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_jnp = jnp.array(grid_points)
    
    colors = [(1.0, 0.0, 0.0, 0.6), (0.0, 0.6, 0.0, 0.6), (0.0, 0.0, 1.0, 0.6)]
    state_names = ["CMP (Stem) Flow", "MEP (Erythroid) Flow", "GMP (Myeloid) Flow"]
    
    for k in range(K):
        mlp = fitted_model.drift.mlps[k]
        drifts = jax.vmap(mlp)(grid_jnp)
        U = np.array(drifts[:, 0]).reshape(X.shape)
        V = np.array(drifts[:, 1]).reshape(Y.shape)
        
        axes[1].streamplot(X, Y, U, V, color=colors[k], density=0.8, arrowsize=1.2, linewidth=1.0)
        axes[1].plot([], [], color=colors[k][:3], label=f"State {k}: {state_names[k]}")
        
    axes[1].set_title("Discovered Nonlinear Latent Dynamics\n(Vector Fields/Flows per Cell State)", fontsize=14)
    axes[1].set_xlabel("Latent Coordinate $z_0$", fontsize=12)
    axes[1].set_ylabel("Latent Coordinate $z_1$", fontsize=12)
    axes[1].set_xlim(-grid_lim, grid_lim)
    axes[1].set_ylim(-grid_lim, grid_lim)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/real_data_paul15.png', dpi=150)
    print("Saved assets/real_data_paul15.png successfully!")

if __name__ == "__main__":
    main()
