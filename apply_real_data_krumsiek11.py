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
    print("1. Loading Krumsiek 2011 Myeloid Progenitor qPCR Dataset...")
    # This dataset contains 640 cells and 11 transcription factors,
    # capturing myeloid differentiation into Megakaryocyte, Erythroid, Monocyte, and Neutrophil lineages.
    adata = sc.datasets.krumsiek11()
    
    # Preprocessing
    print("Preprocessing data...")
    adata.obs_names_make_unique()
    adata.X = adata.X.astype(np.float32)
    
    # Scale data to N(0, 1) for stable SLDS/SDE inference
    sc.pp.scale(adata)
    
    # Compute PCA
    print("Computing PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=5)
    
    # Compute Neighbors and Diffusion Pseudotime (DPT) to get a continuous time proxy
    print("Computing Diffusion Pseudotime (DPT)...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=5)
    # Set the root cell to be the first progenitor cell
    adata.uns['iroot'] = np.flatnonzero(adata.obs['cell_type'] == 'progenitor')[0]
    sc.tl.dpt(adata)
    
    # Sort cells by DPT to create a sequential trajectory
    time_order = np.argsort(adata.obs['dpt_pseudotime'].values)
    
    # We will use the top 4 PCs as our observations (D_x = 4)
    x_obs = jnp.array(adata.obsm['X_pca'][time_order, :4])
    T_cells, D_x = x_obs.shape
    
    # Set pseudotime range from 0 to 10
    ts = adata.obs['dpt_pseudotime'].values[time_order]
    ts = (ts - ts.min()) / (ts.max() - ts.min()) * 10.0
    ts = jnp.array(ts)
    
    K = 5   # 5 cell programs: progenitor, Ery, Mk, Mo, Neu
    D_z = 2 # 2D latent space to easily visualize the learned vector fields
    
    print(f"Initializing Neural SDE Model (K={K}, D_z={D_z}, D_x={D_x})...")
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Set up model and initial parameters
    sde_model = NeuralSLDS(K, D_z, k1)
    
    # Initialize emission mapping C (4 -> 2) and noise R
    C = jax.random.normal(k2, (D_x, D_z)) * 0.5
    R = jnp.eye(D_x) * 0.5
    
    # Initialize fitter
    fitter = HybridSDEFit(sde_model, C, R, K, D_z)
    
    # Run EM + Optax gradient loop
    max_iter = 10
    print(f"Fitting Nonlinear Neural SDE (EM/Optax) on Krumsiek11 for {max_iter} iterations...")
    expected_s, fitted_model = fitter.fit(x_obs, ts, max_iter=max_iter)
    
    # Get final inferred states
    inferred_states = np.argmax(expected_s, axis=1)
    
    # Compute UMAP for visualization
    print("Computing UMAP for visualization...")
    sc.tl.umap(adata)
    umap_coords = adata.obsm['X_umap']
    
    # Unorder the inferred states back to original cell indices
    original_inferred = np.zeros(T_cells, dtype=int)
    original_inferred[time_order] = inferred_states
    
    print("Plotting results & learned vector fields...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    
    # Plot 1: UMAP colored by true experimental cell types
    cell_types = adata.obs['cell_type'].values
    unique_ct = ['progenitor', 'Ery', 'Mk', 'Mo', 'Neu']
    # Harmonious premium color palette
    ct_colors = {
        'progenitor': '#7f8c8d',  # Slate Grey
        'Ery': '#e74c3c',         # Crimson Red
        'Mk': '#9b59b6',          # Amethyst Purple
        'Mo': '#16a085',          # Turquoise/Teal
        'Neu': '#d35400'          # Pumpkin Orange
    }
    
    for ct in unique_ct:
        mask = cell_types == ct
        axes[0].scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                        label=ct, alpha=0.8, s=25, color=ct_colors[ct])
    axes[0].set_title("Krumsiek 2011 Myeloid Progenitors\n(Colored by True Cell Type)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("UMAP 1", fontsize=11)
    axes[0].set_ylabel("UMAP 2", fontsize=11)
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: UMAP colored by SDE inferred macro-states
    scatter = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                             c=original_inferred, cmap='viridis', alpha=0.8, s=25)
    axes[1].set_title("Flowstate Inferred Macro-States\n(Colored by Inferred State)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("UMAP 1", fontsize=11)
    axes[1].set_ylabel("UMAP 2", fontsize=11)
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Inferred State ID", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Plot 3: Discovered Nonlinear Dynamical Vector Fields in Latent Space
    grid_lim = 2.0
    x_grid = np.linspace(-grid_lim, grid_lim, 20)
    y_grid = np.linspace(-grid_lim, grid_lim, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_jnp = jnp.array(grid_points)
    
    # Clean vector field streamplot colors corresponding to the 5 states
    colors = [
        (0.5, 0.5, 0.5, 0.6),    # State 0: Progenitor (Grey)
        (0.9, 0.3, 0.2, 0.6),    # State 1: Ery (Red)
        (0.6, 0.3, 0.7, 0.6),    # State 2: Mk (Purple)
        (0.1, 0.6, 0.5, 0.6),    # State 3: Mo (Teal)
        (0.8, 0.4, 0.0, 0.6)     # State 4: Neu (Orange)
    ]
    state_names = [
        "Progenitor Flow", 
        "Erythroid (Ery) Flow", 
        "Megakaryocyte (Mk) Flow", 
        "Monocyte (Mo) Flow", 
        "Neutrophil (Neu) Flow"
    ]
    
    for k in range(K):
        mlp = fitted_model.drift.mlps[k]
        drifts = jax.vmap(mlp)(grid_jnp)
        U = np.array(drifts[:, 0]).reshape(X.shape)
        V = np.array(drifts[:, 1]).reshape(Y.shape)
        
        axes[2].streamplot(X, Y, U, V, color=colors[k], density=0.8, arrowsize=1.2, linewidth=1.0)
        axes[2].plot([], [], color=colors[k][:3], label=f"State {k}: {state_names[k]}")
        
    axes[2].set_title("Discovered Nonlinear Latent Dynamics\n(Vector Fields/Flows per Cell State)", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Latent Coordinate $z_0$", fontsize=11)
    axes[2].set_ylabel("Latent Coordinate $z_1$", fontsize=11)
    axes[2].set_xlim(-grid_lim, grid_lim)
    axes[2].set_ylim(-grid_lim, grid_lim)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/real_data_krumsiek11.png', dpi=150)
    plt.savefig('real_data_krumsiek11.png', dpi=150)
    print("Saved assets/real_data_krumsiek11.png successfully!")
    print("Saved real_data_krumsiek11.png successfully!")

if __name__ == "__main__":
    main()
