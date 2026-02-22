import scanpy as sc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from stateflow.core import SLDSParams
from stateflow.inference import VariationalEM

def analyze_paul15():
    print("Loading Paul 2015 Myeloid Progenitor Dataset...")
    # This dataset contains ~2730 cells from the bone marrow.
    # It details the continuous bifurcation of 
    # CMP (Common Myeloid Progenitor) -> MEP (Erythroid) and GMP (Granulocyte/Macrophage)
    adata = sc.datasets.paul15()
    
    # Preprocessing
    print("Preprocessing data...")
    # sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select highly variable genes to reduce D_x to a manageable number for SLDS
    print("Selecting top 100 Highly Variable Genes for SDE computational speed...")
    sc.pp.highly_variable_genes(adata, n_top_genes=100)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale data to N(0, 1) for stable SLDS inference
    sc.pp.scale(adata, max_value=10)
    
    x_real = adata.X
    T_cells, D_x = x_real.shape
    
    print("Computing pseudotime ordering via PCA/Diffusion Maps...")
    sc.tl.pca(adata, svd_solver='arpack')
    
    # For a bifurcation, sorting purely by PC1 isn't perfect, but it suffices 
    # to give the EM filter a rough "time" ordering to work with.
    # A true implementation would use Diffusion Pseudotime (DPT) from a root cell.
    # For demonstration, we'll sort by PC1.
    time_order = np.argsort(adata.obsm['X_pca'][:, 0])
    x_ordered = jnp.array(x_real[time_order])
    
    # Initialize StateFlow Model
    K = 3     # 3 metastable states: CMP, MEP, GMP
    D_z = 5   # Latent regulatory dimension
    
    print(f"Initializing Variational EM (K={K}, D_z={D_z}, D_x={D_x})...")
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    pi_init = jnp.ones(K) / K
    A_init = jnp.ones((K, K)) / K + jnp.eye(K) * 2.0
    A_init = A_init / jnp.sum(A_init, axis=1, keepdims=True)
    
    A_s_init = jnp.tile(jnp.eye(D_z)*0.9, (K, 1, 1))
    Q_s_init = jnp.tile(jnp.eye(D_z)*0.1, (K, 1, 1))
    C_init = jax.random.normal(k3, (D_x, D_z)) * 0.1
    R_init = jnp.eye(D_x) * jnp.var(x_ordered, axis=0) + 1e-4*jnp.eye(D_x)
    mu_0_init = jnp.zeros((K, D_z))
    Sigma_0_init = jnp.tile(jnp.eye(D_z), (K, 1, 1))
    
    params_init = SLDSParams(pi_init, A_init, A_s_init, Q_s_init, C_init, R_init, mu_0_init, Sigma_0_init)
    
    vem = VariationalEM(params_init)
    
    print("Fitting model to Paul15 data...")
    exp_s, mu_z, V_z = vem.fit(x_ordered, max_iter=8) # 8 iterations for speed
    
    print("\n--- INFERRED DISCRETE MARKOV TRANSITIONS ---")
    with jnp.printoptions(precision=3, suppress=True):
        print(vem.params.A)
        
    print("\nPlotting Results...")
    inferred_states = np.argmax(exp_s, axis=1)
    
    # We will compute a UMAP for visualization, colored by inferred states
    print("Computing UMAP for visualization...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.umap(adata)
    
    umap_coords = adata.obsm['X_umap']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: True Clusters (Paul15 provides cluster integers 1-19)
    true_clusters = adata.obs['paul15_clusters']
    cluster_codes = true_clusters.cat.codes.values if hasattr(true_clusters, 'cat') else np.array(true_clusters).astype('category').codes
    axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1], c=cluster_codes, cmap='tab20', s=10)
    axes[0].set_title("Paul 2015 Myeloid Bifurcation\n(Colored by 19 True Micro-Clusters)")
    
    # Plot 2: StateFlow Inferred Macro-States
    # We must unordered the inferred states back to original indices
    # time_order maps new_idx -> old_idx.
    # We want: original_array[time_order] = inferred_states
    
    original_inferred = np.zeros(T_cells, dtype=int)
    original_inferred[time_order] = inferred_states
    
    scatter = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=original_inferred, cmap='viridis', s=15, alpha=0.8)
    axes[1].set_title("Flowstate Inferred Macro-Trajectories\n(3 States: e.g. Stem, Erythroid, Granulocyte)")
    plt.colorbar(scatter, ax=axes[1], label='Inferred State (K)')
    
    plt.tight_layout()
    plt.savefig('real_data_paul15.png')
    print("Saved real_data_paul15.png")

if __name__ == "__main__":
    analyze_paul15()
