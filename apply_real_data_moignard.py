import scanpy as sc
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from stateflow.core import SLDSParams
from stateflow.inference import VariationalEM

def analyze_moignard():
    print("Loading Moignard 2015 Hematopoiesis Dataset...")
    # This dataset has ~3934 cells and 33 transcription factors, 
    # capturing early blood development (primitive streak to endothelium/blood).
    adata = sc.datasets.moignard15()
    
    # Preprocessing
    print("Preprocessing data...")
    # Moignard is qPCR dCt data! Do NOT run normalize_total or log1p on it, 
    # as these operate on raw integer RNA counts. dCt is already normalized and logarithmic.
    
    # Check for NaNs and impute with 0
    # First cast to float32 because the raw excel read might return object dtypes
    adata.X = adata.X.astype(np.float32)
    if np.isnan(adata.X).any():
        print("Dataset contains NaNs. Imputing with zeros...")
        adata.X = np.nan_to_num(adata.X, nan=0.0)
    
    # Scale data to N(0, 1) for stable SLDS inference
    sc.pp.scale(adata)
    
    # Use all 33 genes since D_x = 33 is small enough for SLDS natively.
    x_real = adata.X
    T_cells, D_x = x_real.shape
    
    print("Computing pseudotime ordering via PCA...")
    # Filter genes with zero variance
    sc.pp.filter_genes(adata, min_cells=5)
    
    # Try PCA with arpack, fallback to random if ARPACK starting vector is zero due to sparsity
    try:
        sc.tl.pca(adata, svd_solver='arpack')
    except Exception:
        sc.tl.pca(adata, svd_solver='randomized')
    
    # Let's sort simply by PC1 (often correlates with development time in such datasets)
    time_order = np.argsort(adata.obsm['X_pca'][:, 0])
    x_ordered = jnp.array(x_real[time_order])
    
    # Initialize StateFlow Model
    K = 3     # 3 metastable states (e.g. Epiblast -> Mesoderm -> Blood)
    D_z = 5   # Latent dimension
    
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
    
    print("Fitting model to Moignard data...")
    exp_s, mu_z, V_z = vem.fit(x_ordered, max_iter=10)
    
    print("\n--- INFERRED DISCRETE MARKOV TRANSITIONS ---")
    with jnp.printoptions(precision=3, suppress=True):
        print(vem.params.A)
        
    print("\nPlotting Results...")
    # Get inferred discrete states
    inferred_states = np.argmax(exp_s, axis=1)
    
    # Sort the PCA visualization to match the time_order
    pca_ordered = adata.obsm['X_pca'][time_order]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: True Experimental Groups
    groups = adata.obs['exp_groups'].values[time_order]
    unique_groups = np.unique(groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
    
    for idx, g in enumerate(unique_groups):
        mask = groups == g
        axes[0].scatter(pca_ordered[mask, 0], pca_ordered[mask, 1], label=g, alpha=0.6, color=colors[idx])
    axes[0].set_title("Moignard Hematopoiesis\n(Colored by True Sort Stage)")
    axes[0].legend()
    
    # Plot 2: StateFlow Inferred States
    scatter = axes[1].scatter(pca_ordered[:, 0], pca_ordered[:, 1], c=inferred_states, cmap='viridis', alpha=0.6)
    axes[1].set_title("Flowstate Inferred Metastable Programs\n(Colored by Inferred State)")
    plt.colorbar(scatter, ax=axes[1], label='Inferred State (K)')
    
    plt.tight_layout()
    plt.savefig('real_data_moignard.png')
    print("Saved real_data_moignard.png")

if __name__ == "__main__":
    analyze_moignard()
