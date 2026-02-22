import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np

# Import our package
from stateflow.core import SLDSParams
from stateflow.inference import VariationalEM

def main():
    print("Loading Paul15 Myeloid Differentiation Dataset...")
    # This downloads ~25MB of data
    adata = sc.datasets.paul15()
    
    # Preprocessing
    print("Preprocessing data...")
    sc.pp.recipe_zheng17(adata, n_top_genes=200)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    
    # Compute neighbors and diffusion map to get a pseudotime ordering
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
    sc.tl.diffmap(adata)
    
    # Identify root cell (heuristically max of a diffmap component)
    root_id = np.argmax(adata.obsm['X_diffmap'][:, 1]) 
    adata.uns['iroot'] = root_id
    sc.tl.dpt(adata)
    
    print("Ordering cells by Diffusion Pseudotime (DPT)...")
    order = np.argsort(adata.obs['dpt_pseudotime'].values)
    adata = adata[order, :].copy()
    
    # We will use the top 20 PCs as our continuous observations x_t
    x_real = adata.obsm['X_pca']
    x_jnp = jnp.array(x_real)
    T, D_x = x_jnp.shape
    
    K = 4   # Let's assume 4 metastable cell programs in this trajectory
    D_z = 5 # 5 latent regulatory pathways
    
    print(f"Dataset shape: {T} cells, {D_x} PCA components")
    print(f"Initializing SLDS Model (K={K}, D_z={D_z})...")
    
    # Initialize random parameters
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    pi = jnp.ones(K) / K
    
    # Transition matrix (biased towards staying in same state to encourage contiguous programs)
    A_raw = jnp.eye(K) * 5.0 + jax.random.uniform(k1, shape=(K, K))
    A = A_raw / jnp.sum(A_raw, axis=1, keepdims=True)
    
    # Continuous dynamics (stable)
    A_s = jax.random.normal(k2, shape=(K, D_z, D_z)) * 0.1 + 0.5 * jnp.eye(D_z)
    
    # Process noise
    Q_s = jnp.tile(jnp.eye(D_z) * 0.1, (K, 1, 1))
    
    # Emission mapping
    C = jax.random.normal(k3, shape=(D_x, D_z)) * 0.5
    
    # Emission noise
    R = jnp.eye(D_x) * 0.5
    
    mu_0 = jnp.zeros((K, D_z))
    Sigma_0 = jnp.tile(jnp.eye(D_z), (K, 1, 1))
    
    init_params = SLDSParams(pi, A, A_s, Q_s, C, R, mu_0, Sigma_0)
    
    print("Fitting SLDS model on Paul15 Data (10 iterations)...")
    vem = VariationalEM(init_params)
    expected_s, mu_z, V_z = vem.fit(x_jnp, max_iter=10)
    
    # Plot results
    print("Plotting results...")
    
    # Find the most likely state for each cell
    inferred_states = np.argmax(expected_s, axis=1)
    # Convert simply to standard CPU numpy array for scanpy plotting compatibility
    adata.obs['inferred_program'] = np.array(inferred_states).astype(str)
    
    # Make subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Inferred Programs vs DPT
    sc.pl.scatter(adata, x='dpt_pseudotime', y='inferred_program', color='inferred_program', show=False, ax=axes[0])
    axes[0].set_title('Inferred Cell Programs Ordered by Pseudotime')
    
    # 2. Original Paul15 Clusters vs DPT (to see what programs correspond to)
    sc.pl.scatter(adata, x='dpt_pseudotime', y='paul15_clusters', color='paul15_clusters', show=False, ax=axes[1])
    axes[1].set_title('Paul15 Original Clusters Ordered by Pseudotime')
    
    # 3. Latent Dynamics (first 3 dims)
    time = np.arange(T)
    for i in range(min(3, D_z)):
        axes[2].plot(time, mu_z[:, i], label=f'Latent Dim {i}')
    axes[2].set_title('Inferred Continuous Regulatory Pathways over Pseudotime')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('real_data_paul15.png')
    print("Saved real_data_paul15.png")

if __name__ == "__main__":
    main()
