import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from stateflow.core import SLDSParams
from benchmark_data import generate_bifurcating_topology

def generate_barcoded_bifurcation(T=300, D_z=3, D_x=20, n_clones=10, seed=42):
    """
    Simulation of a bifurcating trajectory where a small number of root 'Stem'
    cells are tagged with a unique barcode ID.
    """
    # 1. We generate a large bifurcating dataset
    params, s_true, z_true, x_true = generate_bifurcating_topology(T=T, D_z=D_z, D_x=D_x, seed=seed)
    
    # We want to identify contiguous "lineages". 
    # In a real biological system, one cell divides into two, both having the same barcode.
    # In our mathematical simulation (which generates completely independent T paths), 
    # we can simulate "clones" by defining clusters of initial Z_0 states and assigning them barcodes.
    
    # Wait, the current `simulate_trajectory` simulates T independent cells across ONE time step each?
    # No, our simulator currently assumes it's generating a single time-series or independent samples?
    # Actually, `simulate_trajectory` (Phase 1) simulates a single T-step Markov chain.
    # A single time-series of length T is one cell.
    
    # For lineage tracing, we need N cells.
    # Let's generate N independent cells. Some cells share the exact same z_0 (they are clones).
    N = 300
    
    # 10 distinct clones. Each clone has 30 cells.
    cells_per_clone = N // n_clones
    barcodes = np.repeat(np.arange(n_clones), cells_per_clone)
    
    # Generate continuous trajectories for each cell.
    # Cells in the same barcode start with the exact same initial state z_0.
    np.random.seed(seed)
    
    s_all = []
    z_all = []
    x_all = []
    
    key = jax.random.PRNGKey(seed)
    
    # For simplicity, we just simulate T=1 (one snapshot) for N cells, or a very short time-series?
    # Most scRNA-seq is a single snapshot per cell.
    # Our Variational EM handles independent single-cell snapshots (T length sequence where T=N cells).
    # If the input to Variational EM is (N, D_x), it treats it as a time-series.
    # If we want independent observations, we technically need batch support, but for demo, 
    # ordering them by pseudotime acts as the pseudo-trajectory.
    
    # Let's mock a continuous pseudotime block where clones are grouped together
    # to visualize the effect in PCA space.
    for clone_id in range(n_clones):
        clone_key, key = jax.random.split(key)
        
        # All cells in this clone start at the exact same root state
        # In reality, they are just independent samples from the transition matrix, 
        # but constrained to share similar continuous paths.
        
        # Let's just generate independent paths and tag them. 
        # To make them "familial", we reduce the process noise Q_s for a shared seed.
        
        for _ in range(cells_per_clone):
            cell_key, key = jax.random.split(key)
            # Simulate a short path, grab the final state
            # This is a bit complex for a simple demo. 
            pass

    # A simpler way: just take the generated T=300 bifurcating chain and assign random barcodes.
    # Real biology has contiguous barcodes. So if Cell 10-20 are Fate A, they might share Barcode 2.
    # T may be one less if simulate_trajectory trims 1 state, but x_true shape is what matters
    N_cells = x_true.shape[0]
    barcodes = np.zeros(N_cells, dtype=int)
    
    # Sort the true z path by continuous distance from origin to mock "pseudotime"
    distances = jnp.linalg.norm(z_true - params.mu_0[0], axis=1)
    time_order = np.argsort(distances)
    
    x_ordered = x_true[time_order]
    z_ordered = z_true[time_order]
    s_ordered = s_true[time_order]
    
    # Assign contiguous chunks to clones
    chunk_size = max(1, N_cells // n_clones)
    for i in range(n_clones):
        barcodes[i*chunk_size : (i+1)*chunk_size] = i
        
    # Any remainder goes to last clone
    barcodes[n_clones*chunk_size:] = n_clones - 1
        
    return params, s_ordered, z_ordered, x_ordered, barcodes

if __name__ == "__main__":
    params, s, z, x, barcodes = generate_barcoded_bifurcation()
    
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    scatter1 = axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=s, cmap='viridis')
    axes[0].set_title("Bifurcation (True Discrete State)")
    plt.colorbar(scatter1, ax=axes[0])
    
    scatter2 = axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=barcodes, cmap='tab20')
    axes[1].set_title("Bifurcation (Lineage Barcodes)")
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.savefig('test_barcodes.png')
    print("Saved test_barcodes.png")
