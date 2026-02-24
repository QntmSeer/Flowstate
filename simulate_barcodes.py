import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from stateflow.core import SLDSParams
from benchmark_data import generate_bifurcating_topology

def generate_barcoded_bifurcation(T=300, D_z=3, D_x=20, n_clones=10, seed=42):
    """
    Generates a bifurcating trajectory where cells are tagged with integer lineage barcode IDs.

    Cells are ordered by pseudotime (distance from root in latent space) and then
    partitioned into n_clones contiguous groups, each assigned a unique barcode ID.
    This simulates the biological reality where clonally related cells occupy adjacent
    regions of the trajectory.

    Returns:
        params: Ground truth SLDSParams
        s_ordered: (T,) discrete states sorted by pseudotime
        z_ordered: (T, D_z) continuous latent states sorted by pseudotime
        x_ordered: (T, D_x) gene expression sorted by pseudotime
        barcodes: (T,) integer barcode IDs per cell
    """
    params, s_true, z_true, x_true = generate_bifurcating_topology(T=T, D_z=D_z, D_x=D_x, seed=seed)

    N_cells = x_true.shape[0]
    barcodes = np.zeros(N_cells, dtype=int)

    # Sort cells by distance from the root mean (proxy for pseudotime)
    distances = jnp.linalg.norm(z_true - params.mu_0[0], axis=1)
    time_order = np.argsort(distances)

    x_ordered = x_true[time_order]
    z_ordered = z_true[time_order]
    s_ordered = s_true[time_order]

    # Assign contiguous pseudotime chunks to each barcode clone
    chunk_size = max(1, N_cells // n_clones)
    for i in range(n_clones):
        barcodes[i*chunk_size : (i+1)*chunk_size] = i

    # Any remainder is assigned to the last clone
    barcodes[n_clones*chunk_size:] = n_clones - 1

    return params, s_ordered, z_ordered, x_ordered, barcodes

if __name__ == "__main__":
    params, s, z, x, barcodes = generate_barcoded_bifurcation()

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(x_pca[:, 0], x_pca[:, 1], c=s, cmap='viridis')
    axes[0].set_title("Bifurcation (True Discrete State)")
    plt.colorbar(scatter1, ax=axes[0])

    scatter2 = axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=barcodes, cmap='tab20')
    axes[1].set_title("Bifurcation (Lineage Barcodes)")
    plt.colorbar(scatter2, ax=axes[1])

    plt.savefig('test_barcodes.png')
    print("Saved test_barcodes.png")
