import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from stateflow.core import SLDSParams
from stateflow.inference import VariationalEM
from benchmark_data import generate_bifurcating_topology

def run_bifurcation_benchmark():
    print("1. Generating Synthetic Bifurcating Topology...")
    true_params, s_true, z_true, x_true = generate_bifurcating_topology(T=800, D_z=3, D_x=50, seed=101)
    
    T, D_x = x_true.shape
    K = 3 # We know it's a 3-state system
    D_z = 3
    
    # Let's initialize a COMPLETELY NAIVE model (no knowledge of bifurcation)
    print("2. Initializing Naive SLDS Model...")
    key = jax.random.PRNGKey(999)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    pi_init = jnp.ones(K) / K
    
    # Initialize A as a uniform transition matrix (dense graph mapping everywhere)
    A_raw = jax.random.uniform(k1, shape=(K, K)) + 0.1
    # Bias slightly towards self-transitions to prevent immediate collapse
    A_raw = A_raw + jnp.eye(K) * 2.0
    A_init = A_raw / jnp.sum(A_raw, axis=1, keepdims=True)
    
    A_s_init = jax.random.normal(k2, shape=(K, D_z, D_z)) * 0.1 + 0.5 * jnp.eye(D_z)
    Q_s_init = jnp.tile(jnp.eye(D_z) * 0.1, (K, 1, 1))
    
    C_init = jax.random.normal(k3, shape=(D_x, D_z)) * 0.1
    R_init = jnp.eye(D_x) * jnp.var(x_true, axis=0) # Data-driven noise init
    
    mu_0_init = jnp.zeros((K, D_z))
    Sigma_0_init = jnp.tile(jnp.eye(D_z), (K, 1, 1))
    
    init_params = SLDSParams(pi_init, A_init, A_s_init, Q_s_init, C_init, R_init, mu_0_init, Sigma_0_init)
    
    print("3. Fitting VariationalEM to recover topology...")
    vem = VariationalEM(init_params)
    expected_s, mu_z, V_z = vem.fit(x_true, max_iter=15)
    
    inferred_A = vem.params.A
    
    print("\n--- INFERRED TOPOLOGY (TRANSITION MATRIX) ---")
    with jnp.printoptions(precision=3, suppress=True):
        print(inferred_A)
    
    # We will build a NetworkX graph from the transition matrix to visualize the topology
    G = nx.DiGraph()
    for i in range(K):
        G.add_node(i)
        for j in range(K):
            # Only add edges with transition probability > 5% (to filter noise)
            weight = inferred_A[i, j]
            if i != j and weight > 0.05:
                G.add_edge(i, j, weight=weight)
                
    print("4. Plotting Results...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: The Inferred Graph Topology
    pos = nx.spring_layout(G, seed=42)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=['#1f77b4', '#ff7f0e', '#2ca02c'], node_size=1500)
    nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=16, font_color='white')
    
    # Draw edges with width proportional to probability
    if len(G.edges) > 0:
        edges = G.edges()
        weights = [G[u][v]['weight'] * 10 for u,v in edges]
        nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edges, width=weights, arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.2')
        edge_labels = {(u,v): f"{G[u][v]['weight']:.2f}" for u,v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, ax=axes[0])
        
    axes[0].set_title("Inferred Directed Topology Graph\n(from Transition Matrix)")
    axes[0].axis('off')
    
    # Plot 2: Trajectory in PCA Space
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_true)
    
    # Hard assignment of states
    inferred_states = np.argmax(expected_s, axis=1)
    
    scatter = axes[1].scatter(x_pca[:, 0], x_pca[:, 1], c=inferred_states, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=axes[1], label='Inferred Discrete State')
    axes[1].set_title("PCA of Gene Expression labeled by Inferred State")
    
    plt.tight_layout()
    plt.savefig('benchmark_bifurcation_inferred.png')
    print("Saved benchmark_bifurcation_inferred.png")

if __name__ == "__main__":
    run_bifurcation_benchmark()
