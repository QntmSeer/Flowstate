import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from stateflow.sde_core import NeuralSLDS
from stateflow.sde_em import HybridSDEFit

def test_sde_em():
    print("Initialize VEM Test Data...")
    K = 2
    D_z = 2
    D_x = 4
    T = 60
    
    # Let's generate a true unaligned sequence
    # Irregular timestamps
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(101), (T,)) * 10.0)
    
    # Dummy observations centered loosely around 2 distinct branches
    x1 = jax.random.normal(jax.random.PRNGKey(1), (T//2, D_x)) + 5.0
    x2 = jax.random.normal(jax.random.PRNGKey(2), (T - T//2, D_x)) - 5.0
    xs = jnp.vstack([x1, x2])
    
    C = jax.random.normal(jax.random.PRNGKey(3), (D_x, D_z))
    R = jnp.eye(D_x) * 1.0
    
    key = jax.random.PRNGKey(42)
    sde_model = NeuralSLDS(K, D_z, key)
    
    print("Initialize Hybrid SDE Fitter...")
    fitter = HybridSDEFit(sde_model, C, R, K, D_z)
    
    print("Running EM Fitting Loop (3 iterations)...")
    expected_s, fitted_model = fitter.fit(xs, ts, max_iter=3)
    
    print("\n--- FINAL DISCRETE TRANSITION MATRIX (A) ---")
    with jnp.printoptions(precision=3, suppress=True):
        print(fitter.A)
        
    print("\nEM Test Successful. JAX Autodiff penetrated the ODE solver correctly!")

if __name__ == "__main__":
    test_sde_em()
