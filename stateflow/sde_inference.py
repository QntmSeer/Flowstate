import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from stateflow.sde_core import NeuralSLDS

class ContinuousDiscreteKalmanFilter(eqx.Module):
    """
    Implements a Continuous-Discrete Extended/Linear Kalman Filter.
    It integrates the mean and covariance of the continuous state z_t
    between observation times, and performs standard discrete Bayesian
    updates at the observation times.
    """
    sde_model: NeuralSLDS
    C: jnp.ndarray
    R: jnp.ndarray

    def __init__(self, sde_model: NeuralSLDS, C: jnp.ndarray, R: jnp.ndarray):
        self.sde_model = sde_model
        self.C = C
        self.R = R
        
    def _pack_state(self, mu, P):
        return jnp.concatenate([mu, P.flatten()])
        
    def _unpack_state(self, state, D_z):
        mu = state[:D_z]
        P = state[D_z:].reshape((D_z, D_z))
        return mu, P

    def filter(self, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray, mu_0: jnp.ndarray, P_0: jnp.ndarray):
        """
        Forward filter pass over time series data x at irregular times t.
        xs: (T, D_x)
        ts: (T,)
        expected_s: (T, K) discrete state probabilities at times t
        """
        D_z = self.sde_model.D_z
        T = xs.shape[0]

        # Define the ODE for the mean and covariance prediction
        # dmu_dt = A_s mu + b_s
        # dP_dt = A_s P + P A_s^T + Q_s
        def ode_func(t, state, args):
            # args is s_prob (interpolate discrete expected_s logically, here piecewise constant or linear)
            # For simplicity, we just use the nearest expected_s. Ideally, we interpolate.
            s_prob = args 
            
            mu, P = self._unpack_state(state, D_z)
            
            # Expected drift matrices from sde_model
            drifts = jax.vmap(lambda k: self.sde_model.drift.A_s[k] @ mu + self.sde_model.drift.b_s[k])(jnp.arange(self.sde_model.K))
            expected_drift = jnp.average(drifts, weights=s_prob, axis=0) # dmu_dt
            
            # Expected Jacobians (A_s)
            expected_A = jnp.average(self.sde_model.drift.A_s, weights=s_prob, axis=0)
            
            # Expected Process Noise CVR (g g^T = Q)
            # diffusion(t, y, s) returns diagonal matrix in our case
            diffusions = jax.vmap(lambda k: jnp.diag(self.sde_model.diffusion.Q_s_chol[k])**2)(jnp.arange(self.sde_model.K))
            expected_Q = jnp.average(diffusions, weights=s_prob, axis=0) # dP_dt noise term
            
            dP_dt = expected_A @ P + P @ expected_A.T + expected_Q
            
            return self._pack_state(expected_drift, dP_dt)

        def step(carry, inputs):
            mu_prev, P_prev, t_prev = carry
            x_curr, t_curr, s_prob_curr = inputs
            
            # 1. Predict (Integrate ODE from t_prev to t_curr)
            # If t_curr == t_prev (e.g., at index 0), skip integration
            state_prev = self._pack_state(mu_prev, P_prev)
            
            # We use Tsit5 (Runge-Kutta) for smooth ODE integration
            term = diffrax.ODETerm(ode_func)
            solver = diffrax.Tsit5()
            
            # Conditionally solve ODE only if t_curr > t_prev
            def integrate():
                sol = diffrax.diffeqsolve(
                    term, solver, t_prev, t_curr, dt0=(t_curr - t_prev),
                    y0=state_prev, args=s_prob_curr, max_steps=10
                )
                return sol.ys[-1]
                
            state_pred = jax.lax.cond(
                t_curr > t_prev,
                integrate,
                lambda: state_prev
            )
            
            mu_pred, P_pred = self._unpack_state(state_pred, D_z)
            
            # 2. Update (Standard Kalman Update)
            y = x_curr - self.C @ mu_pred
            S = self.C @ P_pred @ self.C.T + self.R + 1e-4 * jnp.eye(self.R.shape[0])
            
            cho_S, lower = jax.scipy.linalg.cho_factor(S)
            K_gain = jax.scipy.linalg.cho_solve((cho_S, lower), self.C @ P_pred).T
            
            mu_upd = mu_pred + K_gain @ y
            P_upd = P_pred - K_gain @ self.C @ P_pred
            P_upd = 0.5 * (P_upd + P_upd.T) # Enforce symmetry
            
            return (mu_upd, P_upd, t_curr), (mu_upd, P_upd, mu_pred, P_pred)

        # Initial Update at t=0
        # If t=0 has an observation x_0, we update mu_0 prior with x_0
        y_0 = xs[0] - self.C @ mu_0
        S_0 = self.C @ P_0 @ self.C.T + self.R + 1e-4 * jnp.eye(self.R.shape[0])
        cho_S_0, lower_0 = jax.scipy.linalg.cho_factor(S_0)
        K_0 = jax.scipy.linalg.cho_solve((cho_S_0, lower_0), self.C @ P_0).T
        mu_0_upd = mu_0 + K_0 @ y_0
        P_0_upd = P_0 - K_0 @ self.C @ P_0
        P_0_upd = 0.5 * (P_0_upd + P_0_upd.T)
        
        carry_init = (mu_0_upd, P_0_upd, ts[0])
        
        # Scan from t=1 to T-1
        _, (mu_f, P_f, mu_p, P_p) = jax.lax.scan(
            step, carry_init, (xs[1:], ts[1:], expected_s[1:])
        )
        
        mu_filt = jnp.vstack([mu_0_upd[None, ...], mu_f])
        P_filt = jnp.vstack([P_0_upd[None, ...], P_f])
        
        # Return filter results (Smoothing would require backwards ODE integrate)
        return mu_filt, P_filt
