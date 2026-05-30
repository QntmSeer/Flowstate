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
        mu_filt, P_filt, _, _ = self._filter_full(xs, ts, expected_s, mu_0, P_0)
        return mu_filt, P_filt

    def _filter_full(self, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray, mu_0: jnp.ndarray, P_0: jnp.ndarray):
        """
        Runs the full forward filter pass and returns both filtered and predicted moments.
        """
        D_z = self.sde_model.D_z
        T = xs.shape[0]

        # Define the ODE for the mean and covariance prediction
        # dmu_dt = f(mu, s_t)
        # dP_dt = A P + P A^T + Q
        def ode_func(t, state, args):
            s_prob = args 
            mu, P = self._unpack_state(state, D_z)
            
            # Expected drift vectors from sde_model
            expected_drift = self.sde_model.drift(t, mu, s_prob) # dmu_dt
            
            # Expected Jacobians (evaluated at current mean mu)
            state_jacobians = self.sde_model.drift.get_jacobians(mu)
            expected_A = jnp.average(state_jacobians, weights=s_prob, axis=0)
            
            # Expected Process Noise Covariance (Q = g g^T)
            expected_g = self.sde_model.diffusion(t, mu, s_prob)
            expected_Q = expected_g @ expected_g.T
            
            dP_dt = expected_A @ P + P @ expected_A.T + expected_Q
            
            return self._pack_state(expected_drift, dP_dt)

        def step(carry, inputs):
            mu_prev, P_prev, t_prev = carry
            x_curr, t_curr, s_prob_curr = inputs
            
            # 1. Predict (Integrate ODE from t_prev to t_curr)
            state_prev = self._pack_state(mu_prev, P_prev)
            
            term = diffrax.ODETerm(ode_func)
            solver = diffrax.Tsit5()
            
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
        
        mu_pred_all = jnp.vstack([mu_0[None, ...], mu_p])
        P_pred_all = jnp.vstack([P_0[None, ...], P_p])
        
        return mu_filt, P_filt, mu_pred_all, P_pred_all

    def smooth(self, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray, mu_0: jnp.ndarray, P_0: jnp.ndarray):
        """
        Runs the forward filter pass followed by the backward RTS smoothing pass.
        """
        mu_filt, P_filt, mu_pred, P_pred = self._filter_full(xs, ts, expected_s, mu_0, P_0)
        
        T = xs.shape[0]
        D_z = self.sde_model.D_z
        
        def smoother_step(carry, inputs):
            mu_t_plus_1_smooth, P_t_plus_1_smooth = carry
            mu_f_t, P_f_t, mu_p_t_plus_1, P_p_t_plus_1, t_curr, t_next, s_prob_next = inputs
            
            dt = t_next - t_curr
            
            # Compute Transition Matrix Phi_t = I + dt * expected_A
            state_jacobians = self.sde_model.drift.get_jacobians(mu_f_t)
            expected_A = jnp.average(state_jacobians, weights=s_prob_next, axis=0)
            Phi_t = jnp.eye(D_z) + dt * expected_A
            
            # Smoothing gain J_t = P_f_t @ Phi_t.T @ (P_p_t_plus_1)^-1
            P_p_t_plus_1 = P_p_t_plus_1 + 1e-5 * jnp.eye(D_z)
            cho_Pp, lower_Pp = jax.scipy.linalg.cho_factor(P_p_t_plus_1)
            J_t = jax.scipy.linalg.cho_solve((cho_Pp, lower_Pp), Phi_t @ P_f_t).T
            
            mu_smooth = mu_f_t + J_t @ (mu_t_plus_1_smooth - mu_p_t_plus_1)
            P_smooth = P_f_t + J_t @ (P_t_plus_1_smooth - P_p_t_plus_1) @ J_t.T
            P_smooth = 0.5 * (P_smooth + P_smooth.T)
            
            return (mu_smooth, P_smooth), (mu_smooth, P_smooth)

        # Run backward smoother scan from t = T-2 down to 0
        carry_init = (mu_filt[-1], P_filt[-1])
        
        scan_inputs = (
            mu_filt[:-1][::-1],
            P_filt[:-1][::-1],
            mu_pred[1:][::-1],
            P_pred[1:][::-1],
            ts[:-1][::-1],
            ts[1:][::-1],
            expected_s[1:][::-1]
        )
        
        _, (mu_s, P_s) = jax.lax.scan(smoother_step, carry_init, scan_inputs)
        
        # Reverse outputs back to regular time ordering
        mu_s = mu_s[::-1]
        P_s = P_s[::-1]
        
        # Re-append T-1
        mu_smooth_all = jnp.vstack([mu_s, mu_filt[-1][None, ...]])
        P_smooth_all = jnp.vstack([P_s, P_filt[-1][None, ...]])
        
        return mu_smooth_all, P_smooth_all


class SpeculativeSpeculativeKalmanFilter(eqx.Module):
    """
    Implements a Speculative Speculative Continuous-Discrete Extended Kalman Filter (SSD-EKF).
    It parallelizes the target EKF prediction pass (Tsit5) for interval [t_k, t_k1]
    with a cheap draft solver (1-step Euler) pre-emptively drafting step [t_k1, t_k2]
    assuming the step is accepted. If accepted, it applies a first-order Taylor correction
    to align the pre-drafted mean with the Kalman update.
    """
    sde_model: NeuralSLDS
    C: jnp.ndarray
    R: jnp.ndarray
    tolerance: float

    def __init__(self, sde_model: NeuralSLDS, C: jnp.ndarray, R: jnp.ndarray, tolerance: float = 0.05):
        self.sde_model = sde_model
        self.C = C
        self.R = R
        self.tolerance = tolerance

    def _pack_state(self, mu, P):
        return jnp.concatenate([mu, P.flatten()])

    def _unpack_state(self, state, D_z):
        mu = state[:D_z]
        P = state[D_z:].reshape((D_z, D_z))
        return mu, P

    def filter(self, xs: jnp.ndarray, ts: jnp.ndarray, expected_s: jnp.ndarray, mu_0: jnp.ndarray, P_0: jnp.ndarray):
        """
        Forward filter pass utilizing Speculative Speculative SDE Integration (SSD-SDE).
        """
        D_z = self.sde_model.D_z
        T = xs.shape[0]

        # Define the ODE for the mean and covariance prediction
        def ode_func(t, state, args):
            s_prob = args
            mu, P = self._unpack_state(state, D_z)
            
            expected_drift = self.sde_model.drift(t, mu, s_prob)
            state_jacobians = self.sde_model.drift.get_jacobians(mu)
            expected_A = jnp.average(state_jacobians, weights=s_prob, axis=0)
            
            expected_g = self.sde_model.diffusion(t, mu, s_prob)
            expected_Q = expected_g @ expected_g.T
            
            dP_dt = expected_A @ P + P @ expected_A.T + expected_Q
            
            return self._pack_state(expected_drift, dP_dt)

        def step(carry, inputs):
            mu_prev, P_prev, mu_spec_prev, t_prev, was_accepted = carry
            x_curr, t_curr, t_next, s_prob_curr, s_prob_next = inputs
            
            # --- 1. Target Prediction (Path 1 - Tsit5 ODE Integration) ---
            state_prev = self._pack_state(mu_prev, P_prev)
            term = diffrax.ODETerm(ode_func)
            solver = diffrax.Tsit5()
            
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
            mu_pred_target, P_pred_target = self._unpack_state(state_pred, D_z)
            
            # --- 2. Speculative Drafting of Step k+1 -> k+2 (Path 2 - Cheap Euler step) ---
            # Pre-draft starting from the speculated mean of the previous step.
            dt_next = t_next - t_curr
            
            # If t_next == t_curr, dt_next is 0 (we do not draft)
            def draft_step():
                drift_spec = self.sde_model.drift(t_curr, mu_spec_prev, s_prob_next)
                return mu_spec_prev + dt_next * drift_spec
                
            mu_spec_next_raw = jax.lax.cond(
                t_next > t_curr,
                draft_step,
                lambda: mu_spec_prev
            )
            
            # --- 3. Verify Speculation of Step k ---
            # Compare the true target prediction against the speculated mean we used
            error = jnp.linalg.norm(mu_pred_target - mu_spec_prev)
            is_accepted = error < self.tolerance
            
            # --- 4. Kalman Update ---
            y = x_curr - self.C @ mu_pred_target
            S = self.C @ P_pred_target @ self.C.T + self.R + 1e-4 * jnp.eye(self.R.shape[0])
            
            cho_S, lower = jax.scipy.linalg.cho_factor(S)
            K_gain = jax.scipy.linalg.cho_solve((cho_S, lower), self.C @ P_pred_target).T
            
            mu_filt_curr = mu_pred_target + K_gain @ y
            P_filt_curr = P_pred_target - K_gain @ self.C @ P_pred_target
            P_filt_curr = 0.5 * (P_filt_curr + P_filt_curr.T)
            
            # --- 5. Speculative Taylor Correction ---
            # Difference introduced by the Kalman update correction
            delta = mu_filt_curr - mu_spec_prev
            
            # Compute local transition matrix to project correction forward: Phi_t = I + dt * expected_A
            state_jacobians = self.sde_model.drift.get_jacobians(mu_filt_curr)
            expected_A = jnp.average(state_jacobians, weights=s_prob_next, axis=0)
            Phi_t = jnp.eye(D_z) + dt_next * expected_A
            
            # Apply correction to the pre-drafted state
            mu_spec_next_corrected = mu_spec_next_raw + Phi_t @ delta
            
            # If speculation is rejected, we fall back to using the target prediction
            mu_spec_next = jax.lax.cond(
                is_accepted,
                lambda: mu_spec_next_corrected,
                lambda: mu_pred_target
            )
            
            return (mu_filt_curr, P_filt_curr, mu_spec_next, t_curr, is_accepted), (mu_filt_curr, P_filt_curr, is_accepted)

        # Initial Update at t=0
        y_0 = xs[0] - self.C @ mu_0
        S_0 = self.C @ P_0 @ self.C.T + self.R + 1e-4 * jnp.eye(self.R.shape[0])
        cho_S_0, lower_0 = jax.scipy.linalg.cho_factor(S_0)
        K_0 = jax.scipy.linalg.cho_solve((cho_S_0, lower_0), self.C @ P_0).T
        mu_0_upd = mu_0 + K_0 @ y_0
        P_0_upd = P_0 - K_0 @ self.C @ P_0
        P_0_upd = 0.5 * (P_0_upd + P_0_upd.T)
        
        # Initial speculation: draft t=0 to t=1 starting from mu_0
        dt_0 = ts[1] - ts[0]
        drift_0 = self.sde_model.drift(ts[0], mu_0_upd, expected_s[1])
        mu_spec_0 = mu_0_upd + dt_0 * drift_0
        
        # Carry carries the filter state, the speculated next-step mean, time, and whether accepted
        carry_init = (mu_0_upd, P_0_upd, mu_spec_0, ts[0], True)
        
        # Prepare inputs for scan
        # We append ts[-1] to ts[1:] as t_next for the last step so shapes match
        ts_next = jnp.append(ts[2:], ts[-1])
        s_prob_next = jnp.vstack([expected_s[2:], expected_s[-1][None, ...]])
        
        scan_inputs = (
            xs[1:],
            ts[1:],
            ts_next,
            expected_s[1:],
            s_prob_next
        )
        
        # Scan from t=1 to T-1
        _, (mu_f, P_f, accepts) = jax.lax.scan(
            step, carry_init, scan_inputs
        )
        
        mu_filt = jnp.vstack([mu_0_upd[None, ...], mu_f])
        P_filt = jnp.vstack([P_0_upd[None, ...], P_f])
        accept_flags = jnp.append(jnp.array([True]), accepts)
        
        return mu_filt, P_filt, accept_flags
