import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from acm.physics.propagator import rk4_step, rk4_step_batch
from acm.models.collision_predictor import predictor

class ConjunctionScreening:
    def __init__(self):
        self.debris_tree = None
        self.debris_ids = []
        self.debris_states = None  # (N, 6)
        # Cache for debris propagation
        self._cached_t_start = None
        self._cached_horizon = None
        self._cached_dt = None
        self._cached_debris_trajectories = None # (n_steps, N, 3)
        
    def update_debris(self, debris_dict: dict):
        """
        Rebuilds the KD-Tree from current debris states.
        """
        if not debris_dict:
            return
            
        self.debris_ids = list(debris_dict.keys())
        self.debris_states = np.array([debris_dict[i] for i in self.debris_ids])
        self.debris_tree = KDTree(self.debris_states[:, :3])
        # Clear cache when debris changes
        self._cached_t_start = None

    def _get_debris_trajectories(self, debris_states, t_start, horizon_s, dt_scan):
        """
        Internal helper to propagate debris using Optimized Euler for coarse scan.
        """
        if (self._cached_t_start == t_start and 
            self._cached_horizon == horizon_s and 
            self._cached_dt == dt_scan and 
            len(self.debris_ids) == len(debris_states)):
            return self._cached_debris_trajectories

        n_steps = int(horizon_s / dt_scan)
        d_curr = debris_states.copy()
        traj = np.zeros((n_steps, len(debris_states), 3))
        
        from acm.physics.propagator import MU, J2, RE
        J2_FACTOR = 1.5 * J2 * MU * RE**2

        for i in range(n_steps):
            r = d_curr[:, :3]
            v = d_curr[:, 3:]
            traj[i] = r
            
            r_sq = np.sum(r**2, axis=1, keepdims=True)
            r_norm = np.sqrt(r_sq)
            
            # 2-body
            inv_r3 = 1.0 / (r_sq * r_norm)
            a = -(MU * inv_r3) * r
            
            # J2
            factor = J2_FACTOR / (r_sq * r_sq * r_norm)
            common = 5.0 * r[:, 2:3]**2 / r_sq - 1.0
            
            a += (factor * common) * r
            a[:, 2:3] -= (2.0 * factor) * r[:, 2:3]
            
            # Euler Update (Semi-implicit)
            d_curr[:, 3:] += dt_scan * a
            d_curr[:, :3] += dt_scan * d_curr[:, 3:]
            
        self._cached_t_start = t_start
        self._cached_horizon = horizon_s
        self._cached_dt = dt_scan
        self._cached_debris_trajectories = traj
        return traj
        
    def long_range_scan(self, sat_state: np.ndarray, debris_states: np.ndarray, 
                        t_start: float, horizon_s: float = 86400.0, 
                        dt_scan: float = 600.0):
        """
        Vectorized scan for candidates over 24h with caching and pre-filtering.
        """
        if debris_states is None or len(debris_states) == 0:
            return []

        from acm.physics.propagator import MU
        # 1. Quick Altitude Filter
        r_sat_mag = np.linalg.norm(sat_state[:3])
        v_sat_mag = np.linalg.norm(sat_state[3:])
        en = v_sat_mag**2 / 2 - MU / r_sat_mag
        a_sat = -MU / (2 * en)
        h_sat = np.linalg.norm(np.cross(sat_state[:3], sat_state[3:]))
        e_sat = np.sqrt(max(0, 1 + 2 * en * h_sat**2 / MU**2))
        
        sat_min = a_sat * (1 - e_sat) - 100.0
        sat_max = a_sat * (1 + e_sat) + 100.0
        
        d_r_mags = np.linalg.norm(debris_states[:, :3], axis=1)
        d_v_mags = np.linalg.norm(debris_states[:, 3:], axis=1)
        d_en = d_v_mags**2 / 2 - MU / d_r_mags
        d_a = -MU / (2 * d_en)
        d_h = np.linalg.norm(np.cross(debris_states[:, :3], debris_states[:, 3:]), axis=1)
        d_e = np.sqrt(np.maximum(0.0, 1 + 2 * d_en * d_h**2 / MU**2))
        
        d_min = d_a * (1 - d_e)
        d_max = d_a * (1 + d_e)
        
        mask = (d_max >= sat_min) & (d_min <= sat_max)
        filtered_indices = np.where(mask)[0]
        
        if len(filtered_indices) == 0:
            return []
            
        # 2. Get trajectories (cached)
        all_traj = self._get_debris_trajectories(debris_states, t_start, horizon_s, dt_scan)
        d_trajs = all_traj[:, filtered_indices, :3]
        
        # 3. Propagate satellite and check distance
        n_steps = len(d_trajs)
        s_curr = sat_state.reshape(1, 6)
        curr_t = t_start
        
        candidates = set()
        for i in range(n_steps):
            diff = d_trajs[i] - s_curr[:, :3]
            sq_dists = np.sum(diff**2, axis=1)
            
            near_idx = np.where(sq_dists < 2500.0)[0] # 50km threshold
            for idx in near_idx:
                candidates.add(filtered_indices[idx])
                
            s_curr = rk4_step_batch(s_curr, dt_scan, curr_t)
            curr_t += dt_scan
            
        return list(candidates)

    def predict_risk(self, sat_state: np.ndarray, deb_state: np.ndarray) -> float:
        """
        Uses the trained XGBoost model to predict collision risk.
        Mocks CDM features from current relative state.
        """
        if predictor is None:
            return 1.0 # Default to high risk if model not loaded
            
        # Compute some basic features that the model expects
        rel_pos = sat_state[:3] - deb_state[:3]
        rel_vel = sat_state[3:] - deb_state[3:]
        miss_dist = np.linalg.norm(rel_pos)
        rel_vel_mag = np.linalg.norm(rel_vel)
        
        # Create a mock feature dictionary
        # We only populate what we can, predictor will pad the rest with 0s
        features = {
            'miss_distance': miss_dist,
            'relative_speed': rel_vel_mag,
            'relative_position_x': rel_pos[0],
            'relative_position_y': rel_pos[1],
            'relative_position_z': rel_pos[2],
            'relative_velocity_x': rel_vel[0],
            'relative_velocity_y': rel_vel[1],
            'relative_velocity_z': rel_vel[2],
        }
        
        df = pd.DataFrame([features])
        probs = predictor.predict_risk(df)
        return float(probs[0])

    def find_tca(self, sat_state: np.ndarray, deb_state: np.ndarray, 
                 t_start: float, horizon_s: float = 86400.0, 
                 dt_coarse: float = 60.0, skip_filter: bool = False):
        """
        Improved TCA finder using a two-pass approach.
        """
        # 1. Coarse Pass
        s_curr = sat_state.copy()
        d_curr = deb_state.copy()
        
        min_dist = float('inf')
        tca_coarse = t_start
        
        steps = int(horizon_s / dt_coarse)
        
        for i in range(steps):
            t_curr = t_start + i * dt_coarse
            dist = float(np.linalg.norm(s_curr[:3] - d_curr[:3]))
            
            if dist < min_dist:
                min_dist = dist
                tca_coarse = t_curr
            
            s_curr = rk4_step(s_curr, dt_coarse, t_curr)
            d_curr = rk4_step(d_curr, dt_coarse, t_curr)

        # 2. ML Pre-filter
        if not skip_filter:
            if min_dist > 50.0:
                return None

            # Safety-First: Bypass ML if miss distance is already critically low (< 1km)
            if min_dist > 1.0:
                dt_to_tca = tca_coarse - t_start
                s_tca_coarse = rk4_step(sat_state, dt_to_tca, t_start) 
                d_tca_coarse = rk4_step(deb_state, dt_to_tca, t_start)
                
                risk_prob = self.predict_risk(s_tca_coarse, d_tca_coarse)
                
                if risk_prob < 0.5:
                    if min_dist > 10.0:
                        return None

        # 3. Refine using iterative search around tca_coarse (1ms precision)
        def get_dist(t) -> float:
            dt = t - t_start
            s = rk4_step(sat_state, dt, t_start)
            d = rk4_step(deb_state, dt, t_start)
            return float(np.linalg.norm(s[:3] - d[:3]))

        t_best = tca_coarse
        d_best = float(min_dist)
        
        step = dt_coarse / 2.0
        while step > 0.001: # 1ms precision (Upgraded from 100ms)
            d_plus = get_dist(t_best + step)
            d_minus = get_dist(t_best - step)
            
            if d_plus < d_best:
                d_best = float(d_plus)
                t_best += step
            elif d_minus < d_best:
                d_best = float(d_minus)
                t_best -= step
            else:
                step /= 2.0
                
        return t_best, d_best

