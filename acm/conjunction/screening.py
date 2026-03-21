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
        
    def update_debris(self, debris_dict: dict):
        """
        Rebuilds the KD-Tree from current debris states.
        """
        if not debris_dict:
            return
            
        self.debris_ids = list(debris_dict.keys())
        self.debris_states = np.array([debris_dict[i] for i in self.debris_ids])
        self.debris_tree = KDTree(self.debris_states[:, :3])
        
    def long_range_scan(self, sat_state: np.ndarray, debris_states: np.ndarray, 
                        t_start: float, horizon_s: float = 86400.0, 
                        dt_scan: float = 600.0):
        """
        Vectorized scan for candidates over 24h.
        Propagates debris in batch and checks distance to satellite path.
        """
        # 1. Propagate satellite in steps
        # 2. Propagate debris in steps
        # 3. Find any pairs that ever get within 50km
        
        n_steps = int(horizon_s / dt_scan)
        s_curr = sat_state.reshape(1, 6)
        d_curr = debris_states.copy()
        
        candidates = set()
        
        for _ in range(n_steps):
            # Distance check (Vectorized)
            # s_curr is (1, 6), d_curr is (N, 6)
            diff = d_curr[:, :3] - s_curr[:, :3]
            dists = np.linalg.norm(diff, axis=1)
            
            # Any within 50km?
            near_idx = np.where(dists < 50.0)[0]
            for idx in near_idx:
                candidates.add(idx)
                
            # Propagate both
            s_curr = rk4_step_batch(s_curr, dt_scan)
            d_curr = rk4_step_batch(d_curr, dt_scan)
            
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
                 dt_coarse: float = 60.0):
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
            dist = np.linalg.norm(s_curr[:3] - d_curr[:3])
            
            if dist < min_dist:
                min_dist = dist
                tca_coarse = t_curr
            
            s_curr = rk4_step(s_curr, dt_coarse)
            d_curr = rk4_step(d_curr, dt_coarse)

        # 2. ML Pre-filter
        # Check risk at the point of closest approach found in coarse pass
        if min_dist > 50.0:
            return None
            
        # Re-propagate to tca_coarse to get exact states for ML
        dt_to_tca = tca_coarse - t_start
        s_tca_coarse = rk4_step(sat_state, dt_to_tca) # Simplified, should use steps
        d_tca_coarse = rk4_step(deb_state, dt_to_tca)
        
        risk_prob = self.predict_risk(s_tca_coarse, d_tca_coarse)
        
        # If ML model predicts low risk, we skip expensive refinement
        if risk_prob < 0.5:
            # But wait, we should be conservative. 
            # If min_dist is already very small, we might still want to refine.
            if min_dist > 10.0:
                return None

        # 3. Refine using iterative search around tca_coarse
        def get_dist(t):
            dt = t - t_start
            s = rk4_step(sat_state, dt)
            d = rk4_step(deb_state, dt)
            return np.linalg.norm(s[:3] - d[:3])

        t_best = tca_coarse
        d_best = min_dist
        
        step = dt_coarse / 2.0
        while step > 0.1: # 0.1s precision
            d_plus = get_dist(t_best + step)
            d_minus = get_dist(t_best - step)
            
            if d_plus < d_best:
                d_best = d_plus
                t_best += step
            elif d_minus < d_best:
                d_best = d_minus
                t_best -= step
            else:
                step /= 2.0
                
        return t_best, d_best

