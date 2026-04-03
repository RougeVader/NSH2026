import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import numpy as np
import time
import pandas as pd
import math
from acm.conjunction.screening import ConjunctionScreening
from acm.physics.propagator import rk4_step, MU
from acm.planner.cola import plan_evasion
from acm.state import SatelliteState, CDM

def benchmark_10k():
    print("--- Performance Benchmark: 10,000 Objects ---")
    screener = ConjunctionScreening()
    N = 10000
    debris_states = np.zeros((N, 6))
    for i in range(N):
        # Distributed in LEO shells 300-1000km
        r_mag = 6378.137 + 300 + np.random.rand()*700
        # Random unit vector
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        debris_states[i, :3] = vec * r_mag
        # Approx circular velocity
        v_mag = np.sqrt(MU / r_mag)
        # Perpendicular velocity
        v_vec = np.random.randn(3)
        v_vec = np.cross(vec, v_vec)
        v_vec /= np.linalg.norm(v_vec)
        debris_states[i, 3:] = v_vec * v_mag
        
    sat_state = np.array([6700.0, 0.0, 0.0, 0.0, 7.7, 0.0])
    
    # 1. KD-Tree Build
    start = time.time()
    debris_dict = {f"DEB-{i}": debris_states[i] for i in range(N)}
    screener.update_debris(debris_dict)
    end = time.time()
    print(f"KD-Tree Build (10k): {end - start:.4f}s")
    
    # 2. Long Range Scan (24h, 10min steps)
    start = time.time()
    candidates = screener.long_range_scan(sat_state, debris_states, t_start=0, horizon_s=86400, dt_scan=600)
    end = time.time()
    print(f"24h Long-Range Scan (10k debris): {end - start:.4f}s")
    
    # 3. Precise TCA finding (for one candidate)
    if candidates:
        deb_idx = candidates[0]
    else:
        deb_idx = 0
        
    start = time.time()
    screener.find_tca(sat_state, debris_states[deb_idx], t_start=0)
    end = time.time()
    print(f"Precise TCA Refinement (1ms precision): {end - start:.4f}s")

def case_study_collision():
    print("\n--- Case Study: Collision Course & Evasion ---")
    # Use J2000 epoch for t_start to ensure correct GMST/LOS math
    t_start_sim = 946728000.0 
    
    # Satellite at 500km altitude
    r_mag = 6378.137 + 500
    v_mag = np.sqrt(MU / r_mag)
    
    # Sat 1: Nominal - Start over Bangalore (approx 13N, 77E)
    lat, lon = math.radians(13.0), math.radians(77.0)
    rx = r_mag * math.cos(lat) * math.cos(lon)
    ry = r_mag * math.cos(lat) * math.sin(lon)
    rz = r_mag * math.sin(lat)
    
    # We need to rotate rx, ry, rz back from ECEF to ECI at t_start_sim
    # to ensure it's actually over Bangalore at that time.
    from acm.physics.frames import eci_to_ecef_matrix
    M_eci_ecef = eci_to_ecef_matrix(t_start_sim)
    r_ecef = np.array([rx, ry, rz])
    r_eci = M_eci_ecef.T @ r_ecef # ECEF to ECI is transpose
    
    # Simple polar-ish orbit for velocity
    # Construct a velocity vector perpendicular to r_eci
    # Let's just pick one
    v_eci = np.cross(r_eci, np.array([0, 0, 1]))
    v_eci = (v_eci / np.linalg.norm(v_eci)) * v_mag
    
    sat_s = np.concatenate([r_eci, v_eci])
    
    sat = SatelliteState(id="FLEET-01", r={'x':r_eci[0], 'y':r_eci[1], 'z':r_eci[2]}, 
                         v={'x':v_eci[0], 'y':v_eci[1], 'z':v_eci[2]})
    sat.state_vector = sat_s
    sat.nominal_slot = sat_s.copy()
    
    # Debris: Head-on collision in 30 mins
    tca_target = t_start_sim + 1800.0
    # Propagate sat to tca
    from acm.physics.propagator import propagate_state
    sat_at_tca = propagate_state(sat_s, 1800.0, t_start=t_start_sim)
    
    # Debris at tca = same position, opposite velocity
    deb_at_tca = sat_at_tca.copy()
    deb_at_tca[3:] = -sat_at_tca[3:] # Head on!
    
    # Back-propagate debris to t_start_sim
    from acm.physics.propagator import rk4_step
    deb_s = deb_at_tca.copy()
    curr_t = tca_target
    dt = -60.0
    for _ in range(30):
        deb_s = rk4_step(deb_s, dt, curr_t)
        curr_t += dt
        
    print(f"Initial Miss Distance (at start): {np.linalg.norm(sat_s[:3] - deb_s[:3]):.4f} km")
    
    # Verify TCA
    screener = ConjunctionScreening()
    res = screener.find_tca(sat_s, deb_s, t_start_sim)
    if res:
        tca, dist = res
        print(f"Detected TCA: {tca - t_start_sim:.2f}s from start, Miss Distance: {dist*1000:.2f} m")
        
        # Plan Evasion
        cdm = CDM(sat_id=sat.id, deb_id="DEB-999", tca=tca, miss_distance=dist, is_critical=True)
        maneuvers = plan_evasion(sat, cdm, t_start_sim)
        
        print(f"Maneuvers Planned: {len(maneuvers)}")
        for m in maneuvers:
            dv_mag = np.linalg.norm(m.dv_eci)
            print(f"  - {m.type} at T+{m.burn_time - t_start_sim:.0f}s: DV = {dv_mag*1000:.2f} m/s")
            
        # Verify Safety after Evasion
        if len(maneuvers) > 0:
            m_evade = maneuvers[0]
            # Propagate sat to burn time
            sat_at_burn = propagate_state(sat_s, m_evade.burn_time - t_start_sim, t_start=t_start_sim)
            sat_post_burn = sat_at_burn.copy()
            sat_post_burn[3:] += np.array(m_evade.dv_eci)
            
            # Propagate debris to same burn time
            deb_at_burn = propagate_state(deb_s, m_evade.burn_time - t_start_sim, t_start=t_start_sim)
            
            # Check new TCA
            res_new = screener.find_tca(sat_post_burn, deb_at_burn, m_evade.burn_time, horizon_s=3600, skip_filter=True)
            if res_new:
                tca_n, dist_n = res_new
                print(f"Post-Evasion Miss Distance: {dist_n:.4f} km")
                assert dist_n > 1.0 # Should be safe now

if __name__ == "__main__":
    benchmark_10k()
    case_study_collision()
