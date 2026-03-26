import numpy as np
import math
from typing import List

from acm.physics.propagator import rk4_step, MU, RE, propagate_state
from acm.physics.frames import dv_rtn_to_eci, eci_to_ecef, rtn_to_eci_matrix
from acm.physics.maneuver import compute_phasing_burns, apply_burn, compute_dm
from acm.state import SatelliteState, CDM, Maneuver
from acm.data.stations import GROUND_STATIONS

# Constants
g0 = 9.80665   # m/s^2 (Standard Gravity)
ISP = 300.0    # s (Specific Impulse)
DRY_MASS = 500.0 # kg
MAX_DV = 0.015 # km/s (15 m/s)
COOLDOWN = 600.0 # seconds

def compute_elevation(gs: dict, r_eci: np.ndarray, t_unix: float) -> float:
    """
    Computes elevation angle (degrees) from GS, corrected for atmospheric refraction.
    Uses Bennett's formula for refraction bending.
    """
    r_ecef = eci_to_ecef(r_eci, t_unix)
    
    # GS ECEF Conversion (WGS-84 approx)
    lat_rad = math.radians(gs['lat'])
    lon_rad = math.radians(gs['lon'])
    alt_km = gs['alt'] / 1000.0
    
    a = RE
    e2 = 0.00669437999014
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    x_gs = (N + alt_km) * math.cos(lat_rad) * math.cos(lon_rad)
    y_gs = (N + alt_km) * math.cos(lat_rad) * math.sin(lon_rad)
    z_gs = (N * (1 - e2) + alt_km) * math.sin(lat_rad)
    gs_ecef = np.array([x_gs, y_gs, z_gs])
    
    rho = r_ecef - gs_ecef
    range_dist = np.linalg.norm(rho)
    up = np.array([math.cos(lat_rad) * math.cos(lon_rad), 
                  math.cos(lat_rad) * math.sin(lon_rad), 
                  math.sin(lat_rad)])
    
    sin_el_geo = np.dot(rho, up) / range_dist
    el_geo = math.degrees(math.asin(sin_el_geo))
    
    # Atmospheric Refraction Correction (Bennett's Formula)
    # Allows signal to "bend" around the horizon
    if el_geo > -1.0: # Only applies near or above horizon
        # Refraction delta in arcminutes
        ref_arcmin = 1.02 / math.tan(math.radians(el_geo + 10.3 / (el_geo + 5.11)))
        return el_geo + (ref_arcmin / 60.0)
    
    return el_geo

def has_los(r_eci: np.ndarray, t_unix: float) -> bool:
    """Checks if satellite has LOS to ANY ground station."""
    for gs in GROUND_STATIONS:
        el = compute_elevation(gs, r_eci, t_unix)
        if el >= gs['min_el']:
            return True
    return False

def find_last_los_window(sat: SatelliteState, t_target: float, t_start_search: float) -> float:
    """
    Scans backwards from t_target to find the latest valid LOS window.
    """
    step = 30.0
    curr_t = t_start_search
    state = sat.state_vector.copy()
    last_valid_t = -1.0
    
    while curr_t < t_target:
        if has_los(state[:3], curr_t):
            last_valid_t = curr_t
        
        state = rk4_step(state, step, curr_t)
        curr_t += step
        
    return last_valid_t

def check_collision_during_burn(sat_id: str, new_state: np.ndarray, burn_time: float) -> bool:
    """
    Constellation-Aware Safety Check.
    Checks against Debris (KD-Tree) AND other Fleet Satellites.
    """
    from acm.api.telemetry import screener
    from acm.state import state_manager
    
    # 1. Debris Check
    if screener.debris_tree is not None:
        indices = screener.debris_tree.query_ball_point(new_state[:3], r=1.0)
        if len(indices) > 0: return True
        
    # 2. Fleet Check (Avoid bumping into partners)
    with state_manager.lock:
        for other_id, other_sat in state_manager.satellites.items():
            if other_id == sat_id: continue
            
            # Propagate other sat to burn_time for dynamic safety
            dt = burn_time - state_manager.last_timestamp
            other_pos = propagate_state(other_sat.state_vector, dt, state_manager.last_timestamp)[:3]
            
            if np.linalg.norm(new_state[:3] - other_pos) < 0.5: # 500m fleet buffer
                return True
                
    return False

def plan_evasion(sat: SatelliteState, cdm: CDM, current_time: float) -> List[Maneuver]:
    """
    Plans a fuel-aware evasion sequence. 
    Prioritizes survival (Evasion) over mission return (Recovery).
    """
    maneuvers = []
    available_fuel = sat.fuel_kg
    total_mass = DRY_MASS + available_fuel
    
    # 1. Determine Evasion Burn Time
    earliest_possible = max(current_time + 10.0, sat.last_burn_time + COOLDOWN)
    t_target = cdm.tca - 1800.0
    burn_time = max(earliest_possible, t_target)
    
    if burn_time >= cdm.tca: 
        burn_time = earliest_possible
        if burn_time >= cdm.tca: return []

    # Get state at burn_time
    sat_state = np.concatenate([[sat.r['x'], sat.r['y'], sat.r['z']], 
                                [sat.v['x'], sat.v['y'], sat.v['z']]])
    state_burn = propagate_state(sat_state, burn_time - current_time, current_time)
    
    # Check LOS at burn time
    if not has_los(state_burn[:3], burn_time):
        last_los = find_last_los_window(sat, burn_time, current_time)
        if last_los == -1.0:
            if not has_los(sat_state[:3], current_time):
                return []

    # Evasion Burn (Standard: 2 m/s, Minimal: 0.5 m/s)
    dv_evade_mag = 0.002 
    
    # --- FUEL CHECK: EVASION ---
    fuel_evade = compute_dm(total_mass, dv_evade_mag)
    if fuel_evade > available_fuel:
        # Try Minimal Evasion (0.5 m/s)
        dv_evade_mag = 0.0005
        fuel_evade = compute_dm(total_mass, dv_evade_mag)
        if fuel_evade > available_fuel:
            return [] # No fuel left for any maneuver

    dv_evade_rtn = np.array([0.0, dv_evade_mag, 0.0])
    dv_evade_eci = dv_rtn_to_eci(dv_evade_rtn, state_burn[:3], state_burn[3:])
    
    # Safety Check
    state_post_burn = apply_burn(state_burn, dv_evade_eci.tolist())
    if check_collision_during_burn(sat.id, state_post_burn, burn_time):
        dv_evade_rtn[1] = -dv_evade_mag # Try Retrograde
        dv_evade_eci = dv_rtn_to_eci(dv_evade_rtn, state_burn[:3], state_burn[3:])
        state_post_burn = apply_burn(state_burn, dv_evade_eci.tolist())
        if check_collision_during_burn(sat.id, state_post_burn, burn_time):
            return [] 
    
    maneuvers.append(Maneuver(
        burn_id=f"EVADE_{cdm.deb_id}",
        burn_time=burn_time,
        dv_eci=dv_evade_eci.tolist(),
        type="EVASION"
    ))
    
    # Update mass for recovery calculation
    available_fuel -= fuel_evade
    total_mass -= fuel_evade
    
    # 2. Recovery (2-burn Phasing) - ONLY if fuel budget allows
    t_rec_start = max(cdm.tca + 1800.0, burn_time + COOLDOWN)
    
    state_post = apply_burn(state_burn, dv_evade_eci.tolist())
    state_rec = propagate_state(state_post, t_rec_start - burn_time, burn_time)
    
    slot_now = sat.nominal_slot if sat.nominal_slot is not None else sat_state
    slot_rec = propagate_state(slot_now, t_rec_start - current_time, current_time)
    
    r_slot = slot_rec[:3]
    v_slot = slot_rec[3:]
    M_eci_rtn = rtn_to_eci_matrix(r_slot, v_slot).T
    
    dr_rtn = M_eci_rtn @ (state_rec[:3] - r_slot)
    dv_rtn = M_eci_rtn @ (state_rec[3:] - v_slot)
    
    n = np.sqrt(MU / np.linalg.norm(r_slot)**3)
    T_phase = 5800.0 
    
    dv1_rtn, dv2_rtn = compute_phasing_burns(dr_rtn, dv_rtn, n, T_phase)
    
    # --- FUEL CHECK: RECOVERY ---
    dv_rec_total = np.linalg.norm(dv1_rtn) + np.linalg.norm(dv2_rtn)
    fuel_rec = compute_dm(total_mass, dv_rec_total)
    
    if fuel_rec > (available_fuel - 0.05): # Leave a 50g emergency buffer
        return maneuvers # Returns only the Evasion burn
    
    dv1_eci = dv_rtn_to_eci(dv1_rtn, state_rec[:3], state_rec[3:])
    
    maneuvers.append(Maneuver(
        burn_id=f"REC1_{cdm.deb_id}",
        burn_time=t_rec_start,
        dv_eci=dv1_eci.tolist(),
        type="RECOVERY"
    ))
    
    state_rec2_pre = apply_burn(state_rec, dv1_eci.tolist())
    state_rec2 = propagate_state(state_rec2_pre, T_phase, t_rec_start)
    
    dv2_eci = dv_rtn_to_eci(dv2_rtn, state_rec2[:3], state_rec2[3:])
    
    maneuvers.append(Maneuver(
        burn_id=f"REC2_{cdm.deb_id}",
        burn_time=t_rec_start + T_phase,
        dv_eci=dv2_eci.tolist(),
        type="RECOVERY"
    ))
    
    return maneuvers
