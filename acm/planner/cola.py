import numpy as np
import math
from typing import List

from acm.physics.propagator import rk4_step, MU, RE, propagate_state
from acm.physics.frames import dv_rtn_to_eci, eci_to_ecef, rtn_to_eci_matrix
from acm.physics.maneuver import compute_phasing_burns, apply_burn
from acm.state import SatelliteState, CDM, Maneuver
from acm.data.stations import GROUND_STATIONS

# Constants
g0 = 9.80665   # m/s^2 (Standard Gravity)
ISP = 300.0    # s (Specific Impulse)
MAX_DV = 0.015 # km/s (15 m/s)
COOLDOWN = 600.0 # seconds

def compute_elevation(gs: dict, r_eci: np.ndarray, t_unix: float) -> float:
    """
    Computes elevation angle (degrees) of satellite from ground station.
    """
    r_ecef = eci_to_ecef(r_eci, t_unix)
    
    # GS Position in ECEF (Approx spherical or wgs84? Use simple spherical for elevation check)
    # Better: Use geodetic to ECEF properly.
    lat_rad = math.radians(gs['lat'])
    lon_rad = math.radians(gs['lon'])
    alt_km = gs['alt'] / 1000.0
    
    # GS ECEF
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)
    
    # N (Prime Vertical Radius)
    a = RE
    e2 = 0.00669437999014
    N = a / math.sqrt(1 - e2 * sin_lat**2)
    
    x_gs = (N + alt_km) * cos_lat * cos_lon
    y_gs = (N + alt_km) * cos_lat * sin_lon
    z_gs = (N * (1 - e2) + alt_km) * sin_lat
    
    gs_ecef = np.array([x_gs, y_gs, z_gs])
    
    # Vector from GS to Sat
    rho = r_ecef - gs_ecef
    range_dist = np.linalg.norm(rho)
    
    # Up vector at GS (normal to ellipsoid)
    # Approx: just normalize gs_ecef? No, depends on ellipsoid.
    # Correct Up vector:
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    
    sin_el = np.dot(rho, up) / range_dist
    return math.degrees(math.asin(sin_el))

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
    Used for blind conjunctions.
    """
    # Scan back in 30s steps?
    step = 30.0
    
    # Need to propagate sat state to 't' to check LOS?
    # Or assume we have the state history? We don't.
    # We have current state 'sat.state_vector' at 'state_manager.last_timestamp'.
    # We need to propagate from 'current' to 't'.
    
    # This is expensive if we do it for every step.
    # Better: Propagate forward from current to t_target, recording LOS windows.
    
    curr_t = t_start_search
    state = sat.state_vector.copy()
    
    last_valid_t = -1.0
    
    while curr_t < t_target:
        if has_los(state[:3], curr_t):
            last_valid_t = curr_t
        
        state = rk4_step(state, step)
        curr_t += step
        
    return last_valid_t

def check_collision_during_burn(sat_id: str, new_state: np.ndarray, burn_time: float) -> bool:
    """
    Checks if a planned burn results in a new collision risk.
    Uses the KD-Tree for a fast check.
    """
    from acm.api.telemetry import screener
    
    if screener.debris_tree is None:
        return False
        
    # Query for any debris within 1km of the new state
    # This is an instantaneous check at burn time.
    indices = screener.debris_tree.query_ball_point(new_state[:3], r=1.0)
    return len(indices) > 0

def plan_evasion(sat: SatelliteState, cdm: CDM, current_time: float) -> List[Maneuver]:
    """
    Plans a robust 3-burn evasion and recovery sequence.
    """
    maneuvers = []
    
    # 1. Determine Evasion Burn Time (at least 10s from now, respecting cooldown)
    earliest_possible = max(current_time + 10.0, sat.last_burn_time + COOLDOWN)
    
    # Target 30 mins before TCA
    t_target = cdm.tca - 1800.0
    burn_time = max(earliest_possible, t_target)
    
    if burn_time >= cdm.tca: 
        # If we're already past TCA or too close, try immediate burn
        burn_time = earliest_possible
        if burn_time >= cdm.tca: return []

    # Get state at burn_time
    sat_state = np.concatenate([[sat.r['x'], sat.r['y'], sat.r['z']], 
                                [sat.v['x'], sat.v['y'], sat.v['z']]])
    state_burn = propagate_state(sat_state, burn_time - current_time)
    
    # Check LOS at burn time
    if not has_los(state_burn[:3], burn_time):
        # Scan back for last LOS window to upload commands
        last_los = find_last_los_window(sat, burn_time, current_time)
        if last_los == -1.0:
            # No LOS between now and burn. If we HAVE LOS now, we can schedule.
            if not has_los(sat_state[:3], current_time):
                return []
        # Proceed: Commands are "uploaded" during LOS and "executed" at burn_time autonomously.

    # Evasion Burn (Prograde 2 m/s)
    dv_evade_mag = 0.002 
    dv_evade_rtn = np.array([0.0, dv_evade_mag, 0.0])
    dv_evade_eci = dv_rtn_to_eci(dv_evade_rtn, state_burn[:3], state_burn[3:])
    
    # Safety Check
    state_post_burn = apply_burn(state_burn, dv_evade_eci.tolist())
    if check_collision_during_burn(sat.id, state_post_burn, burn_time):
        # Try Retrograde instead
        dv_evade_rtn[1] = -0.002
        dv_evade_eci = dv_rtn_to_eci(dv_evade_rtn, state_burn[:3], state_burn[3:])
        state_post_burn = apply_burn(state_burn, dv_evade_eci.tolist())
        if check_collision_during_burn(sat.id, state_post_burn, burn_time):
            return [] # Both failed, abort
    
    maneuvers.append(Maneuver(
        burn_id=f"EVADE_{cdm.deb_id}",
        burn_time=burn_time,
        dv_eci=dv_evade_eci.tolist(),
        type="EVASION"
    ))
    
    # 2. Recovery (2-burn Phasing)
    # Start recovery 30 mins after TCA
    t_rec_start = max(cdm.tca + 1800.0, burn_time + COOLDOWN)
    
    # State post-evasion at t_rec_start
    state_post = apply_burn(state_burn, dv_evade_eci.tolist())
    state_rec = propagate_state(state_post, t_rec_start - burn_time)
    
    # Nominal Slot at t_rec_start
    slot_now = sat.nominal_slot if sat.nominal_slot is not None else sat_state
    slot_rec = propagate_state(slot_now, t_rec_start - current_time)
    
    # Calculate Phase Gap
    r_slot = slot_rec[:3]
    v_slot = slot_rec[3:]
    M_eci_rtn = rtn_to_eci_matrix(r_slot, v_slot).T
    
    dr_rtn = M_eci_rtn @ (state_rec[:3] - r_slot)
    dv_rtn = M_eci_rtn @ (state_rec[3:] - v_slot)
    
    n = np.sqrt(MU / np.linalg.norm(r_slot)**3)
    T_phase = 5800.0 # Return in approx 1 orbit
    
    dv1_rtn, dv2_rtn = compute_phasing_burns(dr_rtn, dv_rtn, n, T_phase)
    
    # Transform to ECI
    dv1_eci = dv_rtn_to_eci(dv1_rtn, state_rec[:3], state_rec[3:])
    
    maneuvers.append(Maneuver(
        burn_id=f"REC1_{cdm.deb_id}",
        burn_time=t_rec_start,
        dv_eci=dv1_eci.tolist(),
        type="RECOVERY"
    ))
    
    # Propagate to end of phasing for 2nd burn
    state_rec2_pre = apply_burn(state_rec, dv1_eci.tolist())
    state_rec2 = propagate_state(state_rec2_pre, T_phase)
    
    dv2_eci = dv_rtn_to_eci(dv2_rtn, state_rec2[:3], state_rec2[3:])
    
    maneuvers.append(Maneuver(
        burn_id=f"REC2_{cdm.deb_id}",
        burn_time=t_rec_start + T_phase,
        dv_eci=dv2_eci.tolist(),
        type="RECOVERY"
    ))
    
    return maneuvers
