from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

from datetime import datetime
from acm.state import state_manager, SatelliteState, Maneuver, CDM
from acm.physics.propagator import rk4_step_batch, propagate_state, MU
from acm.physics.maneuver import compute_dm, apply_burn
from acm.planner.cola import plan_evasion

router = APIRouter()

class SimStepRequest(BaseModel):
    step_seconds: float

def propagate_satellite_step(sat: SatelliteState, t_start: float, t_end: float):
    """
    Propagates satellite from t_start to t_end, executing any burns in between.
    """
    current_t = t_start
    final_t = t_end
    
    queue = sat.maneuver_queue
    pending = [b for b in queue if b.status == "SCHEDULED" and t_start < b.burn_time <= t_end]
    pending.sort(key=lambda x: x.burn_time)
    
    executed_count = 0
    if sat.state_vector is None: 
        return 0
    state = sat.state_vector.copy()
    
    for b in pending:
        # Propagate to burn time
        dt = b.burn_time - current_t
        state = propagate_state(state, dt)
                
        # Apply Burn
        dv_mag = np.linalg.norm(b.dv_eci)
        current_total_mass = 500.0 + sat.fuel_kg
        dm = compute_dm(current_total_mass, dv_mag)
        
        if sat.fuel_kg >= dm:
            state = apply_burn(state, b.dv_eci)
            sat.fuel_kg -= dm
            sat.last_burn_time = b.burn_time
            b.status = "EXECUTED"
            executed_count += 1
        else:
            b.status = "FAILED_NO_FUEL"
            
        current_t = b.burn_time
        
    # Propagate remainder
    state = propagate_state(state, final_t - current_t)
    sat.state_vector = state
    return executed_count

@router.post("/simulate/step")
def simulate_step(payload: SimStepRequest):
    """
    Advances simulation by step_seconds.
    """
    dt = payload.step_seconds
    t_start = state_manager.last_timestamp
    t_end = t_start + dt
    
    # 1. Propagate Satellites (and Execute Burns)
    total_burns = 0
    with state_manager.lock:
        for sat in state_manager.satellites.values():
            if sat.state_vector is None: 
                continue
            
            # Propagate Satellite
            total_burns += propagate_satellite_step(sat, t_start, t_end)
            
            # Propagate Nominal Slot
            if sat.nominal_slot is not None:
                sat.nominal_slot = propagate_state(sat.nominal_slot, dt)
                
                # Check Uptime (10km box)
                dist = np.linalg.norm(sat.state_vector[:3] - sat.nominal_slot[:3])
                if dist > 10.0:
                    sat.outage_seconds += dt
                    sat.status = "OUT_OF_SLOT"
                else:
                    sat.status = "NOMINAL"
            
    # 2. Propagate Debris (Vectorized)
    with state_manager.lock:
        debris_dict = state_manager.debris
        if debris_dict:
            ids = list(debris_dict.keys())
            states = np.array([debris_dict[i] for i in ids])
            
            # Batch Propagation (30s steps)
            steps = int(dt / 30.0)
            for _ in range(steps):
                states = rk4_step_batch(states, 30.0)
            rem = dt - steps*30.0
            if rem > 0:
                states = rk4_step_batch(states, rem)
                
            for i, did in enumerate(ids):
                debris_dict[did] = states[i]

    # 3. Collision Detection & Autonomous Planning
    from acm.api.telemetry import screener
    screener.update_debris(state_manager.debris)
    
    collisions = 0
    sim_time = t_end
    
    # Instantaneous collision check
    for sat in state_manager.satellites.values():
        if sat.state_vector is None: 
            continue
        candidates = screener.debris_tree.query_ball_point(sat.state_vector[:3], r=0.1)
        collisions += len(candidates)

    # Long-range autonomous planning (Every 10 mins)
    LAST_SCAN_TIME = state_manager.last_scan_time
    if sim_time - LAST_SCAN_TIME >= 600.0 or LAST_SCAN_TIME == 0.0:
        state_manager.last_scan_time = sim_time
        state_manager.cdms = [] # Refresh active warnings
        
        debris_dict = state_manager.debris
        if debris_dict:
            deb_ids = list(debris_dict.keys())
            deb_states = np.array([debris_dict[did] for did in deb_ids])
            
            for sat in state_manager.satellites.values():
                if sat.state_vector is None: 
                    continue
                
                # Scan for threats over next 24h
                indices = screener.long_range_scan(sat.state_vector, deb_states, t_start=sim_time)
                for idx in indices:
                    deb_id = deb_ids[idx]
                    result = screener.find_tca(sat.state_vector, deb_states[idx], t_start=sim_time)
                    if result:
                        tca, dist = result
                        if dist < 5.0: # Track all within 5km
                            cdm = CDM(sat_id=sat.id, deb_id=deb_id, tca=tca, miss_distance=dist, is_critical=(dist < 0.1))
                            state_manager.cdms.append(cdm)
                            
                            if dist < 0.1:
                                has_plan = any(m.burn_id == f"EVADE_{deb_id}" for m in sat.maneuver_queue)
                                if not has_plan:
                                    plans = plan_evasion(sat, cdm, sim_time)
                                    if plans:
                                        with state_manager.lock:
                                            sat.maneuver_queue.extend(plans)
                                            sat.maneuver_queue.sort(key=lambda x: x.burn_time)
                                        state_manager.add_log(f"COLA: Scheduled 3-burn evasion for {sat.id} vs {deb_id}")


    # EOL logic
    for sat in state_manager.satellites.values():
        if sat.fuel_kg < 2.5 and sat.status != "EOL_GRAVEYARD":
            r_mag = np.linalg.norm(sat.state_vector[:3])
            v_mag = np.sqrt(MU / r_mag)
            dv_mag = v_mag * (np.sqrt((2*(r_mag + 200)) / (2*r_mag + 200)) - 1)
            from acm.physics.frames import dv_rtn_to_eci
            dv_eci = dv_rtn_to_eci(np.array([0, dv_mag, 0]), sat.state_vector[:3], sat.state_vector[3:])
            eol_burn = Maneuver(burn_id=f"EOL_{sat.id}", burn_time=sim_time + 600.0, dv_eci=dv_eci.tolist(), type="EOL")
            with state_manager.lock:
                sat.maneuver_queue.append(eol_burn)
                sat.maneuver_queue.sort(key=lambda x: x.burn_time)
                sat.status = "EOL_GRAVEYARD"

    state_manager.last_timestamp = t_end
    iso_time = datetime.fromtimestamp(t_end).isoformat() + "Z"
    
    return {
        "status": "STEP_COMPLETE",
        "new_timestamp": iso_time,
        "collisions_detected": collisions,
        "maneuvers_executed": total_burns
    }


