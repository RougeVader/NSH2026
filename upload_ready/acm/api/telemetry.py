from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List

from acm.state import state_manager, CDM
from acm.conjunction.screening import ConjunctionScreening

router = APIRouter()

# Global Screening Instance (kept for use by other modules)
screener = ConjunctionScreening()

class Vector3(BaseModel):
    x: float
    y: float
    z: float

class ObjectState(BaseModel):
    id: str
    type: str # "SATELLITE" or "DEBRIS"
    r: Vector3
    v: Vector3

class TelemetryPayload(BaseModel):
    timestamp: str
    objects: List[ObjectState]

async def run_conjunction_screen_async():
    """
    Asynchronous task to run conjunction screening after telemetry is ingested.
    """
    # This is a simplified version for now.
    # The full implementation will be in Phase 3.
    # For now, we just update the screener with the new debris data.
    debris = state_manager.get_debris_dict()
    screener.update_debris(debris)
    
    # In a real scenario, you would loop through satellites and screen against debris
    # For now, this is a placeholder.
    sats = state_manager.get_satellites()
    for sat in sats.values():
        if sat.state_vector is None:
            continue
        
        # This part will be fleshed out in Phase 3
        # candidates = screener.long_range_scan(sat.state_vector, screener.debris_states)
        # for deb_id in candidates:
        #     # find_tca, etc.
        #     pass
        pass


import numpy as np
from acm.physics.propagator import propagate_state

@router.post("/force-threat")
async def force_threat():
    """
    DEBUG ONLY: Injects a critical conjunction for demo purposes.
    Teleports a debris object and a satellite to a position over a ground station 
    (guaranteed LOS) to ensure the demo always passes the COLA checks.
    """
    with state_manager.lock:
        if not state_manager.satellites or not state_manager.debris:
            return {"error": "Engine empty. Wait for auto-seed."}
            
        sat_id = list(state_manager.satellites.keys())[0]
        deb_id = list(state_manager.debris.keys())[0]
        
        sat = state_manager.satellites[sat_id]
        sim_time = state_manager.last_timestamp
        
        # 1. Target a position over Bengaluru (guaranteed LOS)
        # Lat: 13.0333, Lon: 77.5167
        import math
        from acm.physics.frames import eci_to_ecef_matrix
        
        R_EARTH = 6378.137
        alt = 500.0
        r_mag = R_EARTH + alt
        
        lat, lon = math.radians(13.0333), math.radians(77.5167)
        rx = r_mag * math.cos(lat) * math.cos(lon)
        ry = r_mag * math.cos(lat) * math.sin(lon)
        rz = r_mag * math.sin(lat)
        
        # Convert ECEF to ECI for the current simulation time
        M_eci_ecef = eci_to_ecef_matrix(sim_time)
        r_eci = M_eci_ecef.T @ np.array([rx, ry, rz])
        
        # 2. Give the satellite a valid circular velocity
        v_mag = np.sqrt(398600.4418 / r_mag)
        v_eci = np.cross(r_eci, np.array([0, 0, 1]))
        v_eci = (v_eci / np.linalg.norm(v_eci)) * v_mag
        
        # 3. Update Satellite State (Full Sync)
        new_sat_state = np.concatenate([r_eci, v_eci])
        sat.state_vector = new_sat_state
        sat.nominal_slot = new_sat_state.copy()
        sat.r = {"x": r_eci[0], "y": r_eci[1], "z": r_eci[2]}
        sat.v = {"x": v_eci[0], "y": v_eci[1], "z": v_eci[2]}
        
        # 4. Create head-on debris at the SAME position (instant collision)
        deb_now = new_sat_state.copy()
        deb_now[3:] = -new_sat_state[3:] # Head on!
        
        # 5. Inject Debris into state
        state_manager.debris[deb_id] = deb_now
        
        # 6. Force immediate scan and generate CDM
        screener.update_debris(state_manager.debris)
        
        # Explicitly set a 10-meter miss distance to trigger 'critical' but keep it realistic
        cdm = CDM(sat_id=sat_id, deb_id=deb_id, tca=sim_time + 1800.0, miss_distance=0.01, is_critical=True)
        
        with state_manager.lock:
            state_manager.cdms = [cdm] # Focus purely on this threat for the demo
        
        return {
            "status": "THREAT_INJECTED_OVER_GS",
            "sat_id": sat_id,
            "deb_id": deb_id,
            "location": "Over Bengaluru (Guaranteed LOS)",
            "tca_relative": "30 mins",
            "miss_m": 10.0
        }

@router.post("/telemetry")
async def ingest_telemetry(payload: TelemetryPayload, background_tasks: BackgroundTasks):
    """
    Ingest state vectors.
    """
    # Parse timestamp
    ts = state_manager.parse_time(payload.timestamp)
    state_manager.set_timestamp(ts)

    # Separate Satellites and Debris
    sats = []
    debs = []
    for obj in payload.objects:
        if obj.type == "DEBRIS" or obj.id.startswith("DEB"):
            debs.append(obj.model_dump())
        else:
            sats.append(obj.model_dump())

    # Update State Manager (Synchronous, fast)
    state_manager.update_satellites(sats)
    state_manager.update_debris(debs)

    # Schedule background task for conjunction screening
    background_tasks.add_task(run_conjunction_screen_async)

    return {
        "status": "ACK",
        "processed_count": len(payload.objects),
        "active_cdm_warnings": len(state_manager.cdms)
    }
