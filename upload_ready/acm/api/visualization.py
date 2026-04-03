from fastapi import APIRouter
import numpy as np
from acm.state import state_manager
from acm.physics.frames import eci_to_geodetic_batch

router = APIRouter()

@router.get("/snapshot")
def get_snapshot():
    """
    Returns current state for visualization.
    Debris cloud is flattened for performance.
    """
    ts = state_manager.last_timestamp
    
    satellites_out = []
    with state_manager.lock:
        sats = list(state_manager.satellites.values())
        if sats:
            sat_states = np.array([s.state_vector[:3] for s in sats if s.state_vector is not None])
            
            if sat_states.size > 0:
                sat_geodetics = eci_to_geodetic_batch(sat_states, ts)
                
                # Zip back with original data
                valid_idx = 0
                for s in sats:
                    if s.state_vector is not None:
                        lat, lon, alt = sat_geodetics[valid_idx]
                        satellites_out.append({
                            "id": s.id,
                            "lat": float(lat),
                            "lon": float(lon),
                            "alt": float(alt),
                            "fuel_kg": s.fuel_kg,
                            "status": s.status,
                            "outage_seconds": s.outage_seconds
                        })
                        valid_idx += 1
            
        debris_cloud = []
        debris_dict = state_manager.debris
        if debris_dict:
            ids = list(debris_dict.keys())
            states = np.array([debris_dict[i][:3] for i in ids])
            
            # Vectorized conversion!
            geodetics = eci_to_geodetic_batch(states, ts)
            
            # Construct flattened tuples: [id, lat, lon, alt]
            # Filter out any NaN/Inf values that could crash the frontend
            debris_cloud = []
            for i in range(len(ids)):
                lat, lon, alt = geodetics[i]
                if np.all(np.isfinite([lat, lon, alt])):
                    debris_cloud.append((ids[i], float(lat), float(lon), float(alt)))
                
    return {
        "timestamp": ts,
        "satellites": satellites_out,
        "debris_cloud": debris_cloud,
        "ground_stations": state_manager.get_ground_stations()
    }
