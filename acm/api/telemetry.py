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
