import threading
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict # Updated import

from datetime import datetime
from acm.data.tle_parser import load_and_parse_debris_tles

class Maneuver(BaseModel):
    burn_id: str
    burn_time: float # Unix timestamp
    dv_eci: List[float] # [vx, vy, vz]
    type: str # EVASION, RECOVERY, EOL, GRAVEYARD
    status: str = "SCHEDULED" # SCHEDULED, EXECUTED, CANCELLED

class SatelliteState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Updated config

    id: str
    r: Dict[str, float]
    v: Dict[str, float]
    state_vector: Optional[np.ndarray] = None 
    nominal_slot: Optional[np.ndarray] = None 
    fuel_kg: float = 50.0
    status: str = "NOMINAL"
    last_burn_time: float = 0.0
    outage_seconds: float = 0.0
    maneuver_queue: List[Maneuver] = []

class DebrisState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Updated config

    id: str
    type: str
    r: Dict[str, float]
    v: Dict[str, float]
    state_vector: Optional[np.ndarray] = None

class CDM(BaseModel):
    sat_id: str
    deb_id: str
    tca: float
    miss_distance: float
    is_critical: bool 

class StateManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.satellites: Dict[str, SatelliteState] = {}
        self.debris: Dict[str, np.ndarray] = {} 
        self.cdms: List[CDM] = []
        self.logs: List[str] = []
        self.last_timestamp: float = 0.0
        self.last_scan_time: float = 0.0
        
    def add_log(self, msg: str):
        with self.lock:
            self.logs.append(f"[{datetime.fromtimestamp(self.last_timestamp).isoformat()}] {msg}")
            if len(self.logs) > 100: 
                self.logs.pop(0)
        
    def parse_time(self, iso_str: str) -> float:
        try:
            # Handle "2026-03-12T08:00:00.000Z"
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0

    def set_timestamp(self, ts_val: float):
        with self.lock:
            self.last_timestamp = ts_val

    def update_satellites(self, sats_data: List[dict]):
        with self.lock:
            for s in sats_data:
                sid = s['id']
                r = np.array([s['r']['x'], s['r']['y'], s['r']['z']])
                v = np.array([s['v']['x'], s['v']['y'], s['v']['z']])
                state_vec = np.concatenate([r, v])
                
                if sid not in self.satellites:
                    # Initialize
                    sat = SatelliteState(id=sid, r=s['r'], v=s['v'])
                    sat.state_vector = state_vec
                    sat.nominal_slot = state_vec.copy() 
                    self.satellites[sid] = sat
                else:
                    # Update state
                    self.satellites[sid].state_vector = state_vec
                    # Do not overwrite maneuver_queue or fuel!


    def update_debris(self, debris_data: List[dict]):
        with self.lock:
            for d in debris_data:
                did = d['id']
                r = np.array([d['r']['x'], d['r']['y'], d['r']['z']])
                v = np.array([d['v']['x'], d['v']['y'], d['v']['z']])
                self.debris[did] = np.concatenate([r, v])
    
    def load_debris_from_tles(self, directory: str, timestamp: float):
        with self.lock:
            new_debris = load_and_parse_debris_tles(directory, timestamp)
            self.debris.update(new_debris)
                
    def get_ground_stations(self):
        from acm.data.stations import GROUND_STATIONS
        return GROUND_STATIONS
                
    def get_debris_dict(self):
        with self.lock:
            return self.debris.copy()
            
    def get_satellites(self):
        with self.lock:
            return self.satellites.copy()

# Global Instance
state_manager = StateManager()
