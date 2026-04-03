import logging
import time
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ACM")

from acm.api import telemetry, maneuver, simulate, visualization
from contextlib import asynccontextmanager
import numpy as np
from acm.state import state_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Auto-Seed Mechanism ---
    # Automatically populate the engine with data if it's empty
    if not state_manager.satellites:
        logger.info("Initializing TeamAtomV2 Default Orbital Scenario...")
        
        # Current time
        t_now = time.time()
        state_manager.set_timestamp(t_now)
        
        # 1. Generate 50 Satellites (Circular LEO)
        R_EARTH = 6378.137
        mu = 398600.4418
        alt = 500.0
        r_mag = R_EARTH + alt
        v_mag = np.sqrt(mu / r_mag)
        
        sats = []
        for i in range(50):
            angle = (i / 50) * 2 * np.pi
            rx = r_mag * np.cos(angle)
            ry = r_mag * np.sin(angle)
            sats.append({
                "id": f"FLEET-{i:02d}",
                "type": "SATELLITE",
                "r": {"x": float(rx), "y": float(ry), "z": 0.0},
                "v": {"x": float(-v_mag * np.sin(angle)), "y": float(v_mag * np.cos(angle)), "z": 0.0}
            })
        state_manager.update_satellites(sats)
        
        # 2. Generate 1000 Debris Objects
        debs = []
        for i in range(1000):
            vec = np.random.randn(3)
            v_norm = np.linalg.norm(vec)
            if v_norm < 1e-9: vec = np.array([1.0, 0.0, 0.0])
            else: vec /= v_norm
            
            d_r_mag = R_EARTH + 300 + np.random.rand() * 700
            pos = vec * d_r_mag
            v_vec = np.random.randn(3)
            v_vec = np.cross(vec, v_vec)
            vv_norm = np.linalg.norm(v_vec)
            if vv_norm < 1e-9: v_vec = np.array([0.0, 1.0, 0.0])
            else: v_vec /= vv_norm
            
            vel = v_vec * np.sqrt(mu / d_r_mag)
            debs.append({
                "id": f"DEB-{i:04d}",
                "type": "DEBRIS",
                "r": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "v": {"x": float(vel[0]), "y": float(vel[1]), "z": float(vel[2])}
            })
        state_manager.update_debris(debs)
        logger.info(f"TeamAtomV2: Auto-seeded {len(sats)} satellites and {len(debs)} debris objects.")

    yield

app = FastAPI(title="Autonomous Constellation Manager", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(telemetry.router, prefix="/api")
app.include_router(maneuver.router, prefix="/api/maneuver")
app.include_router(simulate.router, prefix="/api")
app.include_router(visualization.router, prefix="/api/visualization")

@app.get("/health")
def read_root():
    return {"status": "ACM Operational", "version": "1.0.0"}

# Mount Frontend
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
