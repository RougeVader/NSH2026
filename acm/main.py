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

app = FastAPI(title="Autonomous Constellation Manager")

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
