import pytest
from fastapi.testclient import TestClient
from acm.main import app
from acm.state import state_manager

client = TestClient(app)

def test_ingest_telemetry_success():
    """
    Test successful ingestion of telemetry data.
    """
    payload = {
        "timestamp": "2026-03-12T08:00:00.000Z",
        "objects": [
            {
                "id": "SAT-001",
                "type": "SATELLITE",
                "r": {"x": 7000.0, "y": 0.0, "z": 0.0},
                "v": {"x": 0.0, "y": 7.5, "z": 0.1}
            },
            {
                "id": "DEB-001",
                "type": "DEBRIS",
                "r": {"x": 8000.0, "y": 0.0, "z": 0.0},
                "v": {"x": 0.0, "y": 7.0, "z": 0.1}
            }
        ]
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ACK"
    assert data["processed_count"] == 2
    
    # Ingestion updates the state manager
    assert "SAT-001" in state_manager.satellites
    assert "DEB-001" in state_manager.debris

def test_simulate_step():
    """
    Test that the simulation advances correctly.
    """
    # Set initial time
    state_manager.last_timestamp = 1000.0
    
    payload = {"step_seconds": 60.0}
    response = client.post("/api/simulate/step", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "STEP_COMPLETE"
    assert state_manager.last_timestamp == 1060.0

def test_visualization_snapshot():
    """
    Test that the visualization snapshot returns data.
    """
    response = client.get("/api/visualization/snapshot")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "satellites" in data
    assert "debris_cloud" in data
    assert "ground_stations" in data

def test_health_check():
    """
    Test the health check endpoint.
    """
    # Try with leading slash
    response = client.get("/health")
    if response.status_code == 404:
        # Debugging: Print all routes
        print("\nAvailable routes:")
        for route in app.routes:
            print(f"  {route.path}")
            
    assert response.status_code == 200
    assert response.json()["status"] == "ACM Operational"
