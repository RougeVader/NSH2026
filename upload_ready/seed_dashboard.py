import requests
import numpy as np
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
TELEMETRY_ENDPOINT = f"{BASE_URL}/api/telemetry"

def generate_sample_data():
    print("--- TeamAtomV2: Generating Sample Orbital Data ---")
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    objects = []
    
    # 1. Generate a Constellation (50 Satellites)
    # Using simple circular orbits at 500km altitude
    R_EARTH = 6378.137
    alt = 500.0
    r_mag = R_EARTH + alt
    v_mag = np.sqrt(398600.4418 / r_mag)
    
    for i in range(50):
        angle = (i / 50) * 2 * np.pi
        rx = r_mag * np.cos(angle)
        ry = r_mag * np.sin(angle)
        rz = 0.0
        
        # Velocity vector (perpendicular to position)
        vx = -v_mag * np.sin(angle)
        vy = v_mag * np.cos(angle)
        vz = 0.1 * np.random.randn() # Small inclination variation
        
        objects.append({
            "id": f"SAT-{i:03d}",
            "type": "SATELLITE",
            "r": {"x": float(rx), "y": float(ry), "z": float(rz)},
            "v": {"x": float(vx), "y": float(vy), "z": float(vz)}
        })

    # 2. Generate a Debris Cloud (500 Objects)
    # Randomized around LEO shells
    for i in range(500):
        # Random position vector
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        
        d_alt = 300 + np.random.rand() * 700 # 300km to 1000km
        d_r_mag = R_EARTH + d_alt
        
        pos = vec * d_r_mag
        
        # Random velocity (approx circular)
        d_v_mag = np.sqrt(398600.4418 / d_r_mag)
        v_vec = np.random.randn(3)
        v_vec = np.cross(vec, v_vec)
        v_vec /= np.linalg.norm(v_vec)
        vel = v_vec * d_v_mag
        
        objects.append({
            "id": f"DEB-{i:04d}",
            "type": "DEBRIS",
            "r": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "v": {"x": float(vel[0]), "y": float(vel[1]), "z": float(vel[2])}
        })

    payload = {
        "timestamp": timestamp,
        "objects": objects
    }
    
    print(f"Pushing {len(objects)} objects to ACM Engine...")
    try:
        response = requests.post(TELEMETRY_ENDPOINT, json=payload)
        if response.status_code == 200:
            print("Successfully seeded dashboard!")
            print(f"Server Response: {response.json()}")
        else:
            print(f"Failed to seed data. Status Code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure the ACM server is running at http://localhost:8000")

if __name__ == "__main__":
    generate_sample_data()
