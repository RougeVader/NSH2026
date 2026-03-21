# Autonomous Constellation Manager (ACM) - NSH 2026

## Project Overview
ACM is a high-fidelity, autonomous system designed for the **National Space Hackathon 2026**. It manages large satellite constellations in Low Earth Orbit (LEO), ensuring operational safety through advanced physics-based propagation, machine learning-powered conjunction assessment, and autonomous maneuver planning.

### Key Features
*   **High-Fidelity Physics Engine**: Implements an RK4 integrator with J2 perturbation modeling for accurate orbital propagation.
*   **Two-Stage Conjunction Assessment**:
    *   **Stage 1**: Fast coarse filtering using KD-Trees for rapid candidate identification.
    *   **Stage 2**: GPU-accelerated **XGBoost Risk Predictor** (trained on real ESA CDM data) for intelligent pre-filtering, followed by precise TCA refinement.
*   **Autonomous Maneuver Planner**:
    *   Plans robust 3-burn sequences (Evasion + 2-burn Recovery) to return satellites to their nominal slots.
    *   Autonomous command upload logic based on Ground Station Line-of-Sight (LOS) availability.
    *   Safety pre-check to ensure maneuvers do not introduce new collision risks.
*   **3D Visualization Dashboard**: Real-time orbital insight powered by CesiumJS, providing global situational awareness.
*   **Vectorized Simulation**: Optimized tick processing capable of handling 50+ satellites and 10,000+ debris objects with sub-second latency.

## Getting Started

### Prerequisites
*   Docker installed on your system.
*   (Optional) NVIDIA GPU with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) for GPU-accelerated ML inference inside Docker.

### Running with Docker
1.  **Build the Image**:
    ```bash
    docker build -t acm-system .
    ```
2.  **Run the Container**:
    ```bash
    docker run -p 8000:8000 acm-system
    ```
3.  **Access the Dashboard**:
    Open your browser and navigate to `http://localhost:8000`.

## API Documentation

### 1. Telemetry Ingestion
*   **Endpoint**: `POST /api/telemetry`
*   **Payload**: `{"timestamp": "ISO-8601", "objects": [...]}`
*   **Action**: Ingests state vectors for satellites and debris.

### 2. Simulation Control
*   **Endpoint**: `POST /api/simulate/step`
*   **Payload**: `{"step_seconds": 60.0}`
*   **Action**: Advances the global simulation state by the specified duration.

### 3. Maneuver Scheduling
*   **Endpoint**: `POST /api/maneuver/schedule`
*   **Action**: Manually schedule specific burn sequences for satellites.
*   **Auto-Planning**: `POST /api/maneuver/auto-schedule` triggers the autonomous COLA engine.

### 4. Visualization
*   **Endpoint**: `GET /api/visualization/snapshot`
*   **Action**: Returns geodetic coordinates and status for all objects.

## Technical Details
*   **Propagator**: 4th Order Runge-Kutta (RK4).
*   **Perturbations**: J2 Zonal Harmonic (Earth Oblateness).
*   **ML Model**: XGBoost Classifier trained on ESA's Kelvins Collision Avoidance Challenge dataset.
*   **Frontend**: CesiumJS for 3D globe rendering.
*   **Backend**: FastAPI (Python 3.11).

## Team
Developed for the National Space Hackathon 2026.
