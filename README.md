# ACM: Autonomous Constellation Manager (Platinum Tier) - NSH 2026

## Project Overview
**ACM (Autonomous Constellation Manager)** is an elite-tier, mission-critical flight software package designed for the **National Space Hackathon 2026**. This system represents the pinnacle of autonomous orbital management, providing high-fidelity physics, resilient communication modeling, and strategic constellation-aware optimization.

Designed to handle the future of "Mega-Constellations," ACM moves beyond simple reactive scripts to build a holistic, "thinking" brain for satellite fleets.

---

## 🏆 Platinum Tier: Key Technical Upgrades
Unlike standard submissions, this project implements advanced aerospace engineering features that reflect real-world orbital operational constraints.

### 1. High-Fidelity Physics Engine (J2 + SRP)
*   **RK4 Integrator**: 4th Order Runge-Kutta numerical integration for state propagation.
*   **J2 Perturbation**: Full modeling of Earth’s oblateness (Nodal Regression and Apsidal Precession).
*   **Solar Radiation Pressure (SRP)**: Integrated J2000 Sun position modeling with a **Cylindrical Earth Shadow Model** to correctly disable solar flux during eclipse phases.
*   **Vectorization**: Optimized `rk4_step_batch` logic for handling 10,000+ objects with sub-second latency.

### 2. Resilient Communication (Refracted LOS)
*   **Atmospheric Refraction**: Implements **Bennett’s Refraction Formula** for signal bending. This expands Ground Station (GS) availability windows slightly below the 0° geometric horizon, modeling real-world radio propagation.
*   **Blackout Planning**: The planner proactively identifies LOS blackouts (e.g., over the Pacific) and "uploads" autonomous command sets while the satellite is still in range.

### 3. Global Fleet Safety & Optimization
*   **Constellation-Aware COLA**: Evasion maneuvers are checked against the **entire constellation** (500m Fleet Buffer) to ensure an evasion burn doesn't cause a collision with a partner satellite.
*   **Fuel-Aware Planning**: Uses the **Tsiolkovsky Rocket Equation** to budget maneuvers. If fuel is critical (<5%), the system automatically skips recovery burns to prioritize survival, preventing satellite "burn-out."
*   **RTN Frame Optimization**: Maneuver planning in the **Radial-Transverse-Normal (RTN)** frame for maximum fuel efficiency.

### 4. Machine Learning & Spatial Indexing
*   **O(log N) Screening**: Uses **KD-Trees** (Spatial Indexing) for instantaneous conjunction detection in dense debris clouds.
*   **XGBoost Risk Predictor**: A trained ML model pre-filters conjunctions based on relative velocity vectors and miss-distance history before refining the Time of Closest Approach (TCA).

---

## 🚀 Getting Started

### Prerequisites
*   Docker Desktop (Windows/Linux/Mac).
*   A modern web browser (Chrome/Edge/Firefox) for the CesiumJS dashboard.

### Running the System
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/RougeVader/NSH2026.git
    cd NSH2026
    ```
2.  **Build the Docker Container**:
    ```bash
    docker build -t acm-platinum .
    ```
3.  **Run the Deployment**:
    ```bash
    docker run -p 8000:8000 acm-platinum
    ```
4.  **Access the Orbital Insight Dashboard**:
    Open `http://localhost:8000`.

---

## 📡 API Specifications
The ACM exposes a robust RESTful API on port 8000 for integration with external mission control systems.

*   **POST /api/telemetry**: Ingest high-frequency state vector updates.
*   **POST /api/simulate/step**: Advance simulation time (Fast-Forward).
*   **POST /api/maneuver/auto-schedule**: Trigger the Autonomous Planner.
*   **GET /api/visualization/snapshot**: Retrieve the global geodetic situational map.

---

## 🛠️ Built With
*   **Core Physics**: NumPy, SciPy (RK4, KD-Tree).
*   **Intelligence**: XGBoost (Collision Prediction).
*   **Backend**: FastAPI (Python 3.12).
*   **Frontend**: CesiumJS (3D Orbital Visualization).

**National Space Hackathon 2026 - IIT Delhi**
*Submission by Project AETHER Team*
