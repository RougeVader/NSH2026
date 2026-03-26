# ACM: Autonomous Constellation Manager - NSH 2026

## Project Overview
The **Autonomous Constellation Manager (ACM)** is a high-fidelity flight software suite designed for the autonomous management of large satellite constellations in Low Earth Orbit (LEO). Developed from the ground up for the **National Space Hackathon 2026**, this system implements rigorous orbital mechanics, resilient communication modeling, and multi-objective optimization to ensure mission safety and constellation uptime.

---

## Technical Architecture
The system is built on a modular architecture that separates physical propagation, conjunction assessment, and autonomous maneuver planning.

### 1. High-Fidelity Physics & Propagation
The core engine utilizes a 4th Order Runge-Kutta (RK4) numerical integrator, incorporating the following perturbations for high-accuracy state estimation:
*   **J2 Zonal Harmonic Model**: Accounts for Earth’s oblateness, correctly modeling nodal regression and apsidal precession.
*   **Solar Radiation Pressure (SRP)**: Implements an analytical Sun position model (J2000) with a cylindrical Earth shadow model to ensure physical accuracy during eclipse phases.
*   **Vectorized Processing**: The propagator is fully vectorized to handle 10,000+ objects simultaneously with sub-second latency.

### 2. Refracted Communication Modeling
To ensure realistic ground-to-space links, the system calculates Line-of-Sight (LOS) windows using **Bennett’s Atmospheric Refraction Formula**. This accounts for the signal bending at low elevations, providing a high-fidelity model of operational communication constraints.

### 3. Autonomous Evasion & Safety
The planner implements a multi-tier safety protocol for Conjunction Analysis and Collision Avoidance (COLA):
*   **Constellation-Aware Safety**: Evasion maneuvers are dynamically checked against the entire constellation state to prevent secondary inter-satellite collisions.
*   **Fuel-Budget Awareness**: Utilizing the Tsiolkovsky Rocket Equation, the system budgets maneuvers based on real-time mass depletion. It prioritizes satellite survival (minimal evasion) during low-fuel states.
*   **Optimal Phasing**: Evasion maneuvers are paired with 2-burn recovery sequences in the Radial-Transverse-Normal (RTN) frame to return satellites to their nominal slots with minimal propellant expenditure.

### 4. Scalable Conjunction Screening
*   **Spatial Indexing**: Uses KD-Trees for $O(\log N)$ spatial proximity detection in dense debris environments.
*   **Machine Learning Integration**: An XGBoost classifier pre-filters high-risk conjunctions based on relative state vectors before precise Time of Closest Approach (TCA) refinement.

---

## Deployment & Usage

### Prerequisites
*   Docker Engine / Docker Desktop.
*   A modern web browser for the CesiumJS-based dashboard.

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/RougeVader/NSH2026.git
    cd NSH2026
    ```
2.  **Build and Run**:
    ```bash
    docker build -t acm-system .
    docker run -p 8000:8000 acm-system
    ```
3.  **Access the Dashboard**:
    Open your browser and navigate to `http://localhost:8000`.

---

## API Reference
The system exposes a RESTful API for mission control integration:
*   `POST /api/telemetry`: State vector ingestion for fleet and debris.
*   `POST /api/simulate/step`: Advancement of simulation time.
*   `POST /api/maneuver/auto-schedule`: Execution of the autonomous COLA engine.
*   `GET /api/visualization/snapshot`: Real-time geodetic map data.

---

**National Space Hackathon 2026 - IIT Delhi**  
*Project AETHER Development Team*
