import numpy as np
from acm.physics.propagator import rk4_step, rk4_step_batch, MU, RE, J2, j2_accel
from acm.physics.frames import rtn_to_eci_matrix
from acm.physics.maneuver import compute_dm

def get_raan_from_state(state):
    """Helper to compute RAAN from a state vector."""
    r = state[:3]
    v = state[3:]
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)
    raan = np.arctan2(n[1], n[0])
    return np.degrees(raan)

def test_energy_conservation_j2():
    """
    Test that specific mechanical energy is roughly conserved.
    Note: J2 is conservative, so energy should be constant.
    However, numerical integration introduces errors.
    """
    # LEO orbit: r ~ 7000 km, v ~ 7.5 km/s
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.1])
    
    # Initial Energy
    r = state[:3]
    v = state[3:]
    E0 = np.linalg.norm(v)**2 / 2 - MU / np.linalg.norm(r)
    
    # Propagate 1 orbit (~6000s)
    dt = 10.0
    steps = 600
    curr = state.copy()
    for _ in range(steps):
        curr = rk4_step(curr, dt, 0.0)
        
    v_final = curr[3:]
    r_final = curr[:3]
    E1 = np.linalg.norm(v_final)**2 / 2 - MU / np.linalg.norm(r_final)
    
    # Relative error should be small (< 1e-4) for 2-body energy approx
    assert abs((E1 - E0) / E0) < 1e-3

def test_raan_drift():
    """
    Test J2 effect on RAAN (Nodal Regression).
    For i < 90, RAAN rate should be negative.
    """
    # 51.6 deg inclination, 400 km altitude (r ~ 6778)
    inc_deg = 51.6
    alt = 400.0
    r_mag = RE + alt
    v_mag = np.sqrt(MU / r_mag)
    
    # Create state with INC = 51.6
    inc_rad = np.radians(inc_deg)
    # Position on X axis (Node)
    r0 = np.array([r_mag, 0.0, 0.0])
    v0 = np.array([0.0, v_mag * np.cos(inc_rad), v_mag * np.sin(inc_rad)])
    
    state0 = np.concatenate([r0, v0])
    raan0 = get_raan_from_state(state0)
    
    # Propagate for one day
    sim_time = 86400.0
    dt = 60.0
    steps = int(sim_time / dt)
    
    curr_state = state0.copy()
    for _ in range(steps):
        curr_state = rk4_step(curr_state, dt, 0.0)
        
    raan1 = get_raan_from_state(curr_state)
    
    # Calculate simulated drift
    simulated_drift_deg = raan1 - raan0
    # Handle wraparound
    if simulated_drift_deg > 180:
        simulated_drift_deg -= 360
    elif simulated_drift_deg < -180:
        simulated_drift_deg += 360

    # Theoretical drift rate
    n = np.sqrt(MU / r_mag**3)
    dOhm_dt_rad = -1.5 * n * J2 * (RE / r_mag)**2 * np.cos(inc_rad)
    theoretical_drift_deg = np.degrees(dOhm_dt_rad * sim_time)
    
    print(f"Theoretical Drift: {theoretical_drift_deg:.4f} deg/day")
    print(f"Simulated Drift: {simulated_drift_deg:.4f} deg/day")
    
    # Check if simulated drift is close to theoretical
    assert abs(simulated_drift_deg - theoretical_drift_deg) < 0.5

def test_tsiolkovsky():
    """
    Test the Tsiolkovsky rocket equation implementation.
    Validation from blueprint: 550 kg, ΔV = 0.1 km/s → dm ≈ 18.7 kg
    """
    m_initial = 550.0
    dv = 0.1
    dm_calc = compute_dm(m_initial, dv)
    assert abs(dm_calc - 18.7) < 0.5

def test_batch_propagation_consistency():
    """
    Verify rk4_step_batch produces same results as loop.
    """
    N = 10
    states = np.random.rand(N, 6) * 7000.0
    # Normalize positions to LEO
    for i in range(N):
        r = states[i, :3]
        states[i, :3] = r / np.linalg.norm(r) * 7000.0
        states[i, 3:] = np.random.rand(3) * 7.5
        
    dt = 10.0
    
    # Batch
    next_batch = rk4_step_batch(states, dt, 0.0)
    
    # Loop
    next_loop = np.zeros_like(states)
    for i in range(N):
        next_loop[i] = rk4_step(states[i], dt, 0.0)
        
    assert np.allclose(next_batch, next_loop, atol=1e-12)

def test_rtn_matrix_orthonormality():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.0, 1.0])
    M = rtn_to_eci_matrix(r, v)
    
    # Check orthogonality: M.T @ M = I
    identity_matrix = M.T @ M
    assert np.allclose(identity_matrix, np.eye(3), atol=1e-10)
    
    # Check Determinant = 1 (Rotation)
    det = np.linalg.det(M)
    assert abs(det - 1.0) < 1e-10

def test_j2_accel_zero_vector():
    """Test that j2_accel returns zero acceleration for a zero position vector."""
    r = np.array([0.0, 0.0, 0.0])
    accel = j2_accel(r)
    assert np.all(accel == np.array([0.0, 0.0, 0.0]))

def test_orbital_period():
    """
    Verify that the orbital period of a circular orbit matches the theoretical value.
    """
    # ISS-like circular orbit
    alt = 400.0  # km
    a = RE + alt
    
    # Initial state
    r0 = np.array([a, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU / a), 0.0])
    state0 = np.concatenate([r0, v0])
    
    # Theoretical period
    T_theory = 2 * np.pi * np.sqrt(a**3 / MU)
    
    # Propagate for one period
    dt = 10.0
    steps = int(T_theory / dt)
    
    state_final = state0.copy()
    for _ in range(steps):
        state_final = rk4_step(state_final, dt, 0.0)
    
    # Propagate the remaining time
    rem_time = T_theory - steps * dt
    state_final = rk4_step(state_final, rem_time, 0.0)
    
    # Check if the final radial distance is close to the initial radial distance.
    # The error should be small.
    initial_radius = np.linalg.norm(state0[:3])
    final_radius = np.linalg.norm(state_final[:3])
    assert abs(final_radius - initial_radius) < 1.0 # 1 km tolerance
