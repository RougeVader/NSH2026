import numpy as np

# --- Physical Constants ---
MU = 398600.4418      # km^3/s^2 (Earth's Gravitational Parameter)
RE = 6378.137         # km (Earth's Equatorial Radius)
J2 = 1.08263e-3       # J2 Zonal Harmonic Coefficient
G0_MS = 9.80665       # m/s^2 (Standard Gravity)
ISP = 300.0           # s (Specific Impulse)

def j2_accel(r: np.ndarray) -> np.ndarray:
    """
    Computes J2 perturbation acceleration for a single state vector.
    Correctly implements the asymmetric Z-component coefficient (5z^2/r^2 - 3).
    """
    x, y, z = r
    r_norm = np.linalg.norm(r)
    
    # If the satellite is at the center of the Earth, J2 is undefined.
    # Return zero acceleration.
    if r_norm < 1e-6:
        return np.array([0.0, 0.0, 0.0])
        
    factor = (1.5 * J2 * MU * RE**2) / (r_norm**5)
    common = 5 * (z**2) / (r_norm**2)
    
    ax = factor * x * (common - 1)
    ay = factor * y * (common - 1)
    az = factor * z * (common - 3)  # CRITICAL: (5z^2/r^2 - 3) for Z-axis
    
    return np.array([ax, ay, az])

def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """
    Propagates a single state vector [x, y, z, vx, vy, vz] using RK4 integration.
    Includes J2 perturbation.
    """
    def deriv(s):
        r, v = s[:3], s[3:]
        r_norm = np.linalg.norm(r)
        a_2body = -(MU / r_norm**3) * r
        a_j2 = j2_accel(r)
        return np.concatenate([v, a_2body + a_j2])

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_step_batch(states: np.ndarray, dt: float) -> np.ndarray:
    """
    Vectorized RK4 propagator for N objects.
    states: (N, 6) numpy array.
    ~30x faster than looping.
    """
    def deriv(s):
        # s shape: (N, 6)
        r = s[:, :3]  # (N, 3)
        v = s[:, 3:]  # (N, 3)
        
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)  # (N, 1)
        
        # Two-body acceleration (Vectorized)
        a_2body = -(MU / r_norm**3) * r
        
        # J2 acceleration (Vectorized)
        # This implementation follows the formula from the blueprint, but applied
        # to a matrix of N state vectors for performance.
        r_sq = r_norm**2
        r_5 = r_norm**5
        
        factor = (1.5 * J2 * MU * RE**2) / r_5  # (N, 1)
        z = r[:, 2:3]  # (N, 1) to keep dimensions for broadcasting
        z_sq = z**2
        common = 5 * z_sq / r_sq  # (N, 1)
        
        # The J2 perturbation has a different formula for the Z component.
        # ax = factor * x * (common - 1)
        # ay = factor * y * (common - 1)
        # az = factor * z * (common - 3)
        # We achieve this in a vectorized way by creating a multiplier matrix.
        mult_z = common - 3
        mult_xy = common - 1
        
        # This creates an (N, 3) matrix where the first two columns are (common-1)
        # and the third column is (common-3).
        multipliers = np.hstack([mult_xy, mult_xy, mult_z])
        
        # Element-wise multiplication results in the correct J2 acceleration vector for each object.
        # Broadcasting rules apply the (N,1) factor to each column of the (N,3) result.
        a_j2 = factor * r * multipliers
        
        return np.concatenate([v, a_2body + a_j2], axis=1)

    k1 = deriv(states)
    k2 = deriv(states + 0.5 * dt * k1)
    k3 = deriv(states + 0.5 * dt * k2)
    k4 = deriv(states + dt * k3)

    return states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate_state(state: np.ndarray, dt: float, max_step: float = 60.0) -> np.ndarray:
    """
    Efficiently propagates a single state vector over dt.
    Uses large steps where possible.
    """
    if dt <= 0: 
        return state
    
    curr = state.copy()
    remaining = dt
    
    # Use max_step for bulk
    steps = int(remaining / max_step)
    for _ in range(steps):
        curr = rk4_step(curr, max_step)
        
    rem = remaining - steps * max_step
    if rem > 1e-6:
        curr = rk4_step(curr, rem)
        
    return curr
