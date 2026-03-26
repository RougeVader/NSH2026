import numpy as np

# --- Physical Constants ---
MU = 398600.4418      # km^3/s^2 (Earth's Gravitational Parameter)
RE = 6378.137         # km (Earth's Equatorial Radius)
J2 = 1.08263e-3       # J2 Zonal Harmonic Coefficient
G0_MS = 9.80665       # m/s^2 (Standard Gravity)
ISP = 300.0           # s (Specific Impulse)
SOLAR_P = 4.56e-6     # N/m^2 (Solar Radiation Pressure at 1 AU)
CR = 1.2              # Reflectivity Coefficient
AREA = 10.0           # m^2 (Effective cross-section)
MASS = 500.0          # kg (Initial Mass)

def get_sun_pos(t_unix: float) -> np.ndarray:
    """
    Approximates Sun position in ECI J2000 frame for a given Unix timestamp.
    Uses simplified analytical model of Earth's orbit.
    """
    # Seconds since J2000 epoch (2000-01-01 12:00:00 UTC)
    # Unix epoch is 1970-01-01. J2000 offset is 946728000.0
    t = (t_unix - 946728000.0) / 86400.0 # Days since J2000
    
    # Mean Longitude of the Sun (approx)
    L = np.radians(280.460 + 0.9856474 * t)
    # Mean Anomaly
    g = np.radians(357.528 + 0.9856003 * t)
    # Ecliptic Longitude
    lambda_ecl = L + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    # Obliquity of the Ecliptic
    epsilon = np.radians(23.439 - 0.0000004 * t)
    
    # Sun unit vector in ECI
    u_sun = np.array([
        np.cos(lambda_ecl),
        np.cos(epsilon) * np.sin(lambda_ecl),
        np.sin(epsilon) * np.sin(lambda_ecl)
    ])
    
    # 1 AU in km
    AU = 149597870.7
    return u_sun * AU

def srp_accel(r: np.ndarray, t_unix: float) -> np.ndarray:
    """
    Computes SRP acceleration vector in km/s^2.
    Includes simplified cylindrical shadowing logic.
    """
    r_sun_vec = get_sun_pos(t_unix)
    u_sun = r_sun_vec / np.linalg.norm(r_sun_vec)
    
    # Shadow Check (Cylindrical model)
    # 1. Is the satellite behind the Earth?
    if np.dot(r, u_sun) < 0:
        # 2. Is it within the Earth's radius "tube"?
        # Distance of sat from the Sun-Earth line
        perp_dist = np.linalg.norm(r - np.dot(r, u_sun) * u_sun)
        if perp_dist < RE:
            return np.array([0.0, 0.0, 0.0]) # In shadow
            
    # Acceleration in m/s^2: (P * Cr * A / m)
    # Use current mass approx (for now just using initial MASS constant)
    a_mag = (SOLAR_P * CR * AREA / MASS)
    
    # Convert to km/s^2
    return (a_mag / 1000.0) * u_sun

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

def rk4_step(state: np.ndarray, dt: float, t_unix: float) -> np.ndarray:
    """
    Propagates a single state vector [x, y, z, vx, vy, vz] using RK4 integration.
    Includes J2 and SRP perturbations.
    """
    def deriv(s, t):
        r, v = s[:3], s[3:]
        r_norm = np.linalg.norm(r)
        a_2body = -(MU / r_norm**3) * r
        a_j2 = j2_accel(r)
        a_srp = srp_accel(r, t)
        return np.concatenate([v, a_2body + a_j2 + a_srp])

    k1 = deriv(state, t_unix)
    k2 = deriv(state + 0.5 * dt * k1, t_unix + 0.5 * dt)
    k3 = deriv(state + 0.5 * dt * k2, t_unix + 0.5 * dt)
    k4 = deriv(state + dt * k3, t_unix + dt)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rk4_step_batch(states: np.ndarray, dt: float, t_unix: float) -> np.ndarray:
    """
    Vectorized RK4 propagator for N objects.
    states: (N, 6) numpy array.
    """
    def deriv(s, t):
        # s shape: (N, 6)
        r = s[:, :3]  # (N, 3)
        v = s[:, 3:]  # (N, 3)
        
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)  # (N, 1)
        a_2body = -(MU / r_norm**3) * r
        
        # J2 acceleration (Vectorized)
        r_sq = r_norm**2
        r_5 = r_norm**5
        factor = (1.5 * J2 * MU * RE**2) / r_5
        z = r[:, 2:3]
        z_sq = z**2
        common = 5 * z_sq / r_sq
        mult_z = common - 3
        mult_xy = common - 1
        multipliers = np.hstack([mult_xy, mult_xy, mult_z])
        a_j2 = factor * r * multipliers
        
        # SRP acceleration (Simplified Vectorized Shadowing)
        r_sun_vec = get_sun_pos(t)
        u_sun = r_sun_vec / np.linalg.norm(r_sun_vec)
        
        # Initial SRP for all
        a_srp_mag = (SOLAR_P * CR * AREA / MASS) / 1000.0
        a_srp = np.tile(a_srp_mag * u_sun, (len(s), 1))
        
        # Apply shadow to relevant objects
        # Satellites behind Earth relative to Sun
        dot_products = np.dot(r, u_sun)
        in_shadow_zone = dot_products < 0
        
        # Satellites within Earth's radius "tube"
        perp_components = r - np.outer(dot_products, u_sun)
        perp_dists_sq = np.sum(perp_components**2, axis=1)
        in_tube = perp_dists_sq < RE**2
        
        shadow_mask = in_shadow_zone & in_tube
        a_srp[shadow_mask] = 0.0
        
        return np.concatenate([v, a_2body + a_j2 + a_srp], axis=1)

    k1 = deriv(states, t_unix)
    k2 = deriv(states + 0.5 * dt * k1, t_unix + 0.5 * dt)
    k3 = deriv(states + 0.5 * dt * k2, t_unix + 0.5 * dt)
    k4 = deriv(states + dt * k3, t_unix + dt)

    return states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def propagate_state(state: np.ndarray, dt: float, t_start: float = 0.0, max_step: float = 60.0) -> np.ndarray:
    """
    Efficiently propagates a single state vector over dt starting at t_start.
    """
    if dt <= 0: 
        return state
    
    curr = state.copy()
    remaining = dt
    curr_t = t_start
    
    # Use max_step for bulk
    steps = int(remaining / max_step)
    for _ in range(steps):
        curr = rk4_step(curr, max_step, curr_t)
        curr_t += max_step
        
    rem = remaining - steps * max_step
    if rem > 1e-6:
        curr = rk4_step(curr, rem, curr_t)
        
    return curr
