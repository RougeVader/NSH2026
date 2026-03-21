import numpy as np
import math
from typing import Tuple

# Constants
RE = 6378.137         # km
GMST_J2000 = 280.46061837
EARTH_RATE = 360.98564724  # deg/day

def rtn_to_eci_matrix(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix from RTN to ECI frame.
    M_RTN_ECI = [R_hat | T_hat | N_hat]
    """
    r_norm = np.linalg.norm(r_eci)
    if r_norm < 1e-6:
        return np.eye(3)
        
    R_hat = r_eci / r_norm
    
    h = np.cross(r_eci, v_eci)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-6:
        return np.eye(3)
        
    N_hat = h / h_norm
    T_hat = np.cross(N_hat, R_hat)
    
    return np.column_stack([R_hat, T_hat, N_hat])

def eci_to_rtn_matrix(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix from ECI to RTN frame.
    This is the transpose of RTN->ECI.
    """
    return rtn_to_eci_matrix(r_eci, v_eci).T

def dv_rtn_to_eci(dv_rtn: np.ndarray, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """Rotates a Delta-V vector from RTN to ECI."""
    M = rtn_to_eci_matrix(r_eci, v_eci)
    return M @ dv_rtn

def eci_to_ecef_matrix(t_unix: float) -> np.ndarray:
    """
    Computes rotation matrix from ECI to ECEF based on GMST.
    t_unix: Seconds since 1970-01-01 00:00:00 UTC
    """
    # J2000 epoch: 2000-01-01 12:00:00 TT (approx 946728000 unix)
    # The formula uses days since J2000
    # J2000 UNIX timestamp is approx 946728000
    
    seconds_since_j2000 = t_unix - 946728000.0
    days_since_j2000 = seconds_since_j2000 / 86400.0
    
    theta_deg = (GMST_J2000 + EARTH_RATE * days_since_j2000) % 360.0
    theta_rad = math.radians(theta_deg)
    
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def eci_to_ecef(r_eci: np.ndarray, t_unix: float) -> np.ndarray:
    """Converts ECI position vector to ECEF."""
    M = eci_to_ecef_matrix(t_unix)
    return M @ r_eci

def eci_to_geodetic(r_eci: np.ndarray, t_unix: float) -> Tuple[float, float, float]:
    """
    Converts ECI vector to Geodetic (Lat, Lon, Alt).
    Wrapper around batch version for scalar input.
    """
    res = eci_to_geodetic_batch(r_eci.reshape(1, 3), t_unix)
    return tuple(res[0])

def eci_to_ecef_batch(r_eci_batch: np.ndarray, t_unix: float) -> np.ndarray:
    """
    Vectorized conversion of ECI position vectors to ECEF.
    r_eci_batch: (N, 3) numpy array.
    """
    M = eci_to_ecef_matrix(t_unix)
    return (M @ r_eci_batch.T).T

def eci_to_geodetic_batch(r_eci_batch: np.ndarray, t_unix: float) -> np.ndarray:
    """
    Vectorized conversion of ECI vectors to Geodetic (Lat, Lon, Alt).
    r_eci_batch: (N, 3) numpy array.
    Returns: (N, 3) array with [lat_deg, lon_deg, alt_km].
    """
    if r_eci_batch.size == 0:
        return np.empty((0, 3))
        
    r_ecef = eci_to_ecef_batch(r_eci_batch, t_unix)
    x = r_ecef[:, 0]
    y = r_ecef[:, 1]
    z = r_ecef[:, 2]
    
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    
    e2 = 0.00669437999014
    lat = np.degrees(np.arctan2(z, p * (1 - e2)))
    
    # 2 iterations is sufficient for sub-meter accuracy
    for _ in range(2):
        lat_rad = np.radians(lat)
        sin_lat = np.sin(lat_rad)
        N = RE / np.sqrt(1 - e2 * sin_lat**2)
        lat = np.degrees(np.arctan2(z + e2 * N * sin_lat, p))
        
    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = RE / np.sqrt(1 - e2 * sin_lat**2)
    alt = (p / cos_lat) - N
    
    return np.column_stack([lat, lon, alt])
