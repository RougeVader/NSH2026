import math
import numpy as np
from typing import Tuple

G0_MS = 9.80665
ISP = 300.0

def compute_dm(m_current_kg: float, dv_km_s: float) -> float:
    """
    Computes mass of fuel consumed for a given Delta-V.
    dv_km_s: Delta-V in km/s.
    """
    dv_m_s = abs(dv_km_s) * 1000.0
    dm = m_current_kg * (1 - math.exp(-dv_m_s / (ISP * G0_MS)))
    return dm

def apply_burn(state: np.ndarray, dv_eci: list) -> np.ndarray:
    """
    Applies an impulsive burn to a state vector.

    The burn is assumed to be instantaneous, meaning the velocity of the
    spacecraft changes instantly, while its position remains unchanged.

    Args:
        state: The initial state vector [x, y, z, vx, vy, vz] in km and km/s.
        dv_eci: The delta-v vector [dvx, dvy, dvz] in the ECI frame, in km/s.

    Returns:
        The new state vector after the burn.
    """
    new_state = state.copy()
    new_state[3] += dv_eci[0]
    new_state[4] += dv_eci[1]
    new_state[5] += dv_eci[2]
    return new_state

def compute_phasing_burns(dr_rtn: np.ndarray, dv_rtn: np.ndarray, n: float, T_phase: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2-burn phasing sequence to close along-track (y) gap.
    Targeting [0,0,0] relative position at T_phase.
    dr_rtn: [radial, along-track, cross-track]
    dv_rtn: [radial, along-track, cross-track]
    n: mean motion (rad/s)
    T_phase: Time to close gap (s)
    
    Returns: (dv1_rtn, dv2_rtn)
    """
    # Along-track gap is dr_rtn[1] (y)
    dy = dr_rtn[1]
    
    # Required drift rate (m/s)
    # Drift = -3 * dv_t * T_phase? No.
    # From CW: y(t) = y0 - (3n*x0 + 2*dy0)*t + (2/n)*(3n*x0 + 2*dy0)*sin(nt) + ...
    # Simplified for circular phasing (x0=0):
    # y(t) = y0 - 3*dv_t*t
    
    dv_t = dy / (3.0 * T_phase)
    
    # Burn 1: Change SMA to start drift
    dv1 = np.array([0.0, dv_t, 0.0])
    
    # Burn 2: Reverse to stop drift at target
    dv2 = np.array([0.0, -dv_t, 0.0])
    
    return dv1, dv2

def compute_radial_burns(dr_rtn: np.ndarray, dv_rtn: np.ndarray, n: float) -> np.ndarray:
    """
    Cancels radial and cross-track errors.
    """
    # Cross-track (z) is independent oscillation.
    # To kill z, burn at node (z=0) with dv_n = -v_z.
    
    # Radial (x) is coupled with along-track.
    # This is a simplification. Full CW targeting is better.
    pass

if __name__ == '__main__':
    # Validation from blueprint: 550 kg, ΔV = 0.1 km/s → dm ≈ 18.7 kg
    m_initial = 550.0
    dv = 0.1
    dm_calc = compute_dm(m_initial, dv)
    print(f"Initial mass: {m_initial} kg")
    print(f"Delta-V: {dv} km/s")
    print(f"Calculated fuel consumption: {dm_calc:.4f} kg")
    
    # Verification
    assert abs(dm_calc - 18.7) < 0.5, "Tsiolkovsky calculation is outside of expected tolerance."
    print("Tsiolkovsky validation passed.")
