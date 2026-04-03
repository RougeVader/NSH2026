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
    """Applies an impulsive burn to a state vector."""
    new_state = state.copy()
    new_state[3] += dv_eci[0]
    new_state[4] += dv_eci[1]
    new_state[5] += dv_eci[2]
    return new_state

def cw_stm(n: float, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Clohessy-Wiltshire State Transition Matrix components.
    Returns: (Phi_rr, Phi_rv, Phi_vr, Phi_vv) each 3x3.
    """
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    
    # Position from Position
    Phi_rr = np.array([
        [4 - 3*c, 0, 0],
        [6*(s - nt), 1, 0],
        [0, 0, c]
    ])
    
    # Position from Velocity
    Phi_rv = np.array([
        [s/n, (2/n)*(1 - c), 0],
        [(2/n)*(c - 1), (4*s - 3*nt)/n, 0],
        [0, 0, s/n]
    ])
    
    # Velocity from Position
    Phi_vr = np.array([
        [3*n*s, 0, 0],
        [6*n*(c - 1), 0, 0],
        [0, 0, -n*s]
    ])
    
    # Velocity from Velocity
    Phi_vv = np.array([
        [c, 2*s, 0],
        [-2*s, 4*c - 3, 0],
        [0, 0, c]
    ])
    
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv

def compute_phasing_burns(dr_rtn: np.ndarray, dv_rtn: np.ndarray, n: float, T_phase: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2-burn phasing sequence to close relative gap [dr, dv] at T_phase.
    Uses full Clohessy-Wiltshire (CW) targeting logic.
    Target: [0,0,0] relative position and [0,0,0] relative velocity.
    """
    # Find Required Initial Relative Velocity (v0+) to reach r(T)=0
    # r(T) = Phi_rr @ r0 + Phi_rv @ v0+ = 0
    # v0+ = -inv(Phi_rv) @ Phi_rr @ r0
    
    Phi_rr, Phi_rv, _, _ = cw_stm(n, T_phase)
    
    # Solve Phi_rv @ v_req = -Phi_rr @ dr_rtn
    try:
        v_req = np.linalg.solve(Phi_rv, -Phi_rr @ dr_rtn)
    except np.linalg.LinAlgError:
        # Fallback for very short T_phase or singularity
        return np.zeros(3), np.zeros(3)
        
    # Burn 1: Change velocity from current dv_rtn to v_req
    dv1 = v_req - dv_rtn
    
    # Predict Velocity at T_phase just before Burn 2
    # v(T-) = Phi_vr @ r0 + Phi_vv @ v0+
    _, _, Phi_vr, Phi_vv = cw_stm(n, T_phase)
    v_final_minus = Phi_vr @ dr_rtn + Phi_vv @ v_req
    
    # Burn 2: Change velocity from v_final_minus to target 0
    dv2 = -v_final_minus
    
    return dv1, dv2

def compute_dm_batch(m_current_kg: float, dv_mags: np.ndarray) -> np.ndarray:
    """Vectorized mass consumption."""
    dv_m_s = np.abs(dv_mags) * 1000.0
    return m_current_kg * (1 - np.exp(-dv_m_s / (ISP * G0_MS)))
