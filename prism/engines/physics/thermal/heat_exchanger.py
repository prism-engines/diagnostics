"""
Heat Exchanger Analysis

LMTD, NTU-effectiveness, shell-and-tube, compact HX.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute(T_h_in: float, T_h_out: float, T_c_in: float, T_c_out: float = None,
            m_h: float = 1.0, m_c: float = 1.0, cp_h: float = 4186, cp_c: float = 4186,
            U: float = None, A: float = None, 
            flow: str = 'counterflow') -> Dict[str, Any]:
    """
    Heat exchanger analysis using LMTD or NTU-effectiveness.
    
    Args:
        T_h_in, T_h_out: Hot stream inlet/outlet [K]
        T_c_in, T_c_out: Cold stream inlet/outlet [K]
        m_h, m_c: Mass flow rates [kg/s]
        cp_h, cp_c: Specific heats [J/kgK]
        U: Overall HTC [W/m²K]
        A: Heat transfer area [m²]
        flow: 'counterflow', 'parallel', 'crossflow'
    
    Returns:
        Q: Heat duty [W]
        LMTD: Log mean temperature difference
        effectiveness: ε
        NTU: Number of transfer units
    """
    C_h = m_h * cp_h
    C_c = m_c * cp_c
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max
    
    # Heat duty
    Q = C_h * (T_h_in - T_h_out)
    
    # LMTD
    if flow == 'counterflow':
        dT1 = T_h_in - T_c_out if T_c_out else T_h_in - T_c_in - Q/C_c
        dT2 = T_h_out - T_c_in
    else:  # parallel
        dT1 = T_h_in - T_c_in
        dT2 = T_h_out - (T_c_out if T_c_out else T_c_in + Q/C_c)
    
    if abs(dT1 - dT2) < 0.01:
        LMTD = dT1
    else:
        LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
    
    # Effectiveness
    Q_max = C_min * (T_h_in - T_c_in)
    effectiveness = Q / Q_max if Q_max > 0 else 0
    
    # NTU
    if U is not None and A is not None:
        NTU = U * A / C_min
    else:
        # Invert effectiveness-NTU relation
        if flow == 'counterflow' and C_r < 1:
            NTU = np.log((1 - effectiveness*C_r)/(1 - effectiveness)) / (1 - C_r)
        else:
            NTU = -np.log(1 - effectiveness * (1 + C_r)) / (1 + C_r)
    
    return {
        'Q': float(Q),
        'LMTD': float(LMTD),
        'effectiveness': float(effectiveness),
        'NTU': float(NTU),
        'C_r': float(C_r),
    }
