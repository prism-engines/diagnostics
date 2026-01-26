"""
Equations of State

Cubic EOS: van der Waals, SRK, Peng-Robinson.
"""

import numpy as np
from typing import Dict, Any, Optional


R = 8.314  # J/mol/K


def compute(T: float, P: float, Tc: float, Pc: float, omega: float = 0.0,
            eos: str = 'peng_robinson') -> Dict[str, Any]:
    """
    Compute compressibility factor and molar volume from cubic EOS.
    
    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        Tc: Critical temperature [K]
        Pc: Critical pressure [Pa]
        omega: Acentric factor
        eos: 'van_der_waals', 'srk', 'peng_robinson'
    
    Returns:
        Z: Compressibility factor
        V: Molar volume [m³/mol]
        fugacity_coeff: φ
    """
    Tr = T / Tc
    Pr = P / Pc
    
    if eos == 'van_der_waals':
        a = 27/64 * (R*Tc)**2 / Pc
        b = R*Tc / (8*Pc)
        alpha = 1.0
        u, w = 0, 0
    elif eos == 'srk':
        a = 0.42748 * (R*Tc)**2 / Pc
        b = 0.08664 * R*Tc / Pc
        m = 0.48 + 1.574*omega - 0.176*omega**2
        alpha = (1 + m*(1 - np.sqrt(Tr)))**2
        u, w = 1, 0
    else:  # Peng-Robinson
        a = 0.45724 * (R*Tc)**2 / Pc
        b = 0.07780 * R*Tc / Pc
        m = 0.37464 + 1.54226*omega - 0.26992*omega**2
        alpha = (1 + m*(1 - np.sqrt(Tr)))**2
        u, w = 2, -1
    
    A = alpha * a * P / (R*T)**2
    B = b * P / (R*T)
    
    # Solve cubic: Z³ - (1-B)Z² + (A - uB - wB²)Z - AB + wB² + wB³ = 0
    # For PR: Z³ - (1-B)Z² + (A - 3B² - 2B)Z - (AB - B² - B³) = 0
    coeffs = [1, -(1 - B), A - u*B - (u+w)*B**2, -A*B + w*B**2 + w*B**3]
    roots = np.roots(coeffs)
    
    # Take real positive roots
    Z_vals = [z.real for z in roots if np.isreal(z) and z.real > B]
    
    if len(Z_vals) == 0:
        Z = max(roots, key=lambda x: x.real).real
    elif len(Z_vals) == 1:
        Z = Z_vals[0]
    else:
        Z = max(Z_vals)  # Vapor phase (largest)
    
    V = Z * R * T / P
    
    # Fugacity coefficient (PR)
    if eos == 'peng_robinson':
        sqrt2 = np.sqrt(2)
        ln_phi = Z - 1 - np.log(Z - B) - A/(2*sqrt2*B) * np.log((Z + (1+sqrt2)*B)/(Z + (1-sqrt2)*B))
        phi = np.exp(ln_phi)
    else:
        phi = 1.0  # Simplified
    
    return {
        'Z': float(Z),
        'V': float(V),
        'fugacity_coeff': float(phi),
        'A': float(A),
        'B': float(B),
    }
