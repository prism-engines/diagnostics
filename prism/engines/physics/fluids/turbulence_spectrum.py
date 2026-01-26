"""
Turbulence Energy Spectrum Analysis

Kolmogorov -5/3 law validation, energy cascade, dissipation rate.

E(k) = C_k * ε^(2/3) * k^(-5/3)

where:
    E(k) = energy spectral density
    k = wavenumber
    ε = dissipation rate
    C_k ≈ 1.5 (Kolmogorov constant)
"""

import numpy as np
from numpy.fft import fftn, fftfreq
from typing import Dict, Any, Optional


def compute(velocity_field: np.ndarray, dx: float = 1.0, 
            nu: float = 1e-6) -> Dict[str, Any]:
    """
    Compute turbulence energy spectrum from velocity field.
    
    Args:
        velocity_field: 3D velocity array (nx, ny, nz, 3)
        dx: Grid spacing [m]
        nu: Kinematic viscosity [m²/s]
    
    Returns:
        wavenumber: Array of wavenumbers
        spectrum: Energy spectrum E(k)
        dissipation_rate: ε
        kolmogorov_scale: η
        taylor_scale: λ
        integral_scale: L
    """
    if velocity_field.ndim == 4:
        # 3D velocity field
        u = velocity_field[..., 0]
        v = velocity_field[..., 1] 
        w = velocity_field[..., 2]
        tke = 0.5 * np.mean(u**2 + v**2 + w**2)
    else:
        # 1D velocity signal
        u = velocity_field
        tke = 0.5 * np.mean(u**2)
    
    # FFT for spectrum
    n = len(u.flatten())
    u_hat = np.fft.fftn(u)
    power = np.abs(u_hat)**2 / n**2
    
    # Radial averaging for E(k)
    freqs = np.fft.fftfreq(u.shape[0], d=dx)
    k_mag = 2 * np.pi * np.abs(freqs)
    
    # Bin the spectrum
    k_bins = np.linspace(0, k_mag.max(), 50)
    spectrum = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            spectrum[i] = np.mean(power.flatten()[mask.flatten()])
    
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    # Estimate dissipation rate from spectrum
    # ε ≈ 2ν ∫ k² E(k) dk
    dissipation_rate = 2 * nu * np.trapz(k_centers**2 * spectrum, k_centers)
    
    # Kolmogorov scale η = (ν³/ε)^(1/4)
    kolmogorov_scale = (nu**3 / max(dissipation_rate, 1e-20))**0.25
    
    # Taylor microscale λ = √(10 ν TKE / ε)
    taylor_scale = np.sqrt(10 * nu * tke / max(dissipation_rate, 1e-20))
    
    # Integral scale L = TKE^(3/2) / ε
    integral_scale = tke**1.5 / max(dissipation_rate, 1e-20)
    
    return {
        'wavenumber': k_centers,
        'spectrum': spectrum,
        'tke': float(tke),
        'dissipation_rate': float(dissipation_rate),
        'kolmogorov_scale': float(kolmogorov_scale),
        'taylor_scale': float(taylor_scale),
        'integral_scale': float(integral_scale),
    }
