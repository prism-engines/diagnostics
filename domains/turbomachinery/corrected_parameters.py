"""
Corrected Parameters — Standard Day Normalization

Gas turbine performance varies with ambient conditions. Corrected parameters
normalize measurements to ISA sea-level conditions, enabling:
- Comparison of data across different days/locations
- Detection of true degradation vs ambient effects
- Matching test data to design specifications

Theory:
    Corrected flow:
        W_corr = W × √(T/T_ref) / (P/P_ref) = W × √θ / δ
        
    Corrected speed:
        N_corr = N / √(T/T_ref) = N / √θ
        
    Corrected fuel flow:
        Wf_corr = Wf × √θ / δ
        
    Where:
        θ = T/T_ref (temperature ratio)
        δ = P/P_ref (pressure ratio)
        T_ref = 288.15 K (ISA sea level, 15°C)
        P_ref = 101325 Pa (ISA sea level, 1 atm)

Why this works:
    - Mass flow ∝ ρ × V × A ∝ (P/T) × √T = P/√T
    - So W × √T / P = constant for similar operating point
    - Allows direct comparison regardless of ambient conditions

Usage:
    On a hot day (T=310 K), an engine might show:
        N = 9500 rpm (actual)
        N_corr = 9500 / √(310/288.15) = 9156 rpm (corrected)
    
    This corrected speed can be compared to the design spec.

References:
    - Walsh & Fletcher, "Gas Turbine Performance", Ch. 2
    - SAE AIR1703 "Engine Performance Data"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math


# ISA Sea Level Reference Conditions
T_REF = 288.15  # K (15°C)
P_REF = 101325  # Pa (1 atm)


@dataclass
class CorrectedParametersResult:
    """Corrected parameter calculation results"""
    # Input conditions
    theta: float          # T/T_ref (temperature ratio)
    delta: float          # P/P_ref (pressure ratio)
    
    # Corrected values
    corrected_speed: Optional[float]      # rpm
    corrected_flow: Optional[float]       # kg/s
    corrected_fuel_flow: Optional[float]  # kg/s
    corrected_thrust: Optional[float]     # N
    
    # Original values (for reference)
    actual_speed: Optional[float]
    actual_flow: Optional[float]
    actual_fuel_flow: Optional[float]
    actual_thrust: Optional[float]
    
    # Altitude effect estimate
    equivalent_altitude_m: float  # Approximate altitude for these conditions
    
    confidence: float
    warnings: List[str]


def compute(
    T_ambient_K: float,
    P_ambient_Pa: float,
    speed_rpm: Optional[float] = None,
    mass_flow_kgs: Optional[float] = None,
    fuel_flow_kgs: Optional[float] = None,
    thrust_N: Optional[float] = None,
    T_ref: float = T_REF,
    P_ref: float = P_REF,
    **kwargs
) -> CorrectedParametersResult:
    """
    Compute corrected (normalized) parameters.
    
    Args:
        T_ambient_K: Ambient temperature at engine inlet (K)
        P_ambient_Pa: Ambient pressure at engine inlet (Pa)
        speed_rpm: Rotational speed (rpm), optional
        mass_flow_kgs: Mass flow rate (kg/s), optional
        fuel_flow_kgs: Fuel flow rate (kg/s), optional
        thrust_N: Net thrust (N), optional
        T_ref: Reference temperature (default ISA)
        P_ref: Reference pressure (default ISA)
        
    Returns:
        CorrectedParametersResult
    """
    warnings = []
    confidence = 1.0
    
    # Validate inputs
    if T_ambient_K <= 0:
        warnings.append(f"Invalid temperature: {T_ambient_K} K")
        confidence = 0.0
    if P_ambient_Pa <= 0:
        warnings.append(f"Invalid pressure: {P_ambient_Pa} Pa")
        confidence = 0.0
    
    # Calculate theta and delta
    theta = T_ambient_K / T_ref
    delta = P_ambient_Pa / P_ref
    
    sqrt_theta = math.sqrt(theta) if theta > 0 else 0
    
    # Corrected speed: N_corr = N / √θ
    N_corr = None
    if speed_rpm is not None:
        if sqrt_theta > 0:
            N_corr = speed_rpm / sqrt_theta
        else:
            warnings.append("Cannot correct speed: invalid theta")
    
    # Corrected flow: W_corr = W × √θ / δ
    W_corr = None
    if mass_flow_kgs is not None:
        if delta > 0:
            W_corr = mass_flow_kgs * sqrt_theta / delta
        else:
            warnings.append("Cannot correct flow: invalid delta")
    
    # Corrected fuel flow: Wf_corr = Wf × √θ / δ
    Wf_corr = None
    if fuel_flow_kgs is not None:
        if delta > 0:
            Wf_corr = fuel_flow_kgs * sqrt_theta / delta
        else:
            warnings.append("Cannot correct fuel flow: invalid delta")
    
    # Corrected thrust: F_corr = F / δ
    F_corr = None
    if thrust_N is not None:
        if delta > 0:
            F_corr = thrust_N / delta
        else:
            warnings.append("Cannot correct thrust: invalid delta")
    
    # Estimate equivalent altitude
    # Using approximate ISA: P = P0 × (1 - 0.0065×h/T0)^5.255
    # Simplified: h ≈ (1 - (P/P0)^0.19) × T0 / 0.0065
    if delta > 0 and delta < 2:
        try:
            altitude = (1 - delta ** 0.19) * T_REF / 0.0065
            equivalent_altitude = max(0, altitude)
        except:
            equivalent_altitude = 0
    else:
        equivalent_altitude = 0
    
    # Sanity checks
    if theta < 0.8:
        warnings.append(f"Very cold conditions (θ={theta:.3f}) - verify temperature")
    elif theta > 1.3:
        warnings.append(f"Very hot conditions (θ={theta:.3f}) - verify temperature")
    
    if delta < 0.3:
        warnings.append(f"Very low pressure (δ={delta:.3f}) - high altitude?")
    elif delta > 1.1:
        warnings.append(f"Pressure above sea level (δ={delta:.3f}) - pressurized test cell?")
    
    return CorrectedParametersResult(
        theta=theta,
        delta=delta,
        corrected_speed=N_corr,
        corrected_flow=W_corr,
        corrected_fuel_flow=Wf_corr,
        corrected_thrust=F_corr,
        actual_speed=speed_rpm,
        actual_flow=mass_flow_kgs,
        actual_fuel_flow=fuel_flow_kgs,
        actual_thrust=thrust_N,
        equivalent_altitude_m=equivalent_altitude,
        confidence=confidence,
        warnings=warnings,
    )


def correct_speed(speed_rpm: float, T_ambient_K: float, T_ref: float = T_REF) -> float:
    """Simple speed correction"""
    return speed_rpm / math.sqrt(T_ambient_K / T_ref)


def correct_flow(flow_kgs: float, T_ambient_K: float, P_ambient_Pa: float,
                 T_ref: float = T_REF, P_ref: float = P_REF) -> float:
    """Simple flow correction"""
    theta = T_ambient_K / T_ref
    delta = P_ambient_Pa / P_ref
    return flow_kgs * math.sqrt(theta) / delta


def uncorrect_speed(N_corr: float, T_ambient_K: float, T_ref: float = T_REF) -> float:
    """Convert corrected speed back to actual"""
    return N_corr * math.sqrt(T_ambient_K / T_ref)


def uncorrect_flow(W_corr: float, T_ambient_K: float, P_ambient_Pa: float,
                   T_ref: float = T_REF, P_ref: float = P_REF) -> float:
    """Convert corrected flow back to actual"""
    theta = T_ambient_K / T_ref
    delta = P_ambient_Pa / P_ref
    return W_corr * delta / math.sqrt(theta)


ENGINE_META = {
    'name': 'corrected_parameters',
    'capability': 'CORRECTED_PARAMETERS',
    'description': 'Normalize parameters to standard day conditions',
    'requires_signals': ['T_ambient', 'P_ambient'],
    'optional_signals': ['speed', 'mass_flow', 'fuel_flow', 'thrust'],
    'output_unit': 'various',
}


if __name__ == "__main__":
    print("=" * 60)
    print("Corrected Parameters — Self Test")
    print("=" * 60)
    
    # Test: Same engine, different days
    print("\n--- Same engine on different days ---")
    
    # Design point (ISA)
    N_design = 10000  # rpm
    W_design = 50     # kg/s
    
    test_cases = [
        ("ISA (15°C, sea level)", 288.15, 101325),
        ("Hot day (35°C)", 308.15, 101325),
        ("Cold day (-5°C)", 268.15, 101325),
        ("Denver (1600m)", 288.15, 83000),
        ("Hot & high", 308.15, 75000),
    ]
    
    print(f"\n  Design point: N={N_design} rpm, W={W_design} kg/s (at ISA)\n")
    
    for name, T, P in test_cases:
        result = compute(
            T_ambient_K=T,
            P_ambient_Pa=P,
            speed_rpm=N_design,
            mass_flow_kgs=W_design,
        )
        
        print(f"  {name}:")
        print(f"    θ={result.theta:.3f}, δ={result.delta:.3f}")
        print(f"    Actual:    N={N_design} rpm, W={W_design:.1f} kg/s")
        print(f"    Corrected: N={result.corrected_speed:.0f} rpm, W={result.corrected_flow:.1f} kg/s")
        print(f"    Equiv alt: {result.equivalent_altitude_m:.0f} m")
        print()
    
    # Test: Degradation detection
    print("--- Detecting degradation vs ambient effects ---")
    
    # Healthy engine at ISA
    result_healthy = compute(288.15, 101325, speed_rpm=10000, mass_flow_kgs=50)
    
    # Same engine on hot day - appears to have lower flow
    result_hot = compute(308.15, 101325, speed_rpm=10000, mass_flow_kgs=47)
    
    # Degraded engine at ISA - actually lower flow
    result_degraded = compute(288.15, 101325, speed_rpm=10000, mass_flow_kgs=47)
    
    print(f"\n  Healthy (ISA): W_corr = {result_healthy.corrected_flow:.1f} kg/s")
    print(f"  Hot day:       W_corr = {result_hot.corrected_flow:.1f} kg/s")
    print(f"  Degraded:      W_corr = {result_degraded.corrected_flow:.1f} kg/s")
    print("\n  Hot day shows ~same corrected flow (ambient effect)")
    print("  Degraded shows lower corrected flow (real problem)")
    
    print("\n" + "=" * 60)
