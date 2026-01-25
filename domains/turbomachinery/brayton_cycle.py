"""
Brayton Cycle Analysis — Full Gas Turbine Thermodynamics

Analyzes the complete thermodynamic cycle of a gas turbine:
    1 → 2: Compression (compressor)
    2 → 3: Heat addition (combustor)
    3 → 4: Expansion (turbine)
    4 → 1: Heat rejection (exhaust)

This engine computes:
    - Thermal efficiency
    - Work output
    - Heat input/rejection
    - Back work ratio
    - Specific fuel consumption
    - Carnot efficiency (theoretical max)

Theory:
    Thermal efficiency:
        η_th = W_net / Q_in = (W_t - W_c) / Q_in
        
    For ideal cycle:
        η_th,ideal = 1 - 1/PR^((γ-1)/γ) = 1 - T₁/T₂
        
    Back work ratio:
        BWR = W_c / W_t (typically 40-60% for gas turbines!)

Physical meaning:
    Gas turbines have high back work ratio - the compressor consumes
    a large fraction of the turbine output. This is why gas turbine
    efficiency is very sensitive to compressor degradation.

Stations (typical):
    1: Compressor inlet (ambient)
    2: Compressor outlet / combustor inlet
    3: Combustor outlet / turbine inlet
    4: Turbine outlet (exhaust)

References:
    - Cengel & Boles, "Thermodynamics", Ch. 9
    - Mattingly, "Elements of Gas Turbine Propulsion"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math


@dataclass
class BraytonCycleResult:
    """Complete Brayton cycle analysis"""
    # Efficiencies
    thermal_efficiency: float        # η_th = W_net / Q_in
    carnot_efficiency: float         # η_carnot = 1 - T_cold/T_hot
    compressor_efficiency: float     # η_c
    turbine_efficiency: float        # η_t
    
    # Work terms (J/kg specific)
    work_compressor: float           # W_c (consumed)
    work_turbine: float              # W_t (produced)
    work_net: float                  # W_net = W_t - W_c
    
    # Heat terms (J/kg specific)
    heat_in: float                   # Q_in (combustor)
    heat_out: float                  # Q_out (exhaust)
    
    # Ratios
    back_work_ratio: float           # BWR = W_c / W_t
    pressure_ratio: float            # PR = P2/P1
    temperature_ratio_combustor: float  # T3/T2
    
    # Temperatures at each station (K)
    T1: float  # Compressor inlet
    T2: float  # Compressor outlet
    T3: float  # Turbine inlet
    T4: float  # Turbine outlet
    
    # Fuel consumption
    sfc: Optional[float]             # Specific fuel consumption (kg/J)
    fuel_air_ratio: Optional[float]  # f = m_fuel / m_air
    
    confidence: float
    warnings: List[str]


def compute(
    T1_K: float,           # Compressor inlet
    T2_K: float,           # Compressor outlet
    T3_K: float,           # Turbine inlet (after combustor)
    T4_K: float,           # Turbine outlet
    P1_Pa: float,          # Compressor inlet pressure
    P2_Pa: float,          # Compressor outlet pressure
    gamma_cold: float = 1.4,    # γ for compressor (air)
    gamma_hot: float = 1.33,    # γ for turbine (combustion gases)
    cp_cold: float = 1005,      # J/(kg·K) for air
    cp_hot: float = 1150,       # J/(kg·K) for combustion gases
    LHV: Optional[float] = 43e6,  # Lower heating value of fuel (J/kg)
    **kwargs
) -> BraytonCycleResult:
    """
    Analyze complete Brayton cycle.
    
    Args:
        T1_K: Compressor inlet temperature (K)
        T2_K: Compressor outlet temperature (K)
        T3_K: Turbine inlet temperature (K) - peak cycle temp
        T4_K: Turbine outlet temperature (K)
        P1_Pa: Compressor inlet pressure (Pa)
        P2_Pa: Compressor outlet pressure (Pa)
        gamma_cold: Specific heat ratio for cold section
        gamma_hot: Specific heat ratio for hot section
        cp_cold: Specific heat, cold section (J/kg·K)
        cp_hot: Specific heat, hot section (J/kg·K)
        LHV: Fuel lower heating value (J/kg)
        
    Returns:
        BraytonCycleResult with complete cycle analysis
    """
    warnings = []
    confidence = 1.0
    
    # Validate temperatures are in correct order
    if not (T1_K < T2_K < T3_K > T4_K):
        warnings.append("Temperature order unexpected - check station assignments")
        confidence *= 0.7
    
    if T3_K < T2_K:
        warnings.append("T3 < T2: No heat addition? Check combustor")
        confidence *= 0.5
    
    if T4_K < T1_K:
        warnings.append("Exhaust colder than ambient - physically unlikely")
        confidence *= 0.5
    
    # Pressure ratio
    PR = P2_Pa / P1_Pa
    
    if PR < 1:
        warnings.append("Pressure ratio < 1 - check P1/P2 assignment")
        confidence *= 0.3
    
    # =========================================================================
    # COMPRESSOR
    # =========================================================================
    
    # Work consumed by compressor (per kg of air)
    W_c = cp_cold * (T2_K - T1_K)  # J/kg
    
    # Ideal isentropic temp rise
    T2_ideal = T1_K * (PR ** ((gamma_cold - 1) / gamma_cold))
    
    # Compressor isentropic efficiency
    if abs(T2_K - T1_K) > 0.1:
        eta_c = (T2_ideal - T1_K) / (T2_K - T1_K)
    else:
        eta_c = float('nan')
        warnings.append("No compressor temp rise")
    
    # =========================================================================
    # COMBUSTOR
    # =========================================================================
    
    # Heat added (per kg of air, assuming small fuel mass)
    Q_in = cp_hot * (T3_K - T2_K)  # J/kg
    
    # Temperature ratio across combustor
    TR_combustor = T3_K / T2_K
    
    # Fuel-air ratio (approximate, energy balance)
    if LHV and Q_in > 0:
        # Q_in = f × LHV (approximately)
        f = Q_in / LHV
        if f > 0.05:
            warnings.append(f"High fuel-air ratio ({f:.3f}) - check T3 or LHV")
    else:
        f = None
    
    # =========================================================================
    # TURBINE
    # =========================================================================
    
    # Work produced by turbine (per kg of gas)
    W_t = cp_hot * (T3_K - T4_K)  # J/kg
    
    # For turbine efficiency, need P3 and P4
    # Assume P3 ≈ P2 and P4 ≈ P1 (pressure drops in combustor and exhaust)
    P3_Pa = P2_Pa * 0.95  # ~5% combustor pressure drop
    P4_Pa = P1_Pa * 1.05  # ~5% exhaust back pressure
    
    # Ideal isentropic expansion
    T4_ideal = T3_K * ((P4_Pa / P3_Pa) ** ((gamma_hot - 1) / gamma_hot))
    
    # Turbine isentropic efficiency
    if abs(T3_K - T4_ideal) > 0.1:
        eta_t = (T3_K - T4_K) / (T3_K - T4_ideal)
    else:
        eta_t = float('nan')
        warnings.append("Cannot compute turbine efficiency")
    
    # =========================================================================
    # CYCLE PERFORMANCE
    # =========================================================================
    
    # Net work output
    W_net = W_t - W_c  # J/kg
    
    if W_net < 0:
        warnings.append("Negative net work - compressor consumes more than turbine produces")
        confidence *= 0.5
    
    # Back work ratio
    if W_t > 0:
        BWR = W_c / W_t
    else:
        BWR = float('nan')
    
    # Thermal efficiency
    if Q_in > 0:
        eta_th = W_net / Q_in
    else:
        eta_th = float('nan')
    
    # Heat rejected (exhaust)
    # Energy balance: Q_in = W_net + Q_out
    Q_out = Q_in - W_net
    
    # Carnot efficiency (theoretical maximum)
    # Based on peak and minimum cycle temperatures
    T_hot = T3_K
    T_cold = T1_K
    eta_carnot = 1 - T_cold / T_hot
    
    # Specific fuel consumption
    if W_net > 0 and f is not None:
        sfc = f / W_net  # kg_fuel / J_work
    else:
        sfc = None
    
    # =========================================================================
    # SANITY CHECKS
    # =========================================================================
    
    if eta_th > eta_carnot:
        warnings.append(f"Thermal efficiency ({eta_th:.1%}) > Carnot ({eta_carnot:.1%}) - impossible")
        confidence *= 0.3
    
    if BWR > 0.8:
        warnings.append(f"Very high back work ratio ({BWR:.1%}) - low net output")
    
    if eta_c > 1 or eta_t > 1:
        warnings.append("Component efficiency > 100% - check data")
        confidence *= 0.5
    
    return BraytonCycleResult(
        thermal_efficiency=eta_th,
        carnot_efficiency=eta_carnot,
        compressor_efficiency=eta_c,
        turbine_efficiency=eta_t,
        work_compressor=W_c,
        work_turbine=W_t,
        work_net=W_net,
        heat_in=Q_in,
        heat_out=Q_out,
        back_work_ratio=BWR,
        pressure_ratio=PR,
        temperature_ratio_combustor=TR_combustor,
        T1=T1_K,
        T2=T2_K,
        T3=T3_K,
        T4=T4_K,
        sfc=sfc,
        fuel_air_ratio=f,
        confidence=confidence,
        warnings=warnings,
    )


def compute_ideal(
    T1_K: float,
    T3_K: float,  # Peak temperature
    PR: float,    # Pressure ratio
    gamma: float = 1.4,
    cp: float = 1005,
) -> BraytonCycleResult:
    """
    Compute ideal (100% efficiency) Brayton cycle for comparison.
    """
    # Ideal temperatures
    T2_K = T1_K * (PR ** ((gamma - 1) / gamma))
    T4_K = T3_K / (PR ** ((gamma - 1) / gamma))
    
    return compute(
        T1_K=T1_K,
        T2_K=T2_K,
        T3_K=T3_K,
        T4_K=T4_K,
        P1_Pa=101325,
        P2_Pa=101325 * PR,
        gamma_cold=gamma,
        gamma_hot=gamma,
        cp_cold=cp,
        cp_hot=cp,
    )


ENGINE_META = {
    'name': 'brayton_cycle',
    'capability': 'BRAYTON_CYCLE',
    'description': 'Complete gas turbine thermodynamic cycle analysis',
    'requires_signals': ['T1', 'T2', 'T3', 'T4', 'P1', 'P2'],
    'optional_signals': ['P3', 'P4', 'fuel_flow'],
    'optional_constants': ['gamma_cold', 'gamma_hot', 'cp_cold', 'cp_hot', 'LHV'],
    'output_unit': None,
}


if __name__ == "__main__":
    print("=" * 60)
    print("Brayton Cycle Analysis — Self Test")
    print("=" * 60)
    
    # Test 1: Ideal cycle
    print("\n--- Test 1: Ideal Brayton cycle (PR=10, T3=1400K) ---")
    result = compute_ideal(T1_K=288.15, T3_K=1400, PR=10)
    
    print(f"  Temperatures: T1={result.T1:.0f}K → T2={result.T2:.0f}K → T3={result.T3:.0f}K → T4={result.T4:.0f}K")
    print(f"  Thermal efficiency: {result.thermal_efficiency:.1%}")
    print(f"  Carnot efficiency:  {result.carnot_efficiency:.1%}")
    print(f"  Back work ratio:    {result.back_work_ratio:.1%}")
    print(f"  W_net: {result.work_net/1000:.1f} kJ/kg")
    
    # Test 2: Real cycle with inefficiencies
    print("\n--- Test 2: Real cycle (η_c=85%, η_t=90%) ---")
    
    T1 = 288.15
    PR = 10
    T3 = 1400
    gamma = 1.4
    
    # Ideal T2 and T4
    T2_ideal = T1 * (PR ** ((gamma-1)/gamma))  # 556.8 K
    T4_ideal = T3 / (PR ** ((gamma-1)/gamma))  # 725.3 K
    
    # Real T2 with 85% compressor efficiency
    eta_c = 0.85
    T2_real = T1 + (T2_ideal - T1) / eta_c  # Higher than ideal
    
    # Real T4 with 90% turbine efficiency
    eta_t = 0.90
    T4_real = T3 - eta_t * (T3 - T4_ideal)  # Higher than ideal
    
    result = compute(
        T1_K=T1,
        T2_K=T2_real,
        T3_K=T3,
        T4_K=T4_real,
        P1_Pa=101325,
        P2_Pa=101325 * PR,
    )
    
    print(f"  Temperatures: T1={result.T1:.0f}K → T2={result.T2:.0f}K → T3={result.T3:.0f}K → T4={result.T4:.0f}K")
    print(f"  Thermal efficiency: {result.thermal_efficiency:.1%}")
    print(f"  Carnot efficiency:  {result.carnot_efficiency:.1%}")
    print(f"  Component η_c: {result.compressor_efficiency:.1%}")
    print(f"  Component η_t: {result.turbine_efficiency:.1%}")
    print(f"  Back work ratio:    {result.back_work_ratio:.1%}")
    print(f"  W_net: {result.work_net/1000:.1f} kJ/kg")
    if result.fuel_air_ratio:
        print(f"  Fuel-air ratio: {result.fuel_air_ratio:.4f}")
    
    # Test 3: Degraded compressor impact
    print("\n--- Test 3: Impact of compressor degradation ---")
    
    for eta_c in [0.90, 0.85, 0.80, 0.75]:
        T2_real = T1 + (T2_ideal - T1) / eta_c
        T4_real = T3 - 0.90 * (T3 - T4_ideal)  # Turbine stays at 90%
        
        result = compute(T1_K=T1, T2_K=T2_real, T3_K=T3, T4_K=T4_real,
                        P1_Pa=101325, P2_Pa=101325*PR)
        
        print(f"  η_c={eta_c:.0%}: η_th={result.thermal_efficiency:.1%}, BWR={result.back_work_ratio:.1%}, W_net={result.work_net/1000:.0f} kJ/kg")
    
    print("\n  → Compressor degradation has outsized impact on cycle efficiency!")
    print("  → This is why hd_slope on compressor efficiency is so important.")
    
    print("\n" + "=" * 60)
