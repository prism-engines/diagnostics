"""
Compressor Isentropic Efficiency

Computes the isentropic (adiabatic) efficiency of a compressor by comparing
the actual work input to the ideal (reversible) work.

Theory:
    For an ideal isentropic compression:
        T₂ₛ/T₁ = (P₂/P₁)^((γ-1)/γ)
    
    Isentropic efficiency:
        η_c = (T₂ₛ - T₁) / (T₂ - T₁) = (ideal temp rise) / (actual temp rise)
    
    η_c < 1 because real compression requires more work (higher T₂) than ideal.

Physical meaning:
    η_c = 0.85 means the compressor uses 15% more work than a perfect compressor
    would for the same pressure ratio.

Typical values:
    - Axial compressor: 0.85 - 0.92
    - Centrifugal compressor: 0.78 - 0.88
    - Degraded compressor: 0.70 - 0.80
    - η_c > 1.0: Impossible (check sensors)
    - η_c < 0: Impossible (check T_in/T_out assignment)

References:
    - Mattingly, "Elements of Gas Turbine Propulsion", Ch. 5
    - NASA SP-36, "Equations, Tables, and Charts for Compressible Flow"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math


@dataclass
class CompressorEfficiencyResult:
    """Result of compressor efficiency calculation"""
    efficiency: float               # Isentropic efficiency (0-1)
    pressure_ratio: float           # P_out / P_in
    temperature_ratio: float        # T_out / T_in
    T_ideal: float                  # Ideal outlet temp (K)
    T_actual: float                 # Actual outlet temp (K)
    work_ideal: Optional[float]     # Ideal specific work (J/kg) if cp provided
    work_actual: Optional[float]    # Actual specific work (J/kg) if cp provided
    confidence: float               # Confidence in result (0-1)
    warnings: List[str]             # Any issues detected


def compute(
    T_inlet_K: float,
    T_outlet_K: float,
    P_inlet_Pa: float,
    P_outlet_Pa: float,
    gamma: float = 1.4,
    cp: Optional[float] = None,  # J/(kg·K), optional for work calc
    **kwargs
) -> CompressorEfficiencyResult:
    """
    Compute compressor isentropic efficiency.
    
    Args:
        T_inlet_K: Inlet total temperature (Kelvin)
        T_outlet_K: Outlet total temperature (Kelvin)
        P_inlet_Pa: Inlet total pressure (Pascals)
        P_outlet_Pa: Outlet total pressure (Pascals)
        gamma: Specific heat ratio (default 1.4 for air)
        cp: Specific heat at constant pressure (J/kg·K), optional
        
    Returns:
        CompressorEfficiencyResult with efficiency and diagnostics
    """
    warnings = []
    confidence = 1.0
    
    # Input validation
    if T_inlet_K <= 0:
        warnings.append(f"Invalid inlet temperature: {T_inlet_K} K")
        confidence = 0.0
    if T_outlet_K <= 0:
        warnings.append(f"Invalid outlet temperature: {T_outlet_K} K")
        confidence = 0.0
    if P_inlet_Pa <= 0:
        warnings.append(f"Invalid inlet pressure: {P_inlet_Pa} Pa")
        confidence = 0.0
    if P_outlet_Pa <= 0:
        warnings.append(f"Invalid outlet pressure: {P_outlet_Pa} Pa")
        confidence = 0.0
    
    if confidence == 0.0:
        return CompressorEfficiencyResult(
            efficiency=float('nan'),
            pressure_ratio=float('nan'),
            temperature_ratio=float('nan'),
            T_ideal=float('nan'),
            T_actual=T_outlet_K,
            work_ideal=None,
            work_actual=None,
            confidence=0.0,
            warnings=warnings,
        )
    
    # Pressure ratio
    PR = P_outlet_Pa / P_inlet_Pa
    
    # Check for expansion instead of compression
    if PR < 1.0:
        warnings.append(f"Pressure ratio < 1 ({PR:.3f}) - this is expansion, not compression")
        warnings.append("Use turbine_efficiency for expansion processes")
        confidence *= 0.3
    
    # Temperature ratio
    TR = T_outlet_K / T_inlet_K
    
    # Check for cooling instead of heating
    if TR < 1.0:
        warnings.append(f"Temperature ratio < 1 ({TR:.3f}) - outlet cooler than inlet")
        warnings.append("Check sensor assignment or intercooling")
        confidence *= 0.5
    
    # Isentropic exponent
    isentropic_exp = (gamma - 1) / gamma
    
    # Ideal outlet temperature (isentropic)
    T_ideal = T_inlet_K * (PR ** isentropic_exp)
    
    # Temperature rises
    delta_T_ideal = T_ideal - T_inlet_K
    delta_T_actual = T_outlet_K - T_inlet_K
    
    # Isentropic efficiency
    if abs(delta_T_actual) < 1e-10:
        warnings.append("No temperature rise detected - check sensors")
        efficiency = float('nan')
        confidence = 0.0
    else:
        efficiency = delta_T_ideal / delta_T_actual
    
    # Sanity checks on efficiency
    if efficiency > 1.0:
        warnings.append(f"Efficiency > 100% ({efficiency:.1%}) - physically impossible")
        warnings.append("Check: sensor calibration, heat loss, wrong gamma")
        confidence *= 0.3
    elif efficiency > 0.95:
        warnings.append(f"Very high efficiency ({efficiency:.1%}) - verify sensors")
        confidence *= 0.8
    elif efficiency < 0:
        warnings.append(f"Negative efficiency ({efficiency:.1%}) - impossible")
        warnings.append("Check: T_inlet/T_outlet may be swapped")
        confidence *= 0.1
    elif efficiency < 0.5:
        warnings.append(f"Very low efficiency ({efficiency:.1%}) - severely degraded or wrong data")
        confidence *= 0.7
    
    # Calculate specific work if cp provided
    work_ideal = None
    work_actual = None
    if cp is not None:
        work_ideal = cp * delta_T_ideal    # J/kg
        work_actual = cp * delta_T_actual  # J/kg
    
    return CompressorEfficiencyResult(
        efficiency=efficiency,
        pressure_ratio=PR,
        temperature_ratio=TR,
        T_ideal=T_ideal,
        T_actual=T_outlet_K,
        work_ideal=work_ideal,
        work_actual=work_actual,
        confidence=confidence,
        warnings=warnings,
    )


def compute_from_signals(
    signals: Dict[str, Any],
    constants: Dict[str, Any] = None,
) -> CompressorEfficiencyResult:
    """
    Wrapper that accepts signal dict with Quantities.
    
    Expected signals:
        - T_inlet or T_in or T1 or T2 (depending on station)
        - T_outlet or T_out or T2 or T3
        - P_inlet or P_in or P1 or P2
        - P_outlet or P_out or P2 or P3
    """
    constants = constants or {}
    
    # Find temperature inlet
    T_in = None
    for key in ['T_inlet', 'T_in', 'T1', 'T2']:
        if key in signals:
            T_in = signals[key]
            break
    
    # Find temperature outlet
    T_out = None
    for key in ['T_outlet', 'T_out', 'T2', 'T3', 'T24', 'T30']:
        if key in signals and key != 'T2':  # Don't reuse T2 if it was inlet
            T_out = signals[key]
            break
    
    # Find pressure inlet
    P_in = None
    for key in ['P_inlet', 'P_in', 'P1', 'P2']:
        if key in signals:
            P_in = signals[key]
            break
    
    # Find pressure outlet
    P_out = None
    for key in ['P_outlet', 'P_out', 'P2', 'P3', 'P30']:
        if key in signals and key != 'P2':
            P_out = signals[key]
            break
    
    # Convert to SI if Quantities
    def to_si(val, target_unit):
        if hasattr(val, 'to'):
            return val.to(target_unit)
        return float(val)
    
    return compute(
        T_inlet_K=to_si(T_in, 'K'),
        T_outlet_K=to_si(T_out, 'K'),
        P_inlet_Pa=to_si(P_in, 'Pa'),
        P_outlet_Pa=to_si(P_out, 'Pa'),
        gamma=constants.get('gamma', 1.4),
        cp=constants.get('cp', None),
    )


# Engine metadata for discovery
ENGINE_META = {
    'name': 'compressor_efficiency',
    'capability': 'COMPRESSOR_EFFICIENCY',
    'description': 'Isentropic efficiency of compression process',
    'requires_signals': ['T_inlet', 'T_outlet', 'P_inlet', 'P_outlet'],
    'optional_signals': [],
    'requires_constants': [],
    'optional_constants': ['gamma', 'cp'],
    'output_unit': None,  # Dimensionless
    'output_range': (0, 1),
}


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Compressor Efficiency — Self Test")
    print("=" * 60)
    
    # Test 1: Typical compressor
    print("\n--- Test 1: Typical axial compressor ---")
    result = compute(
        T_inlet_K=288.15,      # ISA sea level
        T_outlet_K=600,        # After compression
        P_inlet_Pa=101325,     # 1 atm
        P_outlet_Pa=1013250,   # 10 atm (PR=10)
        gamma=1.4,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Pressure ratio: {result.pressure_ratio:.1f}")
    print(f"  T_ideal: {result.T_ideal:.1f} K")
    print(f"  T_actual: {result.T_actual:.1f} K")
    print(f"  Confidence: {result.confidence:.0%}")
    # Expected: T_ideal = 288.15 × 10^0.286 = 556.5 K
    # η = (556.5 - 288.15) / (600 - 288.15) = 268.4 / 311.85 = 0.861
    
    # Test 2: Perfect compressor (η = 1.0)
    print("\n--- Test 2: Perfect compressor ---")
    T_ideal = 288.15 * (10 ** 0.286)  # 556.5 K
    result = compute(
        T_inlet_K=288.15,
        T_outlet_K=T_ideal,  # Exactly isentropic
        P_inlet_Pa=101325,
        P_outlet_Pa=1013250,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Confidence: {result.confidence:.0%}")
    
    # Test 3: Degraded compressor
    print("\n--- Test 3: Degraded compressor ---")
    result = compute(
        T_inlet_K=288.15,
        T_outlet_K=680,  # Much higher than ideal
        P_inlet_Pa=101325,
        P_outlet_Pa=1013250,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Warnings: {result.warnings}")
    
    # Test 4: Bad data (impossible)
    print("\n--- Test 4: Impossible efficiency ---")
    result = compute(
        T_inlet_K=288.15,
        T_outlet_K=500,  # Lower than isentropic - impossible
        P_inlet_Pa=101325,
        P_outlet_Pa=1013250,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Confidence: {result.confidence:.0%}")
    
    # Test 5: C-MAPSS style data (Rankine)
    print("\n--- Test 5: C-MAPSS style (Rankine) ---")
    # Convert from Rankine to Kelvin
    T2_R = 518.67   # Fan inlet
    T30_R = 1126.4  # HPC outlet (typical)
    T2_K = T2_R * 5/9
    T30_K = T30_R * 5/9
    
    P2_psia = 14.7    # Sea level
    P30_psia = 367.5  # PR = 25
    P2_Pa = P2_psia * 6894.76
    P30_Pa = P30_psia * 6894.76
    
    result = compute(
        T_inlet_K=T2_K,
        T_outlet_K=T30_K,
        P_inlet_Pa=P2_Pa,
        P_outlet_Pa=P30_Pa,
    )
    print(f"  T_inlet: {T2_K:.1f} K ({T2_R:.1f} R)")
    print(f"  T_outlet: {T30_K:.1f} K ({T30_R:.1f} R)")
    print(f"  Pressure ratio: {result.pressure_ratio:.1f}")
    print(f"  Efficiency: {result.efficiency:.1%}")
    
    print("\n" + "=" * 60)
