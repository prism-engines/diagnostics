"""
Turbine Isentropic Efficiency

Computes the isentropic (adiabatic) efficiency of a turbine by comparing
the actual work output to the ideal (reversible) work.

Theory:
    For an ideal isentropic expansion:
        T₄ₛ/T₃ = (P₄/P₃)^((γ-1)/γ)

    Isentropic efficiency:
        η_t = (T₃ - T₄) / (T₃ - T₄ₛ) = (actual temp drop) / (ideal temp drop)

    Note: This is INVERSE of compressor efficiency definition!
    η_t < 1 because real expansion extracts less work than ideal.

Physical meaning:
    η_t = 0.90 means the turbine extracts 90% of the work that a perfect
    turbine would from the same pressure drop.

Typical values:
    - Axial turbine: 0.88 - 0.93
    - Radial turbine: 0.80 - 0.88
    - Degraded turbine: 0.75 - 0.85
    - η_t > 1.0: Impossible (check sensors)
    - η_t < 0: Impossible (expansion should cool gas)

References:
    - Mattingly, "Elements of Gas Turbine Propulsion", Ch. 6
    - NASA SP-36, "Equations, Tables, and Charts for Compressible Flow"
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math


@dataclass
class TurbineEfficiencyResult:
    """Result of turbine efficiency calculation"""
    efficiency: float               # Isentropic efficiency (0-1)
    pressure_ratio: float           # P_in / P_out (expansion ratio)
    temperature_ratio: float        # T_in / T_out
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
    gamma: float = 1.33,  # Lower for hot combustion gases
    cp: Optional[float] = None,  # J/(kg·K), optional for work calc
    **kwargs
) -> TurbineEfficiencyResult:
    """
    Compute turbine isentropic efficiency.

    Args:
        T_inlet_K: Inlet total temperature (Kelvin) - hot gas from combustor
        T_outlet_K: Outlet total temperature (Kelvin)
        P_inlet_Pa: Inlet total pressure (Pascals)
        P_outlet_Pa: Outlet total pressure (Pascals)
        gamma: Specific heat ratio (default 1.33 for combustion gases)
        cp: Specific heat at constant pressure (J/kg·K), optional

    Returns:
        TurbineEfficiencyResult with efficiency and diagnostics
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
        return TurbineEfficiencyResult(
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

    # Expansion ratio (P_in / P_out for turbine)
    ER = P_inlet_Pa / P_outlet_Pa

    # Check for compression instead of expansion
    if ER < 1.0:
        warnings.append(f"Expansion ratio < 1 ({ER:.3f}) - this is compression, not expansion")
        warnings.append("Use compressor_efficiency for compression processes")
        confidence *= 0.3

    # Temperature ratio (should be > 1 for expansion)
    TR = T_inlet_K / T_outlet_K

    # Check for heating instead of cooling
    if TR < 1.0:
        warnings.append(f"Temperature ratio < 1 ({TR:.3f}) - outlet hotter than inlet")
        warnings.append("Check sensor assignment or reheat")
        confidence *= 0.5

    # Isentropic exponent
    isentropic_exp = (gamma - 1) / gamma

    # Ideal outlet temperature (isentropic expansion)
    # T4s/T3 = (P4/P3)^((γ-1)/γ)
    T_ideal = T_inlet_K * ((P_outlet_Pa / P_inlet_Pa) ** isentropic_exp)

    # Temperature drops
    delta_T_ideal = T_inlet_K - T_ideal    # Ideal temp drop
    delta_T_actual = T_inlet_K - T_outlet_K  # Actual temp drop

    # Isentropic efficiency (actual/ideal for turbine)
    if abs(delta_T_ideal) < 1e-10:
        warnings.append("No ideal temperature drop - check pressure ratio")
        efficiency = float('nan')
        confidence = 0.0
    else:
        efficiency = delta_T_actual / delta_T_ideal

    # Sanity checks on efficiency
    if efficiency > 1.0:
        warnings.append(f"Efficiency > 100% ({efficiency:.1%}) - physically impossible")
        warnings.append("Check: sensor calibration, heat addition, wrong gamma")
        confidence *= 0.3
    elif efficiency > 0.95:
        warnings.append(f"Very high efficiency ({efficiency:.1%}) - verify sensors")
        confidence *= 0.8
    elif efficiency < 0:
        warnings.append(f"Negative efficiency ({efficiency:.1%}) - impossible")
        warnings.append("Check: outlet hotter than inlet?")
        confidence *= 0.1
    elif efficiency < 0.7:
        warnings.append(f"Very low efficiency ({efficiency:.1%}) - severely degraded or wrong data")
        confidence *= 0.7

    # Calculate specific work if cp provided
    work_ideal = None
    work_actual = None
    if cp is not None:
        work_ideal = cp * delta_T_ideal    # J/kg
        work_actual = cp * delta_T_actual  # J/kg

    return TurbineEfficiencyResult(
        efficiency=efficiency,
        pressure_ratio=ER,
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
) -> TurbineEfficiencyResult:
    """
    Wrapper that accepts signal dict with Quantities.
    """
    constants = constants or {}

    # Find temperature inlet (turbine inlet is hot side)
    T_in = None
    for key in ['T_inlet', 'T_in', 'T3', 'T30', 'T_hot']:
        if key in signals:
            T_in = signals[key]
            break

    # Find temperature outlet
    T_out = None
    for key in ['T_outlet', 'T_out', 'T4', 'T50', 'T_exhaust']:
        if key in signals:
            T_out = signals[key]
            break

    # Find pressure inlet
    P_in = None
    for key in ['P_inlet', 'P_in', 'P3', 'P30']:
        if key in signals:
            P_in = signals[key]
            break

    # Find pressure outlet
    P_out = None
    for key in ['P_outlet', 'P_out', 'P4', 'P50', 'P_exhaust']:
        if key in signals:
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
        gamma=constants.get('gamma', 1.33),
        cp=constants.get('cp', None),
    )


# Engine metadata for discovery
ENGINE_META = {
    'name': 'turbine_efficiency',
    'capability': 'TURBINE_EFFICIENCY',
    'description': 'Isentropic efficiency of expansion process',
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
    print("Turbine Efficiency — Self Test")
    print("=" * 60)

    # Test 1: Typical turbine
    print("\n--- Test 1: Typical gas turbine ---")
    result = compute(
        T_inlet_K=1400,        # Turbine inlet (after combustor)
        T_outlet_K=900,        # Turbine outlet
        P_inlet_Pa=1013250,    # 10 atm
        P_outlet_Pa=101325,    # 1 atm (ER=10)
        gamma=1.33,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Expansion ratio: {result.pressure_ratio:.1f}")
    print(f"  T_ideal: {result.T_ideal:.1f} K")
    print(f"  T_actual: {result.T_actual:.1f} K")
    print(f"  Confidence: {result.confidence:.0%}")

    # Test 2: Perfect turbine (η = 1.0)
    print("\n--- Test 2: Perfect turbine ---")
    T_ideal = 1400 * (0.1 ** (0.33/1.33))  # ~780 K
    result = compute(
        T_inlet_K=1400,
        T_outlet_K=T_ideal,  # Exactly isentropic
        P_inlet_Pa=1013250,
        P_outlet_Pa=101325,
        gamma=1.33,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  T_ideal = T_actual = {result.T_ideal:.1f} K")

    # Test 3: Degraded turbine
    print("\n--- Test 3: Degraded turbine ---")
    result = compute(
        T_inlet_K=1400,
        T_outlet_K=950,  # Not as much cooling as ideal
        P_inlet_Pa=1013250,
        P_outlet_Pa=101325,
        gamma=1.33,
    )
    print(f"  Efficiency: {result.efficiency:.1%}")
    print(f"  Warnings: {result.warnings}")

    print("\n" + "=" * 60)
