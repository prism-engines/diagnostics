"""
Polytropic Efficiency — Small-Stage Efficiency

Polytropic efficiency represents the efficiency of infinitesimally small
compression/expansion stages. Unlike isentropic efficiency, it is independent
of pressure ratio, making it better for comparing machines of different sizes.

Theory:
    For polytropic process: PV^n = constant

    Polytropic efficiency (compressor):
        η_p = ((γ-1)/γ) × ln(P₂/P₁) / ln(T₂/T₁)

    Polytropic efficiency (turbine):
        η_p = ln(T₃/T₄) / ((γ-1)/γ) × ln(P₃/P₄)

    Relationship to isentropic efficiency:
        For compressor: η_c = (PR^((γ-1)/γ) - 1) / (PR^((γ-1)/(γ×η_p)) - 1)
        For turbine:    η_t = (1 - PR^(-(γ-1)×η_p/γ)) / (1 - PR^(-(γ-1)/γ))

Why polytropic is useful:
    - Isentropic efficiency DECREASES with pressure ratio (for same technology)
    - Polytropic efficiency stays CONSTANT
    - Better for comparing a 5:1 PR compressor to a 20:1 PR compressor
    - Better for stage-stacking analysis

Typical values:
    - Modern axial compressor: 0.90 - 0.93
    - Centrifugal compressor: 0.85 - 0.90
    - Axial turbine: 0.90 - 0.93
    - Radial turbine: 0.85 - 0.90

References:
    - Saravanamuttoo, "Gas Turbine Theory", Ch. 4
    - Walsh & Fletcher, "Gas Turbine Performance", Ch. 5
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math


@dataclass
class PolytropicEfficiencyResult:
    """Result of polytropic efficiency calculation"""
    efficiency: float               # Polytropic efficiency (0-1)
    polytropic_exponent: float      # n in PV^n = constant
    isentropic_efficiency: float    # Equivalent isentropic efficiency
    pressure_ratio: float           # PR
    temperature_ratio: float        # TR
    process_type: str              # "compression" or "expansion"
    confidence: float
    warnings: List[str]


def compute(
    T_inlet_K: float,
    T_outlet_K: float,
    P_inlet_Pa: float,
    P_outlet_Pa: float,
    gamma: float = 1.4,
    **kwargs
) -> PolytropicEfficiencyResult:
    """
    Compute polytropic efficiency for compression or expansion.

    Automatically detects whether this is compression or expansion based
    on pressure ratio.

    Args:
        T_inlet_K: Inlet temperature (K)
        T_outlet_K: Outlet temperature (K)
        P_inlet_Pa: Inlet pressure (Pa)
        P_outlet_Pa: Outlet pressure (Pa)
        gamma: Specific heat ratio

    Returns:
        PolytropicEfficiencyResult
    """
    warnings = []
    confidence = 1.0

    # Input validation
    if T_inlet_K <= 0 or T_outlet_K <= 0:
        warnings.append("Invalid temperature")
        confidence = 0.0
    if P_inlet_Pa <= 0 or P_outlet_Pa <= 0:
        warnings.append("Invalid pressure")
        confidence = 0.0

    if confidence == 0.0:
        return PolytropicEfficiencyResult(
            efficiency=float('nan'),
            polytropic_exponent=float('nan'),
            isentropic_efficiency=float('nan'),
            pressure_ratio=float('nan'),
            temperature_ratio=float('nan'),
            process_type="unknown",
            confidence=0.0,
            warnings=warnings,
        )

    # Determine if compression or expansion
    PR = P_outlet_Pa / P_inlet_Pa
    TR = T_outlet_K / T_inlet_K

    if PR > 1:
        process_type = "compression"
    else:
        process_type = "expansion"
        # For expansion, use inverse PR for calculations
        PR = P_inlet_Pa / P_outlet_Pa
        TR = T_inlet_K / T_outlet_K

    # Isentropic exponent
    isentropic_exp = (gamma - 1) / gamma

    # Polytropic efficiency
    # η_p = ((γ-1)/γ) × ln(PR) / ln(TR)
    if abs(math.log(TR)) < 1e-10:
        warnings.append("No temperature change - cannot compute efficiency")
        return PolytropicEfficiencyResult(
            efficiency=float('nan'),
            polytropic_exponent=float('nan'),
            isentropic_efficiency=float('nan'),
            pressure_ratio=PR,
            temperature_ratio=TR,
            process_type=process_type,
            confidence=0.0,
            warnings=warnings,
        )

    if process_type == "compression":
        eta_p = isentropic_exp * math.log(PR) / math.log(TR)
    else:  # expansion
        eta_p = math.log(TR) / (isentropic_exp * math.log(PR))

    # Polytropic exponent n
    # For compression: n/(n-1) = γ/(γ-1) × η_p
    # Solving: n = γ × η_p / (γ × η_p - γ + 1)
    if process_type == "compression":
        if abs(gamma * eta_p - gamma + 1) > 1e-10:
            n = gamma * eta_p / (gamma * eta_p - gamma + 1)
        else:
            n = float('inf')
    else:
        # For expansion: n = γ / (1 + η_p × (γ - 1))
        n = gamma / (1 + eta_p * (gamma - 1))

    # Convert to isentropic efficiency for comparison
    if process_type == "compression":
        # η_c = (PR^((γ-1)/γ) - 1) / (PR^((γ-1)/(γ×η_p)) - 1)
        if eta_p > 0:
            try:
                eta_s = (PR ** isentropic_exp - 1) / (PR ** (isentropic_exp / eta_p) - 1)
            except:
                eta_s = float('nan')
        else:
            eta_s = float('nan')
    else:
        # η_t = (1 - PR^(-(γ-1)×η_p/γ)) / (1 - PR^(-(γ-1)/γ))
        try:
            eta_s = (1 - PR ** (-isentropic_exp * eta_p)) / (1 - PR ** (-isentropic_exp))
        except:
            eta_s = float('nan')

    # Sanity checks
    if eta_p > 1.0:
        warnings.append(f"η_p > 100% ({eta_p:.1%}) - physically impossible")
        confidence *= 0.3
    elif eta_p > 0.95:
        warnings.append(f"Very high η_p ({eta_p:.1%}) - verify sensors")
        confidence *= 0.8
    elif eta_p < 0:
        warnings.append(f"Negative η_p ({eta_p:.1%}) - check data")
        confidence *= 0.1
    elif eta_p < 0.7:
        warnings.append(f"Low η_p ({eta_p:.1%}) - degraded or wrong data")
        confidence *= 0.7

    return PolytropicEfficiencyResult(
        efficiency=eta_p,
        polytropic_exponent=n,
        isentropic_efficiency=eta_s,
        pressure_ratio=PR,
        temperature_ratio=TR,
        process_type=process_type,
        confidence=confidence,
        warnings=warnings,
    )


def polytropic_to_isentropic(eta_p: float, PR: float, gamma: float = 1.4,
                              process: str = "compression") -> float:
    """
    Convert polytropic efficiency to isentropic efficiency.

    Args:
        eta_p: Polytropic efficiency
        PR: Pressure ratio
        gamma: Specific heat ratio
        process: "compression" or "expansion"

    Returns:
        Isentropic efficiency
    """
    isentropic_exp = (gamma - 1) / gamma

    if process == "compression":
        return (PR ** isentropic_exp - 1) / (PR ** (isentropic_exp / eta_p) - 1)
    else:
        return (1 - PR ** (-isentropic_exp * eta_p)) / (1 - PR ** (-isentropic_exp))


def isentropic_to_polytropic(eta_s: float, PR: float, gamma: float = 1.4,
                              process: str = "compression") -> float:
    """
    Convert isentropic efficiency to polytropic efficiency.

    Uses numerical iteration since closed-form is complex.
    """
    from scipy.optimize import brentq

    def residual(eta_p):
        return polytropic_to_isentropic(eta_p, PR, gamma, process) - eta_s

    try:
        return brentq(residual, 0.5, 0.99)
    except:
        return float('nan')


ENGINE_META = {
    'name': 'polytropic_efficiency',
    'capability': 'POLYTROPIC_EFFICIENCY',
    'description': 'Small-stage efficiency (PR-independent)',
    'requires_signals': ['T_inlet', 'T_outlet', 'P_inlet', 'P_outlet'],
    'optional_constants': ['gamma'],
    'output_unit': None,
}


if __name__ == "__main__":
    print("=" * 60)
    print("Polytropic Efficiency — Self Test")
    print("=" * 60)

    # Test 1: Compare isentropic vs polytropic at different PRs
    print("\n--- Test 1: η_p stays constant while η_c varies with PR ---")

    # Assume η_p = 0.90 polytropic
    eta_p_design = 0.90
    gamma = 1.4

    print(f"\n  Design polytropic efficiency: {eta_p_design:.1%}")
    print(f"\n  {'PR':>6s}  {'η_c (isentropic)':>16s}  {'η_p (polytropic)':>16s}")
    print("  " + "-" * 45)

    for PR in [2, 5, 10, 20, 30]:
        # For a machine with constant η_p
        eta_c = polytropic_to_isentropic(eta_p_design, PR, gamma, "compression")

        # Verify by computing η_p back from T rise
        T1 = 288.15
        T2_ideal = T1 * (PR ** ((gamma - 1) / gamma))
        T2_actual = T1 + (T2_ideal - T1) / eta_c

        result = compute(T1, T2_actual, 101325, 101325 * PR, gamma)

        print(f"  {PR:>6d}  {eta_c:>16.1%}  {result.efficiency:>16.1%}")

    print("\n  → Polytropic efficiency is constant at 90% for all PRs!")
    print("  → Isentropic efficiency decreases with PR (from 90% to 85%)")

    # Test 2: Real compressor data
    print("\n--- Test 2: Real compressor stage ---")
    result = compute(
        T_inlet_K=288.15,
        T_outlet_K=450,
        P_inlet_Pa=101325,
        P_outlet_Pa=500000,
        gamma=1.4,
    )
    print(f"  Process: {result.process_type}")
    print(f"  Pressure ratio: {result.pressure_ratio:.1f}")
    print(f"  Polytropic efficiency: {result.efficiency:.1%}")
    print(f"  Isentropic efficiency: {result.isentropic_efficiency:.1%}")
    print(f"  Polytropic exponent n: {result.polytropic_exponent:.3f}")

    # Test 3: Turbine
    print("\n--- Test 3: Turbine expansion ---")
    result = compute(
        T_inlet_K=1400,
        T_outlet_K=900,
        P_inlet_Pa=1000000,
        P_outlet_Pa=100000,
        gamma=1.33,
    )
    print(f"  Process: {result.process_type}")
    print(f"  Expansion ratio: {result.pressure_ratio:.1f}")
    print(f"  Polytropic efficiency: {result.efficiency:.1%}")
    print(f"  Isentropic efficiency: {result.isentropic_efficiency:.1%}")

    print("\n" + "=" * 60)
