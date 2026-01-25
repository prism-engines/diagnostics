"""
Turbomachinery Domain — Gas Turbines, Compressors, Turbines

Physics engines for analyzing rotating compression and expansion equipment.
Validated against NASA C-MAPSS turbofan engine simulation data.

Capabilities:
    - COMPRESSOR_EFFICIENCY: Isentropic efficiency of compression
    - TURBINE_EFFICIENCY: Isentropic efficiency of expansion
    - POLYTROPIC_EFFICIENCY: Small-stage efficiency (PR-independent)
    - CORRECTED_PARAMETERS: Standard day normalization
    - BRAYTON_CYCLE: Complete gas turbine cycle analysis

Usage:
    from domains.turbomachinery import compressor_efficiency
    
    result = compressor_efficiency.compute(
        T_inlet_K=288.15,
        T_outlet_K=600,
        P_inlet_Pa=101325,
        P_outlet_Pa=1013250,
    )
    print(f"Efficiency: {result.efficiency:.1%}")

With UnitSpec:
    from prism.unitspec import Q
    from domains.turbomachinery import compressor_efficiency
    
    result = compressor_efficiency.compute(
        T_inlet_K=Q("70 °F").to("K"),
        T_outlet_K=Q("620 °F").to("K"),
        P_inlet_Pa=Q("14.7 psi").to("Pa"),
        P_outlet_Pa=Q("147 psi").to("Pa"),
    )
"""

from . import compressor_efficiency
from . import turbine_efficiency
from . import polytropic_efficiency
from . import corrected_parameters
from . import brayton_cycle

# Engine registry
ENGINES = {
    'compressor_efficiency': compressor_efficiency.compute,
    'turbine_efficiency': turbine_efficiency.compute,
    'polytropic_efficiency': polytropic_efficiency.compute,
    'corrected_parameters': corrected_parameters.compute,
    'brayton_cycle': brayton_cycle.compute,
}

# Capability mapping
CAPABILITIES = {
    'COMPRESSOR_EFFICIENCY': compressor_efficiency,
    'TURBINE_EFFICIENCY': turbine_efficiency,
    'POLYTROPIC_EFFICIENCY': polytropic_efficiency,
    'CORRECTED_PARAMETERS': corrected_parameters,
    'BRAYTON_CYCLE': brayton_cycle,
}

# Required signals for each capability
REQUIREMENTS = {
    'COMPRESSOR_EFFICIENCY': {
        'signals': ['T_inlet', 'T_outlet', 'P_inlet', 'P_outlet'],
        'constants': [],
    },
    'TURBINE_EFFICIENCY': {
        'signals': ['T_inlet', 'T_outlet', 'P_inlet', 'P_outlet'],
        'constants': [],
    },
    'POLYTROPIC_EFFICIENCY': {
        'signals': ['T_inlet', 'T_outlet', 'P_inlet', 'P_outlet'],
        'constants': [],
    },
    'CORRECTED_PARAMETERS': {
        'signals': ['T_ambient', 'P_ambient'],
        'constants': [],
    },
    'BRAYTON_CYCLE': {
        'signals': ['T1', 'T2', 'T3', 'T4', 'P1', 'P2'],
        'constants': [],
    },
}


def check_requirements(signals: list, constants: dict, capability: str) -> dict:
    """
    Check if data supports a given capability.
    
    Args:
        signals: List of available signal names
        constants: Dict of available constants
        capability: Capability to check
        
    Returns:
        {
            'available': bool,
            'missing_signals': list,
            'missing_constants': list,
        }
    """
    req = REQUIREMENTS.get(capability, {'signals': [], 'constants': []})
    
    # Normalize signal names for matching
    signal_lower = [s.lower() for s in signals]
    
    missing_signals = []
    for req_sig in req['signals']:
        # Check for exact match or common variations
        found = False
        variations = [req_sig.lower(), req_sig.lower().replace('_', '')]
        for var in variations:
            if any(var in s for s in signal_lower):
                found = True
                break
        if not found:
            missing_signals.append(req_sig)
    
    missing_constants = []
    for req_const in req['constants']:
        if req_const not in constants:
            missing_constants.append(req_const)
    
    return {
        'available': len(missing_signals) == 0 and len(missing_constants) == 0,
        'missing_signals': missing_signals,
        'missing_constants': missing_constants,
    }


def analyze_cmapss(df, engine_id: int = None):
    """
    Convenience function to analyze C-MAPSS data.
    
    C-MAPSS column mapping:
        T2  → Fan inlet temp (R)
        T24 → LPC outlet temp (R) [not always available]
        T30 → HPC outlet temp (R)
        T50 → LPT outlet temp (R)
        P2  → Fan inlet pressure (psia)
        P15 → Bypass duct pressure (psia)
        P30 → HPC outlet pressure (psia)
        Ps30 → HPC static pressure (psia)
        Nf  → Fan speed (rpm)
        Nc  → Core speed (rpm)
    """
    results = {}
    
    # Filter to single engine if specified
    if engine_id is not None and 'unit' in df.columns:
        df = df[df['unit'] == engine_id]
    elif engine_id is not None and 'engine_id' in df.columns:
        df = df[df['engine_id'] == engine_id]
    
    # Convert Rankine to Kelvin
    def R_to_K(r):
        return r * 5 / 9
    
    # Convert psia to Pa
    def psia_to_Pa(p):
        return p * 6894.76
    
    # Check what columns we have
    has_T2 = 'T2' in df.columns or 's2' in df.columns
    has_T30 = 'T30' in df.columns or 's11' in df.columns
    has_T50 = 'T50' in df.columns or 's17' in df.columns
    has_P2 = 'P2' in df.columns or 's3' in df.columns
    has_P30 = 'P30' in df.columns or 's13' in df.columns
    
    # Compressor efficiency (fan inlet to HPC outlet)
    if has_T2 and has_T30 and has_P2 and has_P30:
        T2_col = 'T2' if 'T2' in df.columns else 's2'
        T30_col = 'T30' if 'T30' in df.columns else 's11'
        P2_col = 'P2' if 'P2' in df.columns else 's3'
        P30_col = 'P30' if 'P30' in df.columns else 's13'
        
        efficiencies = []
        for _, row in df.iterrows():
            result = compressor_efficiency.compute(
                T_inlet_K=R_to_K(row[T2_col]),
                T_outlet_K=R_to_K(row[T30_col]),
                P_inlet_Pa=psia_to_Pa(row[P2_col]),
                P_outlet_Pa=psia_to_Pa(row[P30_col]),
            )
            efficiencies.append(result.efficiency)
        
        results['compressor_efficiency'] = efficiencies
    
    return results


__all__ = [
    'compressor_efficiency',
    'turbine_efficiency', 
    'polytropic_efficiency',
    'corrected_parameters',
    'brayton_cycle',
    'ENGINES',
    'CAPABILITIES',
    'REQUIREMENTS',
    'check_requirements',
    'analyze_cmapss',
]
