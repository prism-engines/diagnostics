# Turbomachinery Domain

Physics engines for gas turbines, jet engines, compressors, and turbines.

## Description

This domain covers rotating equipment that compresses or expands gas:
- **Jet engines** (turbofan, turbojet, turboprop)
- **Gas turbines** (power generation, mechanical drive)
- **Compressors** (axial, centrifugal)
- **Turbines** (gas, steam)
- **Turbochargers**

Validated against NASA C-MAPSS turbofan engine simulation data.

## Required Signals

At minimum, you need inlet/outlet conditions for the component being analyzed:

| Pattern | Unit Category | Description |
|---------|---------------|-------------|
| `T_inlet`, `T_in`, `T2`, `T1` | temperature | Inlet total temperature |
| `T_outlet`, `T_out`, `T3`, `T2` | temperature | Outlet total temperature |
| `P_inlet`, `P_in`, `P2`, `P1` | pressure | Inlet total pressure |
| `P_outlet`, `P_out`, `P3`, `P2` | pressure | Outlet total pressure |

For full engine analysis (Brayton cycle), you need multiple stations:

| C-MAPSS Name | Station | Description |
|--------------|---------|-------------|
| `T2` | 2 | Fan inlet total temp |
| `T24` | 24 | LPC outlet total temp |
| `T30` | 30 | HPC outlet total temp |
| `T50` | 50 | LPT outlet total temp |
| `P2` | 2 | Fan inlet total pressure |
| `P15` | 15 | Bypass duct pressure |
| `P30` | 30 | HPC outlet total pressure |
| `Ps30` | 30 | HPC outlet static pressure |
| `Nf` | - | Fan speed (rpm) |
| `Nc` | - | Core speed (rpm) |

## Optional Signals

| Pattern | Unit Category | Description |
|---------|---------------|-------------|
| `W`, `mdot`, `mass_flow` | mass_flow | Mass flow rate |
| `N`, `rpm`, `speed` | frequency | Rotational speed |
| `fuel_flow`, `Wf` | mass_flow | Fuel mass flow rate |
| `thrust`, `F` | force | Net thrust |

## Required Constants

None required — all have sensible defaults for air.

## Optional Constants

| Name | Unit | Default | Description |
|------|------|---------|-------------|
| `gamma` | - | 1.4 | Specific heat ratio (air) |
| `R_gas` | J/(kg·K) | 287 | Gas constant (air) |
| `cp` | J/(kg·K) | 1005 | Specific heat at constant P |
| `T_ref` | K | 288.15 | Reference temperature (ISA sea level) |
| `P_ref` | Pa | 101325 | Reference pressure (ISA sea level) |

## Capabilities Provided

| Capability | Engine | Output | Description |
|------------|--------|--------|-------------|
| `COMPRESSOR_EFFICIENCY` | compressor_efficiency.py | η_c (0-1) | Isentropic efficiency of compression |
| `TURBINE_EFFICIENCY` | turbine_efficiency.py | η_t (0-1) | Isentropic efficiency of expansion |
| `PRESSURE_RATIO` | pressure_ratio.py | PR | Pressure ratio across component |
| `POLYTROPIC_EFFICIENCY` | polytropic_efficiency.py | η_p (0-1) | Small-stage efficiency |
| `CORRECTED_FLOW` | corrected_parameters.py | W_corr | Mass flow corrected to standard day |
| `CORRECTED_SPEED` | corrected_parameters.py | N_corr | Speed corrected to standard day |
| `WORK_COEFFICIENT` | work_coefficient.py | ψ | Dimensionless work parameter |
| `FLOW_COEFFICIENT` | flow_coefficient.py | φ | Dimensionless flow parameter |
| `THERMAL_EFFICIENCY` | brayton_cycle.py | η_th | Overall thermal efficiency |
| `SFC` | specific_fuel_consumption.py | SFC | Specific fuel consumption |

## Equations

### Isentropic Efficiency (Compressor)
```
η_c = (T₂ₛ - T₁) / (T₂ - T₁)

where T₂ₛ = T₁ × (P₂/P₁)^((γ-1)/γ)
```
T₂ₛ is the ideal (isentropic) outlet temperature.

### Isentropic Efficiency (Turbine)
```
η_t = (T₃ - T₄) / (T₃ - T₄ₛ)

where T₄ₛ = T₃ × (P₄/P₃)^((γ-1)/γ)
```
Note: Turbine efficiency uses actual/ideal (inverse of compressor).

### Polytropic Efficiency
```
η_p = ((γ-1)/γ) × ln(P₂/P₁) / ln(T₂/T₁)
```
More accurate for comparing compressors of different pressure ratios.

### Corrected Parameters
```
W_corr = W × √(T/T_ref) / (P/P_ref)
N_corr = N / √(T/T_ref)
```
Allows comparison across different ambient conditions.

### Thermal Efficiency (Brayton Cycle)
```
η_th = W_net / Q_in = (W_turbine - W_compressor) / (m_fuel × LHV)
```

## Validated Against

| Dataset | Expected | Notes |
|---------|----------|-------|
| C-MAPSS FD001 | η_c: 0.82-0.87 | New engines |
| C-MAPSS FD001 | η_c: 0.70-0.78 | End of life |
| NASA TM-2006-214070 | η_c: 0.85 typical | Reference data |

## References

1. Mattingly, J.D. "Elements of Gas Turbine Propulsion" - Core equations
2. Walsh & Fletcher "Gas Turbine Performance" - Corrected parameters
3. Saravanamuttoo "Gas Turbine Theory" - Polytropic efficiency
4. NASA C-MAPSS documentation - Sensor definitions

## Contributors

- PRISM Team (initial implementation)
- Validated with C-MAPSS public dataset
