"""
Capability Detection and Routing

Determines what physics can be computed from available data.
Routes data to appropriate engines.
Reports what's possible and what's missing.

The PRISM Curriculum:
    Level 0: Raw time series        → Statistics, entropy, memory
    Level 1: Labeled signals        → Derivatives, specific energy
    Level 2: Physical constants     → Real energy, momentum
    Level 3: Related signals        → Gibbs, transfer functions
    Level 4: Spatial fields         → Navier-Stokes, Maxwell

You bring the data. We compute everything possible.
You bring more context. We unlock more physics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum, auto


# =============================================================================
# DATA LEVELS
# =============================================================================

class DataLevel(Enum):
    """
    What level of data context do we have?

    Each level unlocks more physics capabilities.
    """
    RAW_TIMESERIES = 0      # Just numbers in columns
    LABELED_SIGNALS = 1     # Know what signals represent physically
    WITH_CONSTANTS = 2      # Have physical constants (mass, k, etc.)
    RELATED_SIGNALS = 3     # Multiple signals with known relationships
    SPATIAL_FIELD = 4       # Full 3D/4D field data


# =============================================================================
# CAPABILITIES
# =============================================================================

class Capability(Enum):
    """
    What can PRISM compute?

    Organized by the data level required to unlock each capability.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 0: Always available (raw time series)
    # ─────────────────────────────────────────────────────────────────────────
    STATISTICS = auto()           # mean, std, skew, kurtosis
    DISTRIBUTION = auto()         # distribution fitting
    STATIONARITY = auto()         # ADF, KPSS tests
    ENTROPY = auto()              # sample, permutation, spectral entropy
    MEMORY = auto()               # Hurst exponent (R/S, DFA)
    SPECTRAL = auto()             # FFT, wavelets, power spectrum
    RECURRENCE = auto()           # RQA metrics
    CHAOS = auto()                # Lyapunov exponent
    VOLATILITY = auto()           # realized vol, GARCH
    EVENT_DETECTION = auto()      # step and spike detection
    HEAVISIDE_DIRAC = auto()      # H(t-t₀), δ(t-t₀) at events
    GEOMETRY = auto()             # PCA, manifold, clustering
    DYNAMICS = auto()             # trajectory, regimes, hd_slope

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 1: With signal labels (know what x represents)
    # ─────────────────────────────────────────────────────────────────────────
    DERIVATIVES = auto()          # dx/dt, d²x/dt² with proper units
    SPECIFIC_KINETIC = auto()     # T/m = ½v² [J/kg]
    SPECIFIC_POTENTIAL = auto()   # V/k = ½x² [m²]
    SPECIFIC_HAMILTONIAN = auto() # H = T/m + V/k (unitless proxy)

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 2: With physical constants
    # ─────────────────────────────────────────────────────────────────────────
    KINETIC_ENERGY = auto()       # T = ½mv² [J]
    POTENTIAL_ENERGY = auto()     # V = ½kx² [J]
    MOMENTUM = auto()             # p = mv [kg·m/s]
    HAMILTONIAN = auto()          # H = T + V [J]
    LAGRANGIAN = auto()           # L = T - V [J]
    ROTATIONAL_KE = auto()        # T = ½Iω² [J]

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 3: With related signals
    # ─────────────────────────────────────────────────────────────────────────
    WORK = auto()                 # W = ∫F·dx [J]
    POWER = auto()                # P = F·v [W]
    ANGULAR_MOMENTUM = auto()     # L = r × p [kg·m²/s]
    PHASE_SPACE = auto()          # (q, p) trajectories
    ENERGY_CONSERVATION = auto()  # dH/dt ≈ 0 check

    GIBBS_FREE_ENERGY = auto()    # G = H - TS [J]
    ENTHALPY = auto()             # H = U + PV [J]
    ENTROPY_THERMO = auto()       # S from T, P, V [J/K]
    CHEMICAL_POTENTIAL = auto()   # μ = ∂G/∂n [J/mol]

    TRANSFER_FUNCTION = auto()    # G(s) from step/impulse response
    FREQUENCY_RESPONSE = auto()   # Bode plot, bandwidth
    POLES_ZEROS = auto()          # Stability analysis
    IMPULSE_RESPONSE = auto()     # h(t) = L⁻¹{G(s)}

    GRANGER_CAUSALITY = auto()    # Causal relationships
    TRANSFER_ENTROPY = auto()     # Directed information flow

    # ─────────────────────────────────────────────────────────────────────────
    # LEVEL 4: Spatial field data
    # ─────────────────────────────────────────────────────────────────────────
    VORTICITY = auto()            # ω = ∇ × v
    STRAIN_TENSOR = auto()        # S_ij = ½(∂vᵢ/∂xⱼ + ∂vⱼ/∂xᵢ)
    Q_CRITERION = auto()          # Q = ½(||Ω||² - ||S||²)
    TURBULENT_KE = auto()         # k = ½⟨u'ᵢu'ᵢ⟩
    DISSIPATION = auto()          # ε = 2ν⟨SᵢⱼSᵢⱼ⟩
    ENERGY_SPECTRUM = auto()      # E(k) with k⁻⁵/³ check
    REYNOLDS_NUMBER = auto()      # Re = UL/ν
    KOLMOGOROV_SCALES = auto()    # η, τ_η, v_η

    HEAT_FLUX = auto()            # q = -k∇T
    LAPLACIAN_T = auto()          # ∇²T

    MAXWELL_DIV_E = auto()        # ∇·E = ρ/ε₀
    MAXWELL_CURL_B = auto()       # ∇×B = μ₀J + μ₀ε₀∂E/∂t
    POYNTING = auto()             # S = E × B / μ₀


# =============================================================================
# PHYSICAL QUANTITIES
# =============================================================================

class PhysicalQuantity(Enum):
    """Known physical quantities that signals can represent."""
    # Mechanical
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FORCE = "force"
    ANGULAR_POSITION = "angular_position"
    ANGULAR_VELOCITY = "angular_velocity"

    # Thermodynamic
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VOLUME = "volume"
    HEAT = "heat"

    # Electrical
    VOLTAGE = "voltage"
    CURRENT = "current"

    # Field components
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    VELOCITY_Z = "velocity_z"

    # Control
    INPUT = "input"
    OUTPUT = "output"

    # Generic
    UNKNOWN = "unknown"


# =============================================================================
# UNITS
# =============================================================================

STANDARD_UNITS = {
    PhysicalQuantity.POSITION: "m",
    PhysicalQuantity.VELOCITY: "m/s",
    PhysicalQuantity.ACCELERATION: "m/s²",
    PhysicalQuantity.FORCE: "N",
    PhysicalQuantity.ANGULAR_POSITION: "rad",
    PhysicalQuantity.ANGULAR_VELOCITY: "rad/s",
    PhysicalQuantity.TEMPERATURE: "K",
    PhysicalQuantity.PRESSURE: "Pa",
    PhysicalQuantity.VOLUME: "m³",
    PhysicalQuantity.VOLTAGE: "V",
    PhysicalQuantity.CURRENT: "A",
}


# =============================================================================
# SIGNAL SPECIFICATION
# =============================================================================

@dataclass
class SignalSpec:
    """Specification for a single signal/column."""
    name: str
    physical_quantity: PhysicalQuantity = PhysicalQuantity.UNKNOWN
    units: Optional[str] = None

    # For 3D vectors
    is_component: bool = False
    vector_name: Optional[str] = None  # e.g., "velocity" for velocity_x
    component: Optional[str] = None    # e.g., "x"

    @classmethod
    def from_dict(cls, d: Dict) -> 'SignalSpec':
        qty = d.get('physical_quantity', 'unknown')
        if isinstance(qty, str):
            try:
                qty = PhysicalQuantity(qty)
            except ValueError:
                qty = PhysicalQuantity.UNKNOWN

        return cls(
            name=d['name'],
            physical_quantity=qty,
            units=d.get('units'),
            is_component=d.get('is_component', False),
            vector_name=d.get('vector_name'),
            component=d.get('component'),
        )


# =============================================================================
# CONSTANTS SPECIFICATION
# =============================================================================

@dataclass
class ConstantsSpec:
    """Physical constants provided by user."""
    # Mechanical
    mass: Optional[float] = None                    # kg
    spring_constant: Optional[float] = None         # N/m
    moment_of_inertia: Optional[float] = None       # kg·m²
    damping_coefficient: Optional[float] = None     # N·s/m

    # Thermodynamic
    Cp: Optional[float] = None                      # J/(mol·K)
    Cv: Optional[float] = None                      # J/(mol·K)
    n_moles: Optional[float] = None                 # mol

    # Fluid
    kinematic_viscosity: Optional[float] = None     # m²/s
    density: Optional[float] = None                 # kg/m³

    # Spatial
    dx: Optional[float] = None                      # m
    dy: Optional[float] = None                      # m
    dz: Optional[float] = None                      # m
    dt: Optional[float] = None                      # s

    @classmethod
    def from_dict(cls, d: Dict) -> 'ConstantsSpec':
        return cls(
            mass=d.get('mass'),
            spring_constant=d.get('spring_constant'),
            moment_of_inertia=d.get('moment_of_inertia'),
            damping_coefficient=d.get('damping_coefficient'),
            Cp=d.get('Cp'),
            Cv=d.get('Cv'),
            n_moles=d.get('n_moles'),
            kinematic_viscosity=d.get('kinematic_viscosity') or d.get('nu'),
            density=d.get('density') or d.get('rho'),
            dx=d.get('dx'),
            dy=d.get('dy'),
            dz=d.get('dz'),
            dt=d.get('dt'),
        )


# =============================================================================
# RELATIONSHIPS SPECIFICATION
# =============================================================================

@dataclass
class MechanicalRelationship:
    """Mechanical signal relationships."""
    position: Optional[str] = None      # Column name
    velocity: Optional[str] = None
    acceleration: Optional[str] = None
    force: Optional[str] = None

    # Derive flags
    derive_velocity: bool = False       # v = dx/dt
    derive_acceleration: bool = False   # a = dv/dt


@dataclass
class ThermodynamicRelationship:
    """Thermodynamic signal relationships."""
    temperature: Optional[str] = None   # Column name
    pressure: Optional[str] = None
    volume: Optional[str] = None
    heat: Optional[str] = None


@dataclass
class ControlRelationship:
    """Control system I/O relationship."""
    input: Optional[str] = None         # Input signal column
    output: Optional[str] = None        # Output signal column


@dataclass
class Vector3D:
    """3D vector from component signals."""
    name: str
    x: str      # Column name for x component
    y: str      # Column name for y component
    z: str      # Column name for z component


@dataclass
class RelationshipsSpec:
    """All signal relationships."""
    mechanical: Optional[MechanicalRelationship] = None
    thermodynamic: Optional[ThermodynamicRelationship] = None
    control: Optional[ControlRelationship] = None
    vectors_3d: List[Vector3D] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> 'RelationshipsSpec':
        mech = d.get('mechanical', {})
        thermo = d.get('thermodynamic', {})
        control = d.get('control', {})
        vectors = d.get('vectors_3d', [])

        return cls(
            mechanical=MechanicalRelationship(
                position=mech.get('position'),
                velocity=mech.get('velocity'),
                acceleration=mech.get('acceleration'),
                force=mech.get('force'),
                derive_velocity=mech.get('derive_velocity', False),
                derive_acceleration=mech.get('derive_acceleration', False),
            ) if mech else None,
            thermodynamic=ThermodynamicRelationship(
                temperature=thermo.get('temperature'),
                pressure=thermo.get('pressure'),
                volume=thermo.get('volume'),
                heat=thermo.get('heat'),
            ) if thermo else None,
            control=ControlRelationship(
                input=control.get('input'),
                output=control.get('output'),
            ) if control else None,
            vectors_3d=[
                Vector3D(name=v['name'], x=v['x'], y=v['y'], z=v['z'])
                for v in vectors
            ],
        )


# =============================================================================
# SPATIAL SPECIFICATION
# =============================================================================

class SpatialType(Enum):
    """Type of spatial field data."""
    NONE = "none"
    VELOCITY_FIELD = "velocity_field"
    TEMPERATURE_FIELD = "temperature_field"
    EM_FIELD = "em_field"


@dataclass
class SpatialSpec:
    """Spatial field data specification."""
    type: SpatialType = SpatialType.NONE
    dimensions: Optional[Tuple[int, int, int]] = None  # nx, ny, nz

    # Component column names
    u: Optional[str] = None   # velocity_x or E_x
    v: Optional[str] = None   # velocity_y or E_y
    w: Optional[str] = None   # velocity_z or E_z

    # For EM fields
    Bx: Optional[str] = None
    By: Optional[str] = None
    Bz: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict) -> 'SpatialSpec':
        if not d:
            return cls()

        type_str = d.get('type', 'none')
        try:
            spatial_type = SpatialType(type_str)
        except ValueError:
            spatial_type = SpatialType.NONE

        dims = d.get('dimensions')
        if dims:
            dims = tuple(dims)

        return cls(
            type=spatial_type,
            dimensions=dims,
            u=d.get('u') or d.get('velocity_x'),
            v=d.get('v') or d.get('velocity_y'),
            w=d.get('w') or d.get('velocity_z'),
            Bx=d.get('Bx'),
            By=d.get('By'),
            Bz=d.get('Bz'),
        )


# =============================================================================
# FULL DATA SPECIFICATION
# =============================================================================

@dataclass
class DataSpec:
    """Complete specification of available data."""
    signals: List[SignalSpec]
    constants: ConstantsSpec
    relationships: RelationshipsSpec
    spatial: SpatialSpec

    # Windowing (required)
    window_size: int
    window_stride: int

    @classmethod
    def from_config(cls, config: Dict) -> 'DataSpec':
        """Parse configuration dict into DataSpec."""
        signals = [
            SignalSpec.from_dict(s) if isinstance(s, dict) else SignalSpec(name=s)
            for s in config.get('signals', [])
        ]

        return cls(
            signals=signals,
            constants=ConstantsSpec.from_dict(config.get('constants', {})),
            relationships=RelationshipsSpec.from_dict(config.get('relationships', {})),
            spatial=SpatialSpec.from_dict(config.get('spatial', {})),
            window_size=config.get('window_size', 50),
            window_stride=config.get('window_stride', 25),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def has_signal_type(self, qty: PhysicalQuantity) -> bool:
        """Check if we have a signal of given type."""
        return any(s.physical_quantity == qty for s in self.signals)

    def get_signal(self, qty: PhysicalQuantity) -> Optional[SignalSpec]:
        """Get signal spec for a physical quantity."""
        for s in self.signals:
            if s.physical_quantity == qty:
                return s
        return None

    def has_labeled_signals(self) -> bool:
        """Check if any signals have physical quantity labels."""
        return any(s.physical_quantity != PhysicalQuantity.UNKNOWN for s in self.signals)

    def has_velocity(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.VELOCITY) or
            (self.relationships.mechanical and self.relationships.mechanical.velocity)
        )

    def has_position(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.POSITION) or
            (self.relationships.mechanical and self.relationships.mechanical.position)
        )

    def has_force(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.FORCE) or
            (self.relationships.mechanical and self.relationships.mechanical.force)
        )

    def has_temperature(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.TEMPERATURE) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.temperature)
        )

    def has_pressure(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.PRESSURE) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.pressure)
        )

    def has_volume(self) -> bool:
        return (
            self.has_signal_type(PhysicalQuantity.VOLUME) or
            (self.relationships.thermodynamic and self.relationships.thermodynamic.volume)
        )

    def has_io_pair(self) -> bool:
        return (
            self.relationships.control and
            self.relationships.control.input and
            self.relationships.control.output
        )

    def has_3d_position(self) -> bool:
        return any(v.name == 'position' for v in self.relationships.vectors_3d)

    def has_3d_velocity(self) -> bool:
        return any(v.name == 'velocity' for v in self.relationships.vectors_3d)

    def has_velocity_field(self) -> bool:
        return self.spatial.type == SpatialType.VELOCITY_FIELD


# =============================================================================
# CAPABILITY REQUIREMENTS
# =============================================================================

# What each capability requires
CAPABILITY_REQUIREMENTS: Dict[Capability, Dict[str, Any]] = {
    # Level 0: No requirements beyond data existing
    Capability.STATISTICS: {},
    Capability.DISTRIBUTION: {},
    Capability.STATIONARITY: {},
    Capability.ENTROPY: {},
    Capability.MEMORY: {},
    Capability.SPECTRAL: {},
    Capability.RECURRENCE: {},
    Capability.CHAOS: {},
    Capability.VOLATILITY: {},
    Capability.EVENT_DETECTION: {},
    Capability.HEAVISIDE_DIRAC: {},
    Capability.GEOMETRY: {},
    Capability.DYNAMICS: {},

    # Level 1: Need signal labels
    Capability.DERIVATIVES: {'labeled': True},
    Capability.SPECIFIC_KINETIC: {'signals': ['velocity']},
    Capability.SPECIFIC_POTENTIAL: {'signals': ['position']},
    Capability.SPECIFIC_HAMILTONIAN: {'signals': ['position', 'velocity']},

    # Level 2: Need constants
    Capability.KINETIC_ENERGY: {'signals': ['velocity'], 'constants': ['mass']},
    Capability.POTENTIAL_ENERGY: {'signals': ['position'], 'constants': ['spring_constant']},
    Capability.MOMENTUM: {'signals': ['velocity'], 'constants': ['mass']},
    Capability.HAMILTONIAN: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},
    Capability.LAGRANGIAN: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},
    Capability.ROTATIONAL_KE: {'signals': ['angular_velocity'], 'constants': ['moment_of_inertia']},

    # Level 3: Need related signals
    Capability.WORK: {'signals': ['position', 'force']},
    Capability.POWER: {'signals': ['velocity', 'force']},
    Capability.ANGULAR_MOMENTUM: {'vectors_3d': ['position', 'velocity'], 'constants': ['mass']},
    Capability.PHASE_SPACE: {'signals': ['position', 'velocity'], 'constants': ['mass']},
    Capability.ENERGY_CONSERVATION: {'signals': ['position', 'velocity'], 'constants': ['mass', 'spring_constant']},

    Capability.GIBBS_FREE_ENERGY: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.ENTHALPY: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.ENTROPY_THERMO: {'thermo': ['temperature', 'pressure', 'volume'], 'constants': ['Cp']},
    Capability.CHEMICAL_POTENTIAL: {'thermo': ['temperature', 'pressure'], 'constants': ['Cp']},

    Capability.TRANSFER_FUNCTION: {'io_pair': True},
    Capability.FREQUENCY_RESPONSE: {'io_pair': True},
    Capability.POLES_ZEROS: {'io_pair': True},
    Capability.IMPULSE_RESPONSE: {'io_pair': True},

    Capability.GRANGER_CAUSALITY: {'min_signals': 2},
    Capability.TRANSFER_ENTROPY: {'min_signals': 2},

    # Level 4: Need spatial data
    Capability.VORTICITY: {'spatial': 'velocity_field'},
    Capability.STRAIN_TENSOR: {'spatial': 'velocity_field'},
    Capability.Q_CRITERION: {'spatial': 'velocity_field'},
    Capability.TURBULENT_KE: {'spatial': 'velocity_field'},
    Capability.DISSIPATION: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},
    Capability.ENERGY_SPECTRUM: {'spatial': 'velocity_field'},
    Capability.REYNOLDS_NUMBER: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},
    Capability.KOLMOGOROV_SCALES: {'spatial': 'velocity_field', 'constants': ['kinematic_viscosity']},

    Capability.HEAT_FLUX: {'spatial': 'temperature_field'},
    Capability.LAPLACIAN_T: {'spatial': 'temperature_field'},

    Capability.MAXWELL_DIV_E: {'spatial': 'em_field'},
    Capability.MAXWELL_CURL_B: {'spatial': 'em_field'},
    Capability.POYNTING: {'spatial': 'em_field'},
}


# =============================================================================
# CAPABILITY REPORT
# =============================================================================

@dataclass
class CapabilityReport:
    """
    Report of what PRISM can compute from the available data.
    """
    available: Set[Capability]
    unavailable: Dict[Capability, str]  # Capability → what's missing
    data_level: DataLevel
    data_spec: DataSpec

    def __str__(self) -> str:
        """Pretty print the capability report."""
        lines = [
            "=" * 70,
            "PRISM CAPABILITY REPORT",
            "=" * 70,
            f"",
            f"Data Level: {self.data_level.name}",
            f"",
        ]

        # Group available by level
        level_0 = []
        level_1 = []
        level_2 = []
        level_3 = []
        level_4 = []

        for cap in sorted(self.available, key=lambda x: x.name):
            reqs = CAPABILITY_REQUIREMENTS.get(cap, {})
            if not reqs:
                level_0.append(cap)
            elif 'spatial' in reqs:
                level_4.append(cap)
            elif 'io_pair' in reqs or 'thermo' in reqs or 'vectors_3d' in reqs:
                level_3.append(cap)
            elif 'constants' in reqs:
                level_2.append(cap)
            else:
                level_1.append(cap)

        lines.append(f"AVAILABLE CAPABILITIES ({len(self.available)})")
        lines.append("-" * 70)

        if level_0:
            lines.append(f"\n  Level 0 - Raw Time Series ({len(level_0)}):")
            for cap in level_0:
                lines.append(f"    [x] {cap.name}")

        if level_1:
            lines.append(f"\n  Level 1 - Labeled Signals ({len(level_1)}):")
            for cap in level_1:
                lines.append(f"    [x] {cap.name}")

        if level_2:
            lines.append(f"\n  Level 2 - With Constants ({len(level_2)}):")
            for cap in level_2:
                lines.append(f"    [x] {cap.name}")

        if level_3:
            lines.append(f"\n  Level 3 - Related Signals ({len(level_3)}):")
            for cap in level_3:
                lines.append(f"    [x] {cap.name}")

        if level_4:
            lines.append(f"\n  Level 4 - Spatial Fields ({len(level_4)}):")
            for cap in level_4:
                lines.append(f"    [x] {cap.name}")

        if self.unavailable:
            lines.append(f"\n")
            lines.append(f"UNAVAILABLE - What's Missing ({len(self.unavailable)})")
            lines.append("-" * 70)
            for cap, missing in sorted(self.unavailable.items(), key=lambda x: x[0].name):
                lines.append(f"    [ ] {cap.name}")
                lines.append(f"        -> need: {missing}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'data_level': self.data_level.name,
            'available': [c.name for c in sorted(self.available, key=lambda x: x.name)],
            'unavailable': {
                c.name: msg for c, msg in self.unavailable.items()
            },
            'n_available': len(self.available),
            'n_unavailable': len(self.unavailable),
        }


# =============================================================================
# CAPABILITY DETECTOR
# =============================================================================

def check_requirements(spec: DataSpec, reqs: Dict) -> Optional[str]:
    """
    Check if requirements are met.

    Returns None if met, or a string describing what's missing.
    """
    missing = []

    # Check for labeled signals requirement
    if reqs.get('labeled') and not spec.has_labeled_signals():
        return "signal labels (physical_quantity)"

    # Check for specific signal types
    signals_needed = reqs.get('signals', [])
    for sig in signals_needed:
        if sig == 'velocity' and not spec.has_velocity():
            missing.append("velocity signal")
        elif sig == 'position' and not spec.has_position():
            missing.append("position signal")
        elif sig == 'force' and not spec.has_force():
            missing.append("force signal")
        elif sig == 'angular_velocity' and not spec.has_signal_type(PhysicalQuantity.ANGULAR_VELOCITY):
            missing.append("angular_velocity signal")

    # Check for constants
    constants_needed = reqs.get('constants', [])
    for const in constants_needed:
        if const == 'mass' and not spec.constants.mass:
            missing.append("mass [kg]")
        elif const == 'spring_constant' and not spec.constants.spring_constant:
            missing.append("spring_constant [N/m]")
        elif const == 'moment_of_inertia' and not spec.constants.moment_of_inertia:
            missing.append("moment_of_inertia [kg*m^2]")
        elif const == 'Cp' and not spec.constants.Cp:
            missing.append("Cp [J/(mol*K)]")
        elif const == 'kinematic_viscosity' and not spec.constants.kinematic_viscosity:
            missing.append("kinematic_viscosity (nu) [m^2/s]")

    # Check for thermodynamic signals
    thermo_needed = reqs.get('thermo', [])
    for sig in thermo_needed:
        if sig == 'temperature' and not spec.has_temperature():
            missing.append("temperature signal")
        elif sig == 'pressure' and not spec.has_pressure():
            missing.append("pressure signal")
        elif sig == 'volume' and not spec.has_volume():
            missing.append("volume signal")

    # Check for I/O pair
    if reqs.get('io_pair') and not spec.has_io_pair():
        missing.append("input + output signal pair")

    # Check for 3D vectors
    vectors_needed = reqs.get('vectors_3d', [])
    for vec in vectors_needed:
        if vec == 'position' and not spec.has_3d_position():
            missing.append("3D position vector (x, y, z)")
        elif vec == 'velocity' and not spec.has_3d_velocity():
            missing.append("3D velocity vector (vx, vy, vz)")

    # Check for spatial field
    spatial_needed = reqs.get('spatial')
    if spatial_needed:
        if spatial_needed == 'velocity_field' and not spec.has_velocity_field():
            missing.append("velocity field v(x,y,z,t)")
        elif spatial_needed == 'temperature_field' and spec.spatial.type != SpatialType.TEMPERATURE_FIELD:
            missing.append("temperature field T(x,y,z,t)")
        elif spatial_needed == 'em_field' and spec.spatial.type != SpatialType.EM_FIELD:
            missing.append("electromagnetic field E(x,y,z,t), B(x,y,z,t)")

    # Check minimum signals
    min_signals = reqs.get('min_signals')
    if min_signals and len(spec.signals) < min_signals:
        missing.append(f"at least {min_signals} signals")

    if missing:
        return ", ".join(missing)

    return None


def detect_capabilities(config: Dict) -> CapabilityReport:
    """
    Analyze configuration and determine what's computable.

    This is the main entry point for capability detection.

    Args:
        config: Configuration dictionary with signals, constants, relationships

    Returns:
        CapabilityReport with available/unavailable capabilities
    """
    spec = DataSpec.from_config(config)

    available: Set[Capability] = set()
    unavailable: Dict[Capability, str] = {}

    # Determine data level
    data_level = DataLevel.RAW_TIMESERIES

    if spec.has_labeled_signals():
        data_level = DataLevel.LABELED_SIGNALS

    has_any_constant = any([
        spec.constants.mass,
        spec.constants.spring_constant,
        spec.constants.moment_of_inertia,
        spec.constants.Cp,
    ])
    if has_any_constant:
        data_level = DataLevel.WITH_CONSTANTS

    has_relationships = any([
        spec.has_io_pair(),
        (spec.relationships.thermodynamic and spec.has_temperature() and spec.has_pressure()),
        spec.has_3d_position(),
    ])
    if has_relationships:
        data_level = DataLevel.RELATED_SIGNALS

    if spec.has_velocity_field():
        data_level = DataLevel.SPATIAL_FIELD

    # Check each capability
    for cap, reqs in CAPABILITY_REQUIREMENTS.items():
        missing = check_requirements(spec, reqs)

        if not missing:
            available.add(cap)
        else:
            unavailable[cap] = missing

    return CapabilityReport(
        available=available,
        unavailable=unavailable,
        data_level=data_level,
        data_spec=spec,
    )


# =============================================================================
# CAPABILITY → ENGINE MAPPING
# =============================================================================

CAPABILITY_TO_ENGINE: Dict[Capability, str] = {
    # Level 0
    Capability.STATISTICS: 'typology.statistics',
    Capability.DISTRIBUTION: 'typology.distribution',
    Capability.STATIONARITY: 'typology.stationarity',
    Capability.ENTROPY: 'information.*',
    Capability.MEMORY: 'memory.*',
    Capability.SPECTRAL: 'frequency.*',
    Capability.RECURRENCE: 'recurrence.rqa',
    Capability.CHAOS: 'dynamics.lyapunov',
    Capability.VOLATILITY: 'volatility.*',
    Capability.EVENT_DETECTION: 'events.heaviside_dirac',
    Capability.HEAVISIDE_DIRAC: 'events.heaviside_dirac',
    Capability.GEOMETRY: 'geometry.*',
    Capability.DYNAMICS: 'dynamics.*',

    # Level 1
    Capability.DERIVATIVES: 'pointwise.derivatives',
    Capability.SPECIFIC_KINETIC: 'physics.kinetic_energy',
    Capability.SPECIFIC_POTENTIAL: 'physics.potential_energy',
    Capability.SPECIFIC_HAMILTONIAN: 'physics.hamiltonian',

    # Level 2
    Capability.KINETIC_ENERGY: 'physics.kinetic_energy',
    Capability.POTENTIAL_ENERGY: 'physics.potential_energy',
    Capability.MOMENTUM: 'physics.momentum',
    Capability.HAMILTONIAN: 'physics.hamiltonian',
    Capability.LAGRANGIAN: 'physics.lagrangian',
    Capability.ROTATIONAL_KE: 'physics.kinetic_energy',

    # Level 3
    Capability.WORK: 'physics.work_energy',
    Capability.POWER: 'physics.work_energy',
    Capability.ANGULAR_MOMENTUM: 'physics.momentum',
    Capability.PHASE_SPACE: 'physics.hamiltonian',
    Capability.ENERGY_CONSERVATION: 'physics.hamiltonian',

    Capability.GIBBS_FREE_ENERGY: 'physics.gibbs_free_energy',
    Capability.ENTHALPY: 'physics.gibbs_free_energy',
    Capability.ENTROPY_THERMO: 'physics.gibbs_free_energy',
    Capability.CHEMICAL_POTENTIAL: 'physics.gibbs_free_energy',

    Capability.TRANSFER_FUNCTION: 'systems.transfer_function',
    Capability.FREQUENCY_RESPONSE: 'systems.frequency_response',
    Capability.POLES_ZEROS: 'systems.poles_zeros',
    Capability.IMPULSE_RESPONSE: 'systems.transfer_function',

    Capability.GRANGER_CAUSALITY: 'state.granger',
    Capability.TRANSFER_ENTROPY: 'state.transfer_entropy',

    # Level 4
    Capability.VORTICITY: 'fields.navier_stokes',
    Capability.STRAIN_TENSOR: 'fields.navier_stokes',
    Capability.Q_CRITERION: 'fields.navier_stokes',
    Capability.TURBULENT_KE: 'fields.navier_stokes',
    Capability.DISSIPATION: 'fields.navier_stokes',
    Capability.ENERGY_SPECTRUM: 'fields.navier_stokes',
    Capability.REYNOLDS_NUMBER: 'fields.navier_stokes',
    Capability.KOLMOGOROV_SCALES: 'fields.navier_stokes',

    Capability.HEAT_FLUX: 'fields.heat_transfer',
    Capability.LAPLACIAN_T: 'fields.heat_transfer',

    Capability.MAXWELL_DIV_E: 'fields.electromagnetism',
    Capability.MAXWELL_CURL_B: 'fields.electromagnetism',
    Capability.POYNTING: 'fields.electromagnetism',
}


def get_engines_for_capabilities(capabilities: Set[Capability]) -> Set[str]:
    """
    Get the set of engines needed for the given capabilities.
    """
    engines = set()

    for cap in capabilities:
        engine = CAPABILITY_TO_ENGINE.get(cap)
        if engine:
            if '*' in engine:
                # Wildcard - add the module
                engines.add(engine.replace('.*', ''))
            else:
                engines.add(engine)

    return engines


# =============================================================================
# PRINT CURRICULUM
# =============================================================================

def print_curriculum():
    """Print the full PRISM capability curriculum."""
    print("""
+==============================================================================+
|                    PRISM COMPUTATIONAL PHYSICS CURRICULUM                    |
+==============================================================================+
|                                                                              |
|  You bring the data. We compute everything possible.                         |
|  You bring more context. We unlock more physics.                             |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 0: Raw Time Series                                                    |
|  ------------------------                                                    |
|  Input:  Just columns of numbers                                             |
|  Unlock: Statistics, Entropy, Memory, Spectral, Stationarity,                |
|          Recurrence, Chaos, Volatility, Event Detection, H(t), d(t)          |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 1: Labeled Signals                                                    |
|  ------------------------                                                    |
|  Input:  + Tell us what signals represent                                    |
|          physical_quantity: position, velocity, temperature, etc.            |
|  Unlock: + Derivatives with units, Specific KE (1/2 v^2), Specific PE        |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 2: Physical Constants                                                 |
|  ---------------------------                                                 |
|  Input:  + mass [kg], spring_constant [N/m], Cp [J/(mol*K)], etc.            |
|  Unlock: + REAL kinetic energy (1/2 mv^2), REAL potential (1/2 kx^2)         |
|          + Momentum (mv), Hamiltonian (T+V), Lagrangian (T-V)                |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 3: Related Signals                                                    |
|  ------------------------                                                    |
|  Input:  + Signal relationships                                              |
|          mechanical: {position: x, velocity: v, force: F}                    |
|          thermodynamic: {temperature: T, pressure: P, volume: V}             |
|          control: {input: u, output: y}                                      |
|  Unlock: + Work (integral F dx), Power (F v)                                 |
|          + Angular momentum (r x p) for 3D data                              |
|          + Gibbs free energy (H - TS)                                        |
|          + Transfer functions G(s), Bode plots, stability                    |
|                                                                              |
+==============================================================================+
|                                                                              |
|  LEVEL 4: Spatial Fields                                                     |
|  -----------------------                                                     |
|  Input:  + Velocity field v(x,y,z,t), grid spacing, viscosity                |
|  Unlock: + FULL NAVIER-STOKES                                                |
|            Vorticity (curl v), Strain tensor, Q-criterion                    |
|            Turbulent KE, Dissipation rate, Energy spectrum E(k)              |
|            Reynolds number, Kolmogorov scales                                |
|                                                                              |
+==============================================================================+
""")
