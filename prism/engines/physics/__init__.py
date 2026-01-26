"""
PRISM Physics Engines
=====================

Physics-based computation engines organized by domain:

Fluids (5):
    - navier_stokes: Incompressible flow solver
    - turbulence_spectrum: Kolmogorov energy spectrum
    - reynolds_stress: Turbulent stress tensor
    - vorticity: Curl and circulation
    - two_phase_flow: Drift-flux, void fraction

Thermal (5):
    - heat_equation: Conduction solver
    - convection: Nusselt correlations
    - radiation: Stefan-Boltzmann
    - stefan_problem: Phase change
    - heat_exchanger: LMTD, NTU-effectiveness

Thermo (5):
    - phase_equilibria: VLE, bubble/dew point
    - equation_of_state: VdW, SRK, Peng-Robinson
    - fugacity: Fugacity coefficients
    - exergy: Available work, second law
    - activity_models: NRTL, Wilson, UNIQUAC

Chemical (3):
    - reaction_kinetics: Arrhenius, rate laws, reactor design
    - separations: Distillation stages, absorption
    - electrochemistry: Butler-Volmer, Tafel kinetics

Mechanical (5):
    - modal_analysis: Natural frequencies, mode shapes
    - fatigue: S-N curves, rainflow counting
    - bearing_fault: Defect frequencies, envelope analysis
    - gear_mesh: Mesh frequencies, sidebands
    - rotor_dynamics: Critical speeds, unbalance

Electrical (3):
    - power_quality: THD, power factor, harmonics
    - motor_signature: MCSA, broken bar detection
    - impedance: EIS, equivalent circuits

Control (3):
    - transfer_function: Frequency response, step response
    - kalman: State estimation, filtering
    - stability: Routh-Hurwitz, Nyquist, Lyapunov

Process (3):
    - reactor_ode: Batch, CSTR, PFR dynamics
    - distillation: Tray-by-tray, FUG shortcut
    - crystallization: Population balance, nucleation

Total: 32 physics engines
"""

from . import fluids
from . import thermal
from . import thermo
from . import chemical
from . import mechanical
from . import electrical
from . import control
from . import process

__all__ = [
    'fluids',
    'thermal',
    'thermo',
    'chemical',
    'mechanical',
    'electrical',
    'control',
    'process',
]
