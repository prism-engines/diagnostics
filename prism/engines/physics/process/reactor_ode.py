"""
Reactor ODE Systems

Solve coupled ODEs for CSTR, batch, and PFR reactors.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from scipy import integrate


def batch_reactor_ode(y0: List[float], t_span: tuple, k: List[float],
                      stoich: np.ndarray, orders: np.ndarray,
                      n_points: int = 100) -> Dict[str, Any]:
    """
    Solve batch reactor ODEs for multiple reactions.

    dC_i/dt = Σ_j (ν_ij × r_j)

    Args:
        y0: Initial concentrations [mol/m³]
        t_span: (t_start, t_end) [s]
        k: Rate constants for each reaction
        stoich: Stoichiometric matrix (n_species × n_reactions)
        orders: Reaction orders (n_species × n_reactions)
        n_points: Number of output points

    Returns:
        t: Time points
        C: Concentration profiles (n_points × n_species)
        conversion: Conversion of each species
    """
    y0 = np.atleast_1d(y0)
    k = np.atleast_1d(k)
    stoich = np.atleast_2d(stoich)
    orders = np.atleast_2d(orders)

    n_species = len(y0)
    n_reactions = len(k)

    def rate_equations(t, C):
        C = np.maximum(C, 0)  # Prevent negative concentrations

        # Calculate rates
        r = np.zeros(n_reactions)
        for j in range(n_reactions):
            r[j] = k[j]
            for i in range(n_species):
                r[j] *= C[i] ** orders[i, j]

        # Species balances
        dCdt = stoich @ r

        return dCdt

    # Solve ODE
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = integrate.solve_ivp(
        rate_equations, t_span, y0, t_eval=t_eval, method='BDF'
    )

    # Conversions
    C_final = solution.y[:, -1]
    conversion = (y0 - C_final) / y0
    conversion = np.where(y0 > 0, conversion, 0)

    return {
        't': solution.t.tolist(),
        'C': solution.y.T.tolist(),
        'C_final': C_final.tolist(),
        'conversion': conversion.tolist(),
        'success': solution.success,
    }


def cstr_ode(y0: List[float], t_span: tuple, k: List[float],
             stoich: np.ndarray, orders: np.ndarray,
             tau: float, C_in: List[float],
             n_points: int = 100) -> Dict[str, Any]:
    """
    Solve CSTR dynamics (startup or perturbation response).

    V×dC_i/dt = F×(C_in,i - C_i) + V×Σ_j(ν_ij × r_j)

    Args:
        y0: Initial concentrations [mol/m³]
        t_span: (t_start, t_end) [s]
        k: Rate constants
        stoich: Stoichiometric matrix
        orders: Reaction orders
        tau: Residence time V/F [s]
        C_in: Inlet concentrations [mol/m³]
        n_points: Number of output points

    Returns:
        t: Time points
        C: Concentration profiles
        steady_state: Whether steady state reached
    """
    y0 = np.atleast_1d(y0)
    k = np.atleast_1d(k)
    stoich = np.atleast_2d(stoich)
    orders = np.atleast_2d(orders)
    C_in = np.atleast_1d(C_in)

    n_species = len(y0)
    n_reactions = len(k)

    def cstr_equations(t, C):
        C = np.maximum(C, 0)

        # Reaction rates
        r = np.zeros(n_reactions)
        for j in range(n_reactions):
            r[j] = k[j]
            for i in range(n_species):
                r[j] *= C[i] ** orders[i, j]

        # CSTR balance
        dCdt = (C_in - C) / tau + stoich @ r

        return dCdt

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = integrate.solve_ivp(
        cstr_equations, t_span, y0, t_eval=t_eval, method='BDF'
    )

    # Check for steady state
    if solution.success and len(solution.t) > 10:
        dC_final = np.abs(np.diff(solution.y[:, -10:], axis=1))
        steady_state = np.all(dC_final < 1e-6)
    else:
        steady_state = False

    return {
        't': solution.t.tolist(),
        'C': solution.y.T.tolist(),
        'C_final': solution.y[:, -1].tolist(),
        'steady_state': bool(steady_state),
        'tau': tau,
        'success': solution.success,
    }


def pfr_ode(C0: List[float], V_span: tuple, k: List[float],
            stoich: np.ndarray, orders: np.ndarray,
            F: float, n_points: int = 100) -> Dict[str, Any]:
    """
    Solve PFR along reactor length.

    F×dC_i/dV = Σ_j(ν_ij × r_j)

    Args:
        C0: Inlet concentrations [mol/m³]
        V_span: (V_start, V_end) reactor volume [m³]
        k: Rate constants
        stoich: Stoichiometric matrix
        orders: Reaction orders
        F: Volumetric flow rate [m³/s]
        n_points: Number of output points

    Returns:
        V: Volume points
        C: Concentration profiles
        conversion: Exit conversion
    """
    C0 = np.atleast_1d(C0)
    k = np.atleast_1d(k)
    stoich = np.atleast_2d(stoich)
    orders = np.atleast_2d(orders)

    n_species = len(C0)
    n_reactions = len(k)

    def pfr_equations(V, C):
        C = np.maximum(C, 0)

        # Reaction rates
        r = np.zeros(n_reactions)
        for j in range(n_reactions):
            r[j] = k[j]
            for i in range(n_species):
                r[j] *= C[i] ** orders[i, j]

        # PFR balance
        dCdV = stoich @ r / F

        return dCdV

    V_eval = np.linspace(V_span[0], V_span[1], n_points)
    solution = integrate.solve_ivp(
        pfr_equations, V_span, C0, t_eval=V_eval, method='BDF'
    )

    # Conversion
    C_exit = solution.y[:, -1]
    conversion = (C0 - C_exit) / C0
    conversion = np.where(C0 > 0, conversion, 0)

    # Space time
    space_time = V_span[1] / F

    return {
        'V': solution.t.tolist(),
        'C': solution.y.T.tolist(),
        'C_exit': C_exit.tolist(),
        'conversion': conversion.tolist(),
        'space_time': float(space_time),
        'success': solution.success,
    }


def nonisothermal_batch(y0: List[float], T0: float, t_span: tuple,
                        k0: float, Ea: float, delta_H: float,
                        stoich: List[float], orders: List[float],
                        rho: float, Cp: float, V: float,
                        UA: float = 0, T_coolant: float = 298,
                        n_points: int = 100) -> Dict[str, Any]:
    """
    Non-isothermal batch reactor with energy balance.

    dC/dt = ν × r
    dT/dt = (-ΔH × r × V - UA × (T - Tc)) / (ρ × Cp × V)

    Args:
        y0: Initial concentrations [mol/m³]
        T0: Initial temperature [K]
        t_span: Time span [s]
        k0: Pre-exponential factor
        Ea: Activation energy [J/mol]
        delta_H: Heat of reaction [J/mol] (negative = exothermic)
        stoich: Stoichiometric coefficients
        orders: Reaction orders
        rho: Density [kg/m³]
        Cp: Heat capacity [J/(kg·K)]
        V: Volume [m³]
        UA: Heat transfer coefficient × area [W/K]
        T_coolant: Coolant temperature [K]
        n_points: Number of output points

    Returns:
        t: Time points
        C: Concentration profiles
        T: Temperature profile
    """
    R = 8.314  # Gas constant

    y0 = np.atleast_1d(y0)
    stoich = np.atleast_1d(stoich)
    orders = np.atleast_1d(orders)

    n_species = len(y0)

    # Initial state: [C1, C2, ..., T]
    state0 = np.concatenate([y0, [T0]])

    def equations(t, state):
        C = state[:-1]
        T = state[-1]

        C = np.maximum(C, 0)

        # Temperature-dependent rate constant
        k = k0 * np.exp(-Ea / (R * T))

        # Rate
        r = k
        for i in range(n_species):
            r *= C[i] ** orders[i]

        # Mass balance
        dCdt = stoich * r

        # Energy balance
        dTdt = (-delta_H * r * V - UA * (T - T_coolant)) / (rho * Cp * V)

        return np.concatenate([dCdt, [dTdt]])

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = integrate.solve_ivp(
        equations, t_span, state0, t_eval=t_eval, method='BDF'
    )

    return {
        't': solution.t.tolist(),
        'C': solution.y[:-1, :].T.tolist(),
        'T': solution.y[-1, :].tolist(),
        'C_final': solution.y[:-1, -1].tolist(),
        'T_final': float(solution.y[-1, -1]),
        'T_max': float(np.max(solution.y[-1, :])),
        'adiabatic_rise': float(solution.y[-1, -1] - T0) if UA == 0 else None,
        'success': solution.success,
    }


def compute(reactor_type: str = 'batch', y0: List[float] = None,
            t_span: tuple = None, k: List[float] = None,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for reactor ODEs.

    Args:
        reactor_type: 'batch', 'cstr', 'pfr', or 'nonisothermal'
    """
    if y0 is None or k is None:
        return {'error': 'Provide initial concentrations (y0) and rate constants (k)'}

    # Default stoichiometry and orders for single reaction A → B
    n_species = len(y0)
    n_reactions = len(k) if isinstance(k, list) else 1

    stoich = kwargs.get('stoich', np.array([[-1], [1]][:n_species]))
    orders = kwargs.get('orders', np.array([[1], [0]][:n_species]))

    stoich = np.atleast_2d(stoich)
    orders = np.atleast_2d(orders)

    if t_span is None:
        t_span = (0, 100)

    if reactor_type == 'batch':
        return batch_reactor_ode(y0, t_span, k, stoich, orders, kwargs.get('n_points', 100))

    elif reactor_type == 'cstr':
        tau = kwargs.get('tau', 60)
        C_in = kwargs.get('C_in', y0)
        return cstr_ode(y0, t_span, k, stoich, orders, tau, C_in, kwargs.get('n_points', 100))

    elif reactor_type == 'pfr':
        V_span = kwargs.get('V_span', (0, 1))
        F = kwargs.get('F', 0.01)
        return pfr_ode(y0, V_span, k, stoich, orders, F, kwargs.get('n_points', 100))

    elif reactor_type == 'nonisothermal':
        return nonisothermal_batch(
            y0, kwargs.get('T0', 300), t_span,
            k[0] if isinstance(k, list) else k,
            kwargs.get('Ea', 50000),
            kwargs.get('delta_H', -50000),
            stoich[:, 0], orders[:, 0],
            kwargs.get('rho', 1000),
            kwargs.get('Cp', 4000),
            kwargs.get('V', 1),
            kwargs.get('UA', 0),
            kwargs.get('T_coolant', 298),
            kwargs.get('n_points', 100)
        )

    else:
        return {'error': f"Unknown reactor type: {reactor_type}"}
