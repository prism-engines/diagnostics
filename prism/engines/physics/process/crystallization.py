"""
Crystallization Process

Population balance, nucleation, growth kinetics.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import integrate


def solubility(T: float, A: float = 0.1, B: float = 0.01,
               C: float = 0) -> Dict[str, Any]:
    """
    Solubility as function of temperature.

    C_sat = A + B×T + C×T²

    Args:
        T: Temperature [°C or K]
        A, B, C: Solubility coefficients

    Returns:
        C_sat: Saturation concentration [kg/kg solvent]
    """
    C_sat = A + B * T + C * T ** 2

    return {
        'C_sat': float(C_sat),
        'temperature': T,
        'dC_dT': float(B + 2 * C * T),  # Solubility curve slope
    }


def supersaturation(C: float, C_sat: float) -> Dict[str, Any]:
    """
    Calculate supersaturation metrics.

    Args:
        C: Actual concentration [kg/kg solvent]
        C_sat: Saturation concentration [kg/kg solvent]

    Returns:
        S: Supersaturation ratio C/C_sat
        sigma: Relative supersaturation (C-C_sat)/C_sat
        delta_C: Absolute supersaturation
    """
    S = C / C_sat if C_sat > 0 else float('inf')
    sigma = (C - C_sat) / C_sat if C_sat > 0 else float('inf')
    delta_C = C - C_sat

    return {
        'S': float(S),
        'sigma': float(sigma),
        'delta_C': float(delta_C),
        'metastable': 1.0 < S < 1.5,  # Typical metastable zone
        'labile': S >= 1.5,
    }


def nucleation_rate(sigma: float, k_n: float = 1e10,
                    n: float = 2.0, T: float = 298,
                    primary: bool = True) -> Dict[str, Any]:
    """
    Nucleation rate models.

    Primary: B = k_n × exp(-A_n / ln²(S))
    Secondary: B = k_n × S^n × M_T^m

    Args:
        sigma: Relative supersaturation
        k_n: Nucleation rate constant
        n: Supersaturation exponent
        T: Temperature [K]
        primary: Use primary nucleation model

    Returns:
        B: Nucleation rate [#/(m³·s)]
    """
    S = 1 + sigma

    if primary:
        # Classical nucleation theory (simplified)
        if S > 1:
            A_n = 1000  # Nucleation parameter
            B = k_n * np.exp(-A_n / (np.log(S) ** 2))
        else:
            B = 0
    else:
        # Secondary nucleation
        if sigma > 0:
            B = k_n * sigma ** n
        else:
            B = 0

    return {
        'B': float(B),
        'S': float(S),
        'sigma': sigma,
        'nucleation_type': 'primary' if primary else 'secondary',
    }


def growth_rate(sigma: float, k_g: float = 1e-7,
                g: float = 1.0, T: float = 298) -> Dict[str, Any]:
    """
    Crystal growth rate.

    G = k_g × σ^g

    Args:
        sigma: Relative supersaturation
        k_g: Growth rate constant [m/s]
        g: Growth order
        T: Temperature [K]

    Returns:
        G: Linear growth rate [m/s]
    """
    if sigma > 0:
        G = k_g * sigma ** g
    else:
        G = 0

    return {
        'G': float(G),
        'sigma': sigma,
        'g': g,
        'mass_growth_rate': float(3 * G),  # Approximate for spheres
    }


def population_balance_moments(sigma: float, t_span: tuple,
                               k_n: float = 1e10, k_g: float = 1e-7,
                               n: float = 2.0, g: float = 1.0,
                               mu0_init: float = 0, mu1_init: float = 0,
                               mu2_init: float = 0, mu3_init: float = 0,
                               n_points: int = 100) -> Dict[str, Any]:
    """
    Solve population balance using method of moments.

    dμ_k/dt = k×G×μ_{k-1} + L_k^k × B

    μ_0 = total number
    μ_1 = total length
    μ_2 = total surface area
    μ_3 = total volume (mass)

    Args:
        sigma: Relative supersaturation (constant)
        t_span: Time span [s]
        k_n, k_g: Kinetic constants
        n, g: Kinetic orders
        mu_init: Initial moments
        n_points: Output points

    Returns:
        t: Time points
        mu: Moment profiles [μ0, μ1, μ2, μ3]
        mean_size: Mean crystal size
        cv: Coefficient of variation
    """
    # Growth rate
    G = k_g * sigma ** g if sigma > 0 else 0

    # Nucleation rate
    B = k_n * sigma ** n if sigma > 0 else 0

    # Nuclei size (characteristic)
    L0 = 1e-6  # 1 μm nuclei

    def moment_equations(t, mu):
        mu0, mu1, mu2, mu3 = mu

        dmu0_dt = B  # Number rate = nucleation
        dmu1_dt = G * mu0 + L0 * B
        dmu2_dt = 2 * G * mu1 + L0 ** 2 * B
        dmu3_dt = 3 * G * mu2 + L0 ** 3 * B

        return [dmu0_dt, dmu1_dt, dmu2_dt, dmu3_dt]

    mu_init = [mu0_init, mu1_init, mu2_init, mu3_init]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = integrate.solve_ivp(
        moment_equations, t_span, mu_init, t_eval=t_eval, method='RK45'
    )

    mu = solution.y

    # Derived quantities
    mu0, mu1, mu2, mu3 = mu

    # Mean size L43 (volume-weighted)
    L43 = np.where(mu2 > 0, mu3 / mu2, 0)

    # Mean size L10 (number-weighted)
    L10 = np.where(mu0 > 0, mu1 / mu0, 0)

    # Coefficient of variation
    L20 = np.where(mu0 > 0, mu2 / mu0, 0)
    variance = L20 - L10 ** 2
    cv = np.where(L10 > 0, np.sqrt(np.maximum(variance, 0)) / L10, 0)

    return {
        't': solution.t.tolist(),
        'mu0': mu0.tolist(),
        'mu1': mu1.tolist(),
        'mu2': mu2.tolist(),
        'mu3': mu3.tolist(),
        'mean_size_L43': L43.tolist(),
        'mean_size_L10': L10.tolist(),
        'cv': cv.tolist(),
        'total_mass': (mu3 * 1500).tolist(),  # Assuming density 1500 kg/m³
        'G': float(G),
        'B': float(B),
    }


def batch_cooling_crystallization(C0: float, T0: float, T_final: float,
                                   cooling_rate: float, V: float,
                                   k_n: float = 1e10, k_g: float = 1e-7,
                                   n: float = 2.0, g: float = 1.0,
                                   sol_A: float = 0.1, sol_B: float = 0.01,
                                   n_points: int = 100) -> Dict[str, Any]:
    """
    Batch cooling crystallization simulation.

    Args:
        C0: Initial concentration [kg/kg]
        T0: Initial temperature [°C]
        T_final: Final temperature [°C]
        cooling_rate: Cooling rate [°C/min]
        V: Volume [m³]
        k_n, k_g: Kinetic constants
        n, g: Kinetic orders
        sol_A, sol_B: Solubility coefficients

    Returns:
        t: Time points
        T: Temperature profile
        C: Concentration profile
        M_c: Crystal mass profile
    """
    # Time span
    t_cool = abs(T0 - T_final) / cooling_rate * 60  # seconds
    t_span = (0, t_cool)

    # Crystal density
    rho_c = 1500  # kg/m³
    k_v = np.pi / 6  # Volume shape factor (spheres)

    def equations(t, state):
        C, mu0, mu1, mu2, mu3 = state

        # Current temperature (linear cooling)
        T = T0 - cooling_rate / 60 * t

        # Solubility
        C_sat = sol_A + sol_B * T

        # Supersaturation
        sigma = max(0, (C - C_sat) / C_sat)

        # Kinetics
        G = k_g * sigma ** g if sigma > 0 else 0
        B = k_n * sigma ** n if sigma > 0 else 0
        L0 = 1e-6

        # Population balance moments
        dmu0_dt = B
        dmu1_dt = G * mu0 + L0 * B
        dmu2_dt = 2 * G * mu1 + L0 ** 2 * B
        dmu3_dt = 3 * G * mu2 + L0 ** 3 * B

        # Mass balance: dC/dt = -3×ρ_c×k_v×G×μ2/V
        dC_dt = -3 * rho_c * k_v * G * mu2 / V if mu2 > 0 else 0

        return [dC_dt, dmu0_dt, dmu1_dt, dmu2_dt, dmu3_dt]

    state0 = [C0, 0, 0, 0, 0]

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    solution = integrate.solve_ivp(
        equations, t_span, state0, t_eval=t_eval, method='BDF'
    )

    C = solution.y[0]
    mu0, mu1, mu2, mu3 = solution.y[1:5]

    # Temperature profile
    T = T0 - cooling_rate / 60 * solution.t

    # Crystal mass
    M_c = rho_c * k_v * mu3 * V

    # Yield
    C_sat_final = sol_A + sol_B * T_final
    yield_theoretical = (C0 - C_sat_final) / C0 * 100
    yield_actual = (C0 - C[-1]) / C0 * 100

    # Mean size
    L43 = mu3 / mu2 if mu2[-1] > 0 else 0

    return {
        't': solution.t.tolist(),
        'T': T.tolist(),
        'C': C.tolist(),
        'M_c': M_c.tolist(),
        'mu0': mu0.tolist(),
        'mu3': mu3.tolist(),
        'final_mass': float(M_c[-1]),
        'final_mean_size': float(L43[-1] if hasattr(L43, '__len__') else L43) * 1e6,  # μm
        'yield_theoretical': float(yield_theoretical),
        'yield_actual': float(yield_actual),
        'success': solution.success,
    }


def compute(mode: str = 'supersaturation', **kwargs) -> Dict[str, Any]:
    """
    Main entry point for crystallization calculations.

    Args:
        mode: 'supersaturation', 'nucleation', 'growth', 'moments', 'batch'
    """
    if mode == 'supersaturation':
        C = kwargs.get('C', 0.15)
        C_sat = kwargs.get('C_sat')
        if C_sat is None:
            T = kwargs.get('T', 25)
            sol_result = solubility(T, kwargs.get('A', 0.1), kwargs.get('B', 0.01))
            C_sat = sol_result['C_sat']
        return supersaturation(C, C_sat)

    elif mode == 'solubility':
        return solubility(kwargs.get('T', 25), kwargs.get('A', 0.1),
                         kwargs.get('B', 0.01), kwargs.get('C', 0))

    elif mode == 'nucleation':
        return nucleation_rate(kwargs.get('sigma', 0.1),
                              kwargs.get('k_n', 1e10),
                              kwargs.get('n', 2.0),
                              kwargs.get('T', 298),
                              kwargs.get('primary', True))

    elif mode == 'growth':
        return growth_rate(kwargs.get('sigma', 0.1),
                          kwargs.get('k_g', 1e-7),
                          kwargs.get('g', 1.0),
                          kwargs.get('T', 298))

    elif mode == 'moments':
        return population_balance_moments(
            kwargs.get('sigma', 0.1),
            kwargs.get('t_span', (0, 3600)),
            kwargs.get('k_n', 1e10),
            kwargs.get('k_g', 1e-7),
            kwargs.get('n', 2.0),
            kwargs.get('g', 1.0),
        )

    elif mode == 'batch':
        return batch_cooling_crystallization(
            kwargs.get('C0', 0.2),
            kwargs.get('T0', 60),
            kwargs.get('T_final', 20),
            kwargs.get('cooling_rate', 0.5),
            kwargs.get('V', 0.01),
            kwargs.get('k_n', 1e10),
            kwargs.get('k_g', 1e-7),
        )

    return {'error': f"Unknown mode: {mode}"}
