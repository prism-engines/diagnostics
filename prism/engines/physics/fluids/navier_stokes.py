"""
Navier-Stokes Field Analysis Engine

THE REAL EQUATIONS. NOT INSPIRED BY. THE REAL THING.

Implements:
    dv/dt + (v . nabla)v = -nabla(p)/rho + nu * nabla^2(v) + f

    Vorticity:           omega = curl(v)
    Strain rate tensor:  S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)
    Rotation tensor:     Omega_ij = 0.5 * (dv_i/dx_j - dv_j/dx_i)
    Q-criterion:         Q = 0.5 * (||Omega||^2 - ||S||^2)
    lambda_2 criterion:  Second eigenvalue of S^2 + Omega^2
    TKE:                 k = 0.5 * <u'_i * u'_i>
    Dissipation:         epsilon = 2 * nu * <S_ij * S_ij>
    Enstrophy:           0.5 * <omega_i * omega_i>
    Helicity:            H = v . omega
    Reynolds stress:     tau_ij = -rho * <u'_i * u'_j>
    Energy spectrum:     E(k) with Kolmogorov k^(-5/3) validation

References:
    Pope, S.B. (2000) "Turbulent Flows" - Cambridge University Press
    Tennekes & Lumley (1972) "A First Course in Turbulence"
    Davidson, P.A. (2015) "Turbulence: An Introduction for Scientists and Engineers"
    Kolmogorov (1941) "The local structure of turbulence in incompressible viscous fluid"
    Hunt, Wray & Moin (1988) "Eddies, streams, and convergence zones" CTR Report S88
    Jeong & Hussain (1995) "On the identification of a vortex" J. Fluid Mech. 285
"""

import numpy as np
from numpy.fft import fftn, fftfreq
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FlowRegime(Enum):
    """Flow regime classification based on Reynolds number."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


class MissingFieldConstantError(Exception):
    """Raised when required field constant is not provided."""
    pass


@dataclass
class VelocityField:
    """
    3D velocity field container.

    REQUIRES: nu [m²/s], rho [kg/m³], dx/dy/dz [m]

    Attributes:
        u: x-component of velocity, shape (nx, ny, nz) or (nx, ny, nz, nt)
        v: y-component of velocity
        w: z-component of velocity
        dx, dy, dz: Grid spacing in each direction [m]. REQUIRED.
        dt: Time step (if time-varying) [s]
        nu: Kinematic viscosity [m^2/s]. REQUIRED - no default.
        rho: Density [kg/m^3]. REQUIRED - no default.
    """
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    dx: float
    dy: float
    dz: float
    dt: float = 1.0

    # NO DEFAULTS - must be explicitly provided
    nu: float = None      # Kinematic viscosity [m²/s]
    rho: float = None     # Density [kg/m³]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.u.shape

    @property
    def is_time_varying(self) -> bool:
        return len(self.u.shape) == 4

    @property
    def velocity_magnitude(self) -> np.ndarray:
        """||v|| = sqrt(u^2 + v^2 + w^2)"""
        return np.sqrt(self.u**2 + self.v**2 + self.w**2)

    def validate(self) -> None:
        """Validate field data and REQUIRED constants."""
        # VALIDATION: nu MUST be provided
        if self.nu is None:
            raise MissingFieldConstantError(
                "Missing required constant: nu (kinematic viscosity) [m²/s]. "
                "Common values: water=1e-6, air=1.5e-5, oil=1e-4"
            )

        # VALIDATION: rho MUST be provided
        if self.rho is None:
            raise MissingFieldConstantError(
                "Missing required constant: rho (density) [kg/m³]. "
                "Common values: water=1000, air=1.2, oil=900"
            )

        if self.u.shape != self.v.shape or self.u.shape != self.w.shape:
            raise ValueError("Velocity components must have same shape")
        if len(self.u.shape) not in [3, 4]:
            raise ValueError("Field must be 3D (nx,ny,nz) or 4D (nx,ny,nz,nt)")
        if self.dx <= 0 or self.dy <= 0 or self.dz <= 0:
            raise ValueError("Grid spacing must be positive")
        if self.nu <= 0:
            raise ValueError("Viscosity must be positive")
        if self.rho <= 0:
            raise ValueError("Density must be positive")


# =============================================================================
# SPATIAL DERIVATIVES
# =============================================================================

def gradient(f: np.ndarray, dx: float, axis: int, method: str = 'central') -> np.ndarray:
    """
    Compute spatial gradient using finite differences.

    Args:
        f: Field array
        dx: Grid spacing
        axis: Axis along which to differentiate
        method: 'central' (2nd order), 'spectral' (FFT-based)

    Returns:
        df/dx along specified axis
    """
    if method == 'central':
        # 2nd-order central difference
        return np.gradient(f, dx, axis=axis)

    elif method == 'spectral':
        # Spectral derivative (more accurate for periodic domains)
        n = f.shape[axis]
        k = fftfreq(n, dx) * 2 * np.pi

        # Reshape k for broadcasting
        shape = [1] * f.ndim
        shape[axis] = n
        k = k.reshape(shape)

        f_hat = fftn(f, axes=[axis])
        df_hat = 1j * k * f_hat
        return np.real(np.fft.ifftn(df_hat, axes=[axis]))

    else:
        raise ValueError(f"Unknown method: {method}")


def laplacian(f: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute Laplacian: nabla^2(f) = d^2f/dx^2 + d^2f/dy^2 + d^2f/dz^2

    Uses 2nd-order central differences.
    """
    d2f_dx2 = np.gradient(np.gradient(f, dx, axis=0), dx, axis=0)
    d2f_dy2 = np.gradient(np.gradient(f, dy, axis=1), dy, axis=1)
    d2f_dz2 = np.gradient(np.gradient(f, dz, axis=2), dz, axis=2)

    return d2f_dx2 + d2f_dy2 + d2f_dz2


# =============================================================================
# VORTICITY
# =============================================================================

def compute_vorticity(field: VelocityField) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute vorticity vector: omega = curl(v)

    omega_x = dw/dy - dv/dz
    omega_y = du/dz - dw/dx
    omega_z = dv/dx - du/dy

    Physical meaning:
        - Local rotation rate of fluid element
        - Non-zero vorticity indicates rotational flow
        - Units: [1/s]

    Returns:
        Tuple of (omega_x, omega_y, omega_z) arrays
    """
    dudx = gradient(field.u, field.dx, axis=0)
    dudy = gradient(field.u, field.dy, axis=1)
    dudz = gradient(field.u, field.dz, axis=2)

    dvdx = gradient(field.v, field.dx, axis=0)
    dvdy = gradient(field.v, field.dy, axis=1)
    dvdz = gradient(field.v, field.dz, axis=2)

    dwdx = gradient(field.w, field.dx, axis=0)
    dwdy = gradient(field.w, field.dy, axis=1)
    dwdz = gradient(field.w, field.dz, axis=2)

    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy

    return omega_x, omega_y, omega_z


def compute_vorticity_magnitude(field: VelocityField) -> np.ndarray:
    """||omega|| = sqrt(omega_x^2 + omega_y^2 + omega_z^2)"""
    omega_x, omega_y, omega_z = compute_vorticity(field)
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


# =============================================================================
# STRAIN RATE & ROTATION TENSORS
# =============================================================================

def compute_velocity_gradient_tensor(field: VelocityField) -> np.ndarray:
    """
    Compute velocity gradient tensor: A_ij = dv_i/dx_j

    Returns:
        Array of shape (*field.shape, 3, 3)
    """
    dudx = gradient(field.u, field.dx, axis=0)
    dudy = gradient(field.u, field.dy, axis=1)
    dudz = gradient(field.u, field.dz, axis=2)

    dvdx = gradient(field.v, field.dx, axis=0)
    dvdy = gradient(field.v, field.dy, axis=1)
    dvdz = gradient(field.v, field.dz, axis=2)

    dwdx = gradient(field.w, field.dx, axis=0)
    dwdy = gradient(field.w, field.dy, axis=1)
    dwdz = gradient(field.w, field.dz, axis=2)

    shape = field.u.shape
    A = np.zeros((*shape, 3, 3))

    A[..., 0, 0] = dudx
    A[..., 0, 1] = dudy
    A[..., 0, 2] = dudz

    A[..., 1, 0] = dvdx
    A[..., 1, 1] = dvdy
    A[..., 1, 2] = dvdz

    A[..., 2, 0] = dwdx
    A[..., 2, 1] = dwdy
    A[..., 2, 2] = dwdz

    return A


def compute_strain_rate_tensor(field: VelocityField) -> np.ndarray:
    """
    Compute strain rate tensor: S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)

    Physical meaning:
        - Symmetric part of velocity gradient
        - Rate of deformation of fluid element
        - Diagonal: normal strain (stretching/compression)
        - Off-diagonal: shear strain

    Returns:
        Array of shape (*field.shape, 3, 3)
    """
    A = compute_velocity_gradient_tensor(field)
    S = 0.5 * (A + np.swapaxes(A, -2, -1))
    return S


def compute_rotation_tensor(field: VelocityField) -> np.ndarray:
    """
    Compute rotation tensor: Omega_ij = 0.5 * (dv_i/dx_j - dv_j/dx_i)

    Physical meaning:
        - Antisymmetric part of velocity gradient
        - Related to vorticity: omega_k = -epsilon_ijk * Omega_ij

    Returns:
        Array of shape (*field.shape, 3, 3)
    """
    A = compute_velocity_gradient_tensor(field)
    Omega = 0.5 * (A - np.swapaxes(A, -2, -1))
    return Omega


# =============================================================================
# VORTEX IDENTIFICATION CRITERIA
# =============================================================================

def compute_Q_criterion(field: VelocityField) -> np.ndarray:
    """
    Compute Q-criterion for vortex identification.

    Q = 0.5 * (||Omega||^2 - ||S||^2)

    Physical meaning:
        Q > 0: Rotation dominates -> vortex core
        Q < 0: Strain dominates -> shear region
        Q = 0: Balance between rotation and strain

    Reference:
        Hunt, Wray & Moin (1988) "Eddies, streams, and convergence zones
        in turbulent flows" CTR Report S88

    Returns:
        Q field, same shape as velocity components
    """
    S = compute_strain_rate_tensor(field)
    Omega = compute_rotation_tensor(field)

    # Frobenius norms: ||A||^2 = tr(A^T * A) = A_ij * A_ij
    S_norm_sq = np.sum(S**2, axis=(-2, -1))
    Omega_norm_sq = np.sum(Omega**2, axis=(-2, -1))

    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    return Q


def compute_lambda2_criterion(field: VelocityField) -> np.ndarray:
    """
    Compute lambda_2 criterion for vortex identification.

    lambda_2 = second eigenvalue of (S^2 + Omega^2)

    Physical meaning:
        lambda_2 < 0: Pressure minimum due to rotation -> vortex core

    Reference:
        Jeong & Hussain (1995) "On the identification of a vortex"
        J. Fluid Mech. 285, 69-94

    Note: More expensive than Q-criterion (requires eigenvalue computation)

    Returns:
        lambda_2 field, same shape as velocity components
    """
    S = compute_strain_rate_tensor(field)
    Omega = compute_rotation_tensor(field)

    # S^2 + Omega^2
    S2 = np.einsum('...ij,...jk->...ik', S, S)
    Omega2 = np.einsum('...ij,...jk->...ik', Omega, Omega)
    M = S2 + Omega2

    # Eigenvalues at each point
    # Note: This is computationally expensive for large fields
    shape = field.u.shape
    lambda2 = np.zeros(shape)

    for idx in np.ndindex(shape):
        eigenvalues = np.linalg.eigvalsh(M[idx])
        eigenvalues.sort()
        lambda2[idx] = eigenvalues[1]  # Second eigenvalue

    return lambda2


def compute_delta_criterion(field: VelocityField) -> np.ndarray:
    """
    Compute Delta criterion for vortex identification.

    Delta = (Q/3)^3 + (R/2)^2

    where Q and R are second and third invariants of velocity gradient.

    Physical meaning:
        Delta > 0: Complex eigenvalues -> swirling motion -> vortex

    Reference:
        Chong, Perry & Cantwell (1990)

    Returns:
        Delta field
    """
    A = compute_velocity_gradient_tensor(field)

    # Invariants of velocity gradient tensor
    # P = -tr(A) = 0 for incompressible
    # Q = 0.5 * (tr(A)^2 - tr(A^2))
    # R = -det(A)

    trace_A = np.trace(A, axis1=-2, axis2=-1)
    A2 = np.einsum('...ij,...jk->...ik', A, A)
    trace_A2 = np.trace(A2, axis1=-2, axis2=-1)

    Q = 0.5 * (trace_A**2 - trace_A2)
    R = -np.linalg.det(A)

    Delta = (Q/3)**3 + (R/2)**2

    return Delta


# =============================================================================
# TURBULENT KINETIC ENERGY
# =============================================================================

def reynolds_decomposition(
    field: VelocityField
) -> Tuple[VelocityField, VelocityField]:
    """
    Reynolds decomposition: v = <v> + v'

    Decomposes velocity into mean and fluctuating components.
    Requires time-varying field (4D array).

    Returns:
        (mean_field, fluctuation_field)
    """
    if not field.is_time_varying:
        raise ValueError("Reynolds decomposition requires time-varying field")

    # Time average (over last axis)
    u_mean = np.mean(field.u, axis=-1, keepdims=True)
    v_mean = np.mean(field.v, axis=-1, keepdims=True)
    w_mean = np.mean(field.w, axis=-1, keepdims=True)

    # Broadcast mean to full shape
    u_mean_full = np.broadcast_to(u_mean, field.u.shape)
    v_mean_full = np.broadcast_to(v_mean, field.v.shape)
    w_mean_full = np.broadcast_to(w_mean, field.w.shape)

    mean_field = VelocityField(
        u=u_mean.squeeze(-1),
        v=v_mean.squeeze(-1),
        w=w_mean.squeeze(-1),
        dx=field.dx, dy=field.dy, dz=field.dz,
        nu=field.nu, rho=field.rho
    )

    fluctuation_field = VelocityField(
        u=field.u - u_mean_full,
        v=field.v - v_mean_full,
        w=field.w - w_mean_full,
        dx=field.dx, dy=field.dy, dz=field.dz, dt=field.dt,
        nu=field.nu, rho=field.rho
    )

    return mean_field, fluctuation_field


def compute_turbulent_kinetic_energy(field: VelocityField) -> np.ndarray:
    """
    Compute turbulent kinetic energy: k = 0.5 * <u'_i * u'_i>

    k = 0.5 * (<u'^2> + <v'^2> + <w'^2>)

    Physical meaning:
        - Kinetic energy in turbulent fluctuations
        - Units: [m^2/s^2] = [J/kg]

    Returns:
        TKE field (spatial distribution of turbulent energy)
    """
    if field.is_time_varying:
        _, fluct = reynolds_decomposition(field)

        # Time-averaged squared fluctuations
        u_prime_sq = np.mean(fluct.u**2, axis=-1)
        v_prime_sq = np.mean(fluct.v**2, axis=-1)
        w_prime_sq = np.mean(fluct.w**2, axis=-1)
    else:
        # For single snapshot, use spatial fluctuations (less accurate)
        u_mean = np.mean(field.u)
        v_mean = np.mean(field.v)
        w_mean = np.mean(field.w)

        u_prime_sq = (field.u - u_mean)**2
        v_prime_sq = (field.v - v_mean)**2
        w_prime_sq = (field.w - w_mean)**2

    tke = 0.5 * (u_prime_sq + v_prime_sq + w_prime_sq)

    return tke


# =============================================================================
# DISSIPATION RATE
# =============================================================================

def compute_dissipation_rate(field: VelocityField) -> np.ndarray:
    """
    Compute turbulent dissipation rate: epsilon = 2 * nu * <S'_ij * S'_ij>

    For instantaneous field: epsilon = 2 * nu * S_ij * S_ij

    Physical meaning:
        - Rate at which turbulent kinetic energy is converted to heat
        - Units: [m^2/s^3] = [W/kg]

    Returns:
        Dissipation rate field
    """
    S = compute_strain_rate_tensor(field)

    # S_ij * S_ij = ||S||^2 (Frobenius norm squared)
    S_norm_sq = np.sum(S**2, axis=(-2, -1))

    epsilon = 2 * field.nu * S_norm_sq

    return epsilon


# =============================================================================
# ENSTROPHY
# =============================================================================

def compute_enstrophy(field: VelocityField) -> np.ndarray:
    """
    Compute enstrophy: xi = 0.5 * omega_i * omega_i = 0.5 * ||omega||^2

    Physical meaning:
        - Intensity of vorticity
        - Related to dissipation in 2D: epsilon = nu * xi (exact in 2D)
        - Units: [1/s^2]

    Returns:
        Enstrophy field
    """
    omega_x, omega_y, omega_z = compute_vorticity(field)
    enstrophy = 0.5 * (omega_x**2 + omega_y**2 + omega_z**2)

    return enstrophy


# =============================================================================
# HELICITY
# =============================================================================

def compute_helicity(field: VelocityField) -> np.ndarray:
    """
    Compute helicity: H = v . omega

    Physical meaning:
        - Measure of "corkscrew" or "helical" motion
        - H != 0 indicates 3D vortical structures
        - H > 0: Right-handed helix
        - H < 0: Left-handed helix
        - Conserved in inviscid, barotropic flow
        - Units: [m/s^2]

    Reference:
        Moffatt (1969) "The degree of knottedness of tangled vortex lines"

    Returns:
        Helicity field
    """
    omega_x, omega_y, omega_z = compute_vorticity(field)
    helicity = field.u * omega_x + field.v * omega_y + field.w * omega_z

    return helicity


def compute_helicity_density_normalized(field: VelocityField) -> np.ndarray:
    """
    Compute normalized helicity density: h = v . omega / (|v| * |omega|)

    Range: [-1, 1]
        h = +1: Velocity parallel to vorticity (right helix)
        h = -1: Velocity antiparallel to vorticity (left helix)
        h = 0: Velocity perpendicular to vorticity
    """
    omega_x, omega_y, omega_z = compute_vorticity(field)

    v_mag = field.velocity_magnitude
    omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    helicity = field.u * omega_x + field.v * omega_y + field.w * omega_z

    # Avoid division by zero
    denominator = v_mag * omega_mag + 1e-10
    h_normalized = helicity / denominator

    return h_normalized


# =============================================================================
# REYNOLDS STRESS TENSOR
# =============================================================================

def compute_reynolds_stress_tensor(field: VelocityField) -> np.ndarray:
    """
    Compute Reynolds stress tensor: tau_ij = -rho * <u'_i * u'_j>

    Physical meaning:
        - Apparent stress due to turbulent momentum transport
        - Appears in Reynolds-averaged Navier-Stokes (RANS) equations
        - Symmetric tensor (6 independent components)
        - Units: [Pa] = [kg/(m * s^2)]

    Returns:
        Reynolds stress tensor, shape (*spatial_shape, 3, 3)
    """
    if not field.is_time_varying:
        raise ValueError("Reynolds stress requires time-varying field")

    _, fluct = reynolds_decomposition(field)

    shape = field.u.shape[:-1]  # Spatial dimensions
    R = np.zeros((*shape, 3, 3))

    # tau_ij = -rho * <u'_i * u'_j> (time average)
    R[..., 0, 0] = -field.rho * np.mean(fluct.u * fluct.u, axis=-1)
    R[..., 1, 1] = -field.rho * np.mean(fluct.v * fluct.v, axis=-1)
    R[..., 2, 2] = -field.rho * np.mean(fluct.w * fluct.w, axis=-1)

    R[..., 0, 1] = R[..., 1, 0] = -field.rho * np.mean(fluct.u * fluct.v, axis=-1)
    R[..., 0, 2] = R[..., 2, 0] = -field.rho * np.mean(fluct.u * fluct.w, axis=-1)
    R[..., 1, 2] = R[..., 2, 1] = -field.rho * np.mean(fluct.v * fluct.w, axis=-1)

    return R


# =============================================================================
# ENERGY SPECTRUM
# =============================================================================

def compute_energy_spectrum(
    field: VelocityField,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy spectrum E(k) in wavenumber space.

    E(k) = energy content at wavenumber k

    For isotropic turbulence in inertial range:
        E(k) ~ k^(-5/3) (Kolmogorov 1941)

    Physical meaning:
        - Distribution of kinetic energy across length scales
        - Large k -> small scales
        - Small k -> large scales

    Returns:
        (k_centers, E_k) - wavenumber bins and energy spectrum
    """
    # Use single time snapshot if time-varying
    if field.is_time_varying:
        u = field.u[..., 0]
        v = field.v[..., 0]
        w = field.w[..., 0]
    else:
        u = field.u
        v = field.v
        w = field.w

    nx, ny, nz = u.shape

    # FFT of velocity components
    u_hat = fftn(u)
    v_hat = fftn(v)
    w_hat = fftn(w)

    # Energy in Fourier space: 0.5 * |v_hat|^2
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

    # Wavenumber magnitudes
    kx = fftfreq(nx, field.dx) * 2 * np.pi
    ky = fftfreq(ny, field.dy) * 2 * np.pi
    kz = fftfreq(nz, field.dz) * 2 * np.pi

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Bin energy by wavenumber magnitude (shell integration)
    k_max = K.max()
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_k = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if np.any(mask):
            E_k[i] = np.sum(E_hat[mask])

    return k_centers, E_k


def compute_kolmogorov_slope(k: np.ndarray, E_k: np.ndarray) -> Optional[float]:
    """
    Fit power law to inertial range and return slope.

    Expected: slope ~ -5/3 ~ -1.667 for Kolmogorov turbulence

    Returns:
        Fitted slope (should be close to -5/3 for developed turbulence)
        Returns None if insufficient data
    """
    # Remove zeros
    valid = (k > 0) & (E_k > 0)
    k_valid = k[valid]
    E_valid = E_k[valid]

    if len(k_valid) < 10:
        return None

    # Use middle portion as inertial range estimate
    n = len(k_valid)
    inertial_start = n // 4
    inertial_end = 3 * n // 4

    k_inertial = k_valid[inertial_start:inertial_end]
    E_inertial = E_valid[inertial_start:inertial_end]

    if len(k_inertial) < 5:
        return None

    # Log-log fit
    log_k = np.log10(k_inertial)
    log_E = np.log10(E_inertial)

    slope, _ = np.polyfit(log_k, log_E, 1)

    return slope


# =============================================================================
# KOLMOGOROV SCALES
# =============================================================================

def compute_kolmogorov_scales(field: VelocityField, epsilon: np.ndarray = None) -> Dict:
    """
    Compute Kolmogorov microscales.

    eta = (nu^3 / epsilon)^(1/4)  - length scale
    tau_eta = (nu / epsilon)^(1/2) - time scale
    v_eta = (nu * epsilon)^(1/4)  - velocity scale

    Physical meaning:
        - Smallest scales of turbulent motion
        - Where viscous dissipation occurs
        - Universal for all turbulent flows at high Re

    Returns:
        Dictionary with Kolmogorov scales
    """
    if epsilon is None:
        epsilon = compute_dissipation_rate(field)

    nu = field.nu
    eps_mean = np.mean(epsilon)

    # Avoid division by zero
    eps_mean = max(eps_mean, 1e-20)

    eta = (nu**3 / eps_mean)**0.25      # Length scale
    tau_eta = (nu / eps_mean)**0.5       # Time scale
    v_eta = (nu * eps_mean)**0.25        # Velocity scale

    return {
        'kolmogorov_length': eta,
        'kolmogorov_time': tau_eta,
        'kolmogorov_velocity': v_eta,
    }


def compute_taylor_microscale(field: VelocityField) -> float:
    """
    Compute Taylor microscale: lambda = sqrt(15 * nu * <u^2> / epsilon)

    Physical meaning:
        - Intermediate scale between integral and Kolmogorov scales
        - Related to strain rate fluctuations

    Returns:
        Taylor microscale lambda
    """
    epsilon = compute_dissipation_rate(field)
    eps_mean = np.mean(epsilon)

    u_rms_sq = np.mean(field.u**2)

    lambda_taylor = np.sqrt(15 * field.nu * u_rms_sq / (eps_mean + 1e-20))

    return lambda_taylor


def compute_integral_length_scale(field: VelocityField) -> float:
    """
    Compute integral length scale from autocorrelation.

    L = integral(R(r) dr, 0, infinity)

    where R(r) is the spatial autocorrelation function.

    Physical meaning:
        - Characteristic size of energy-containing eddies
        - Largest scale of turbulent motion

    Returns:
        Integral length scale L
    """
    # Use first velocity component along x
    u = field.u if not field.is_time_varying else field.u[..., 0]

    # Compute autocorrelation along x
    u_fluct = u - np.mean(u)
    u_var = np.var(u)

    if u_var < 1e-20:
        return 0.0

    nx = u.shape[0]
    R = np.zeros(nx // 2)

    for r in range(nx // 2):
        if r == 0:
            R[r] = 1.0
        else:
            R[r] = np.mean(u_fluct[:-r] * u_fluct[r:]) / u_var

    # Integrate to first zero crossing
    r_vals = np.arange(len(R)) * field.dx

    # Find first zero crossing
    zero_idx = np.where(R < 0)[0]
    if len(zero_idx) > 0:
        R = R[:zero_idx[0]]
        r_vals = r_vals[:zero_idx[0]]

    L = np.trapz(R, r_vals)

    return L


# =============================================================================
# REYNOLDS NUMBER
# =============================================================================

def compute_reynolds_number(
    field: VelocityField,
    length_scale: float = None
) -> float:
    """
    Compute Reynolds number: Re = U * L / nu

    Args:
        field: Velocity field
        length_scale: Characteristic length (if None, uses domain size)

    Returns:
        Reynolds number
    """
    U = np.mean(field.velocity_magnitude)

    if length_scale is None:
        length_scale = max(
            field.u.shape[0] * field.dx,
            field.u.shape[1] * field.dy,
            field.u.shape[2] * field.dz
        )

    Re = U * length_scale / field.nu

    return Re


def compute_taylor_reynolds_number(field: VelocityField) -> float:
    """
    Compute Taylor Reynolds number: Re_lambda = u' * lambda / nu

    Physical meaning:
        - Reynolds number based on Taylor microscale
        - Common measure of turbulence intensity
        - Re_lambda > 100 typically indicates fully developed turbulence

    Returns:
        Taylor Reynolds number
    """
    u_rms = np.std(field.u)
    lambda_taylor = compute_taylor_microscale(field)

    Re_lambda = u_rms * lambda_taylor / field.nu

    return Re_lambda


# =============================================================================
# FLOW REGIME CLASSIFICATION
# =============================================================================

def classify_flow_regime(Re: float) -> FlowRegime:
    """
    Classify flow regime based on Reynolds number.

    Thresholds are approximate and geometry-dependent.
    These are typical values for pipe/channel flow.
    """
    if Re < 2300:
        return FlowRegime.LAMINAR
    elif Re < 4000:
        return FlowRegime.TRANSITIONAL
    else:
        return FlowRegime.TURBULENT


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_velocity_field(field: VelocityField) -> Dict:
    """
    Comprehensive Navier-Stokes analysis of velocity field.

    Returns dictionary with all computed quantities.
    """
    field.validate()

    # Core quantities
    vorticity = compute_vorticity(field)
    vorticity_mag = compute_vorticity_magnitude(field)
    Q = compute_Q_criterion(field)
    epsilon = compute_dissipation_rate(field)
    enstrophy = compute_enstrophy(field)
    helicity = compute_helicity(field)

    # TKE (different methods depending on time data)
    tke = compute_turbulent_kinetic_energy(field)

    # Scales
    kolmogorov = compute_kolmogorov_scales(field, epsilon)
    taylor_microscale = compute_taylor_microscale(field)
    integral_scale = compute_integral_length_scale(field)

    # Reynolds numbers
    Re = compute_reynolds_number(field)
    Re_lambda = compute_taylor_reynolds_number(field)
    regime = classify_flow_regime(Re)

    # Energy spectrum
    k, E_k = compute_energy_spectrum(field)
    spectrum_slope = compute_kolmogorov_slope(k, E_k)

    # Vortex identification
    vortex_volume_fraction = np.mean(Q > 0)

    # Statistics
    mean_velocity = np.mean(field.velocity_magnitude)
    turbulence_intensity = np.std(field.velocity_magnitude) / (mean_velocity + 1e-10)

    return {
        # Scalars
        'reynolds_number': Re,
        'taylor_reynolds_number': Re_lambda,
        'flow_regime': regime.value,
        'turbulence_intensity': turbulence_intensity,

        'mean_tke': np.mean(tke),
        'mean_dissipation': np.mean(epsilon),
        'mean_enstrophy': np.mean(enstrophy),
        'mean_helicity': np.mean(helicity),
        'mean_Q': np.mean(Q),
        'mean_vorticity': np.mean(vorticity_mag),

        'vortex_volume_fraction': vortex_volume_fraction,

        'kolmogorov_length': kolmogorov['kolmogorov_length'],
        'kolmogorov_time': kolmogorov['kolmogorov_time'],
        'kolmogorov_velocity': kolmogorov['kolmogorov_velocity'],
        'taylor_microscale': taylor_microscale,
        'integral_length_scale': integral_scale,

        'energy_spectrum_slope': spectrum_slope,
        'kolmogorov_slope_expected': -5/3,
        'is_kolmogorov_turbulence': (
            abs(spectrum_slope - (-5/3)) < 0.3
            if spectrum_slope is not None
            else False
        ),

        # Fields (for visualization - not stored in parquet)
        '_vorticity': vorticity,
        '_Q_criterion': Q,
        '_tke_field': tke,
        '_dissipation_field': epsilon,
        '_enstrophy_field': enstrophy,
        '_energy_spectrum': (k, E_k),
    }
