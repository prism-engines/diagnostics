"""
Kalman Filter

State estimation, prediction, smoothing.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class KalmanFilter:
    """
    Linear Kalman Filter.

    State equation:  x[k] = A*x[k-1] + B*u[k] + w[k]
    Observation:     z[k] = H*x[k] + v[k]

    w ~ N(0, Q), v ~ N(0, R)
    """

    def __init__(self, A: np.ndarray, B: np.ndarray = None,
                 H: np.ndarray = None, Q: np.ndarray = None,
                 R: np.ndarray = None):
        """
        Initialize Kalman filter.

        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m), optional
            H: Observation matrix (p x n), default identity
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (p x p)
        """
        self.A = np.atleast_2d(A)
        self.n = self.A.shape[0]

        self.B = np.atleast_2d(B) if B is not None else np.zeros((self.n, 1))

        if H is None:
            self.H = np.eye(self.n)
        else:
            self.H = np.atleast_2d(H)

        self.p = self.H.shape[0]

        self.Q = np.atleast_2d(Q) if Q is not None else np.eye(self.n) * 0.01
        self.R = np.atleast_2d(R) if R is not None else np.eye(self.p) * 0.1

        # State and covariance
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

    def predict(self, u: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict step.

        Args:
            u: Control input (optional)

        Returns:
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        if u is None:
            u = np.zeros(self.B.shape[1])
        u = np.atleast_1d(u)

        # State prediction
        x_pred = self.A @ self.x + self.B @ u

        # Covariance prediction
        P_pred = self.A @ self.P @ self.A.T + self.Q

        return x_pred, P_pred

    def update(self, z: np.ndarray, x_pred: np.ndarray,
               P_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step.

        Args:
            z: Measurement
            x_pred: Predicted state
            P_pred: Predicted covariance

        Returns:
            x: Updated state
            P: Updated covariance
            K: Kalman gain
        """
        z = np.atleast_1d(z)

        # Innovation
        y = z - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # State update
        x = x_pred + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H
        P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        return x, P, K

    def filter_step(self, z: np.ndarray, u: np.ndarray = None) -> Dict[str, Any]:
        """
        Single filter step (predict + update).

        Args:
            z: Measurement
            u: Control input (optional)

        Returns:
            state: Updated state estimate
            covariance: Updated covariance
            innovation: Measurement innovation
            kalman_gain: Kalman gain
        """
        x_pred, P_pred = self.predict(u)
        x, P, K = self.update(z, x_pred, P_pred)

        self.x = x
        self.P = P

        innovation = np.atleast_1d(z) - self.H @ x_pred

        return {
            'state': x.tolist(),
            'covariance': P.tolist(),
            'predicted_state': x_pred.tolist(),
            'innovation': innovation.tolist(),
            'kalman_gain': K.tolist(),
        }


def filter_sequence(measurements: np.ndarray, A: np.ndarray,
                    H: np.ndarray = None, Q: np.ndarray = None,
                    R: np.ndarray = None, B: np.ndarray = None,
                    u: np.ndarray = None,
                    x0: np.ndarray = None, P0: np.ndarray = None) -> Dict[str, Any]:
    """
    Apply Kalman filter to measurement sequence.

    Args:
        measurements: Measurement sequence (T x p)
        A: State transition matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        B: Control input matrix
        u: Control input sequence (T x m)
        x0: Initial state
        P0: Initial covariance

    Returns:
        states: Filtered state estimates (T x n)
        covariances: State covariances (T x n x n)
        innovations: Measurement innovations (T x p)
    """
    measurements = np.atleast_2d(measurements)
    if measurements.shape[0] < measurements.shape[1]:
        measurements = measurements.T

    T = measurements.shape[0]
    A = np.atleast_2d(A)
    n = A.shape[0]

    kf = KalmanFilter(A, B, H, Q, R)

    if x0 is not None:
        kf.x = np.atleast_1d(x0)
    if P0 is not None:
        kf.P = np.atleast_2d(P0)

    states = np.zeros((T, n))
    covariances = np.zeros((T, n, n))
    innovations = np.zeros((T, kf.p))
    gains = np.zeros((T, n, kf.p))

    for t in range(T):
        z = measurements[t]
        u_t = u[t] if u is not None else None

        result = kf.filter_step(z, u_t)

        states[t] = kf.x
        covariances[t] = kf.P
        innovations[t] = result['innovation']
        gains[t] = np.array(result['kalman_gain'])

    return {
        'states': states.tolist(),
        'covariances': covariances.tolist(),
        'innovations': innovations.tolist(),
        'final_state': kf.x.tolist(),
        'final_covariance': kf.P.tolist(),
        'innovation_variance': float(np.var(innovations)),
    }


def rts_smoother(states: np.ndarray, covariances: np.ndarray,
                 A: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    """
    Rauch-Tung-Striebel smoother for offline estimation.

    Args:
        states: Filtered states from forward pass (T x n)
        covariances: Filtered covariances (T x n x n)
        A: State transition matrix
        Q: Process noise covariance

    Returns:
        smoothed_states: Smoothed state estimates
        smoothed_covariances: Smoothed covariances
    """
    states = np.atleast_2d(states)
    covariances = np.atleast_3d(covariances)
    A = np.atleast_2d(A)
    Q = np.atleast_2d(Q)

    T, n = states.shape
    smoothed_states = np.zeros_like(states)
    smoothed_covariances = np.zeros_like(covariances)

    # Initialize with last filtered values
    smoothed_states[-1] = states[-1]
    smoothed_covariances[-1] = covariances[-1]

    # Backward pass
    for t in range(T - 2, -1, -1):
        # Predicted covariance for t+1
        P_pred = A @ covariances[t] @ A.T + Q

        # Smoother gain
        C = covariances[t] @ A.T @ np.linalg.inv(P_pred)

        # Smoothed state
        smoothed_states[t] = states[t] + C @ (smoothed_states[t + 1] - A @ states[t])

        # Smoothed covariance
        smoothed_covariances[t] = covariances[t] + C @ (smoothed_covariances[t + 1] - P_pred) @ C.T

    return {
        'smoothed_states': smoothed_states.tolist(),
        'smoothed_covariances': smoothed_covariances.tolist(),
    }


def estimate_noise_covariances(measurements: np.ndarray, states: np.ndarray,
                               A: np.ndarray, H: np.ndarray) -> Dict[str, Any]:
    """
    Estimate Q and R from data using innovation-based method.

    Args:
        measurements: Measurement sequence
        states: State estimates
        A: State transition matrix
        H: Observation matrix

    Returns:
        Q_est: Estimated process noise covariance
        R_est: Estimated measurement noise covariance
    """
    measurements = np.atleast_2d(measurements)
    states = np.atleast_2d(states)
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)

    T = measurements.shape[0]

    # Process noise estimate from state transitions
    process_residuals = states[1:] - (A @ states[:-1].T).T
    Q_est = np.cov(process_residuals.T)

    # Measurement noise estimate from innovations
    measurement_residuals = measurements - (H @ states.T).T
    R_est = np.cov(measurement_residuals.T)

    return {
        'Q': Q_est.tolist() if Q_est.ndim > 0 else [[float(Q_est)]],
        'R': R_est.tolist() if R_est.ndim > 0 else [[float(R_est)]],
        'process_noise_std': float(np.sqrt(np.trace(Q_est) / Q_est.shape[0])),
        'measurement_noise_std': float(np.sqrt(np.trace(R_est) / R_est.shape[0])),
    }


def compute(measurements: np.ndarray = None, A: np.ndarray = None,
            H: np.ndarray = None, Q: np.ndarray = None,
            R: np.ndarray = None, smooth: bool = False,
            **kwargs) -> Dict[str, Any]:
    """
    Main entry point for Kalman filtering.
    """
    if measurements is None or A is None:
        return {'error': 'Provide measurements and state transition matrix A'}

    measurements = np.atleast_2d(measurements)
    A = np.atleast_2d(A)

    # Filter
    result = filter_sequence(measurements, A, H, Q, R,
                             kwargs.get('B'), kwargs.get('u'),
                             kwargs.get('x0'), kwargs.get('P0'))

    # Smooth if requested
    if smooth and Q is not None:
        states = np.array(result['states'])
        covariances = np.array(result['covariances'])
        smooth_result = rts_smoother(states, covariances, A, Q)
        result['smoothed_states'] = smooth_result['smoothed_states']
        result['smoothed_covariances'] = smooth_result['smoothed_covariances']

    return result
