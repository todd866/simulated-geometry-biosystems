"""
Gaussian Multiplicative Chaos (GMC) Simulation

Simulates GMC measures on the circle for various values of gamma.
Demonstrates the subcritical regime (γ < √2) where dimension matching holds.

The GMC measure is constructed as:
    μ_γ = exp(γX - γ²/2 * E[X²]) dx

where X is a log-correlated Gaussian field.

Reference:
- Garban & Vargas (2023), arXiv:2311.04027
- Lin, Qiu & Tan (2024), arXiv:2411.13923
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from typing import Tuple, Optional

# Critical value
GAMMA_CRITICAL = np.sqrt(2)


def theory_dimension(gamma: float) -> float:
    """
    Theoretical dimension D*(γ) for GMC (Lin-Qiu-Tan 2024).

    D*(γ) = 1 - γ²           if 0 < γ < 1/√2
          = (√2 - γ)²        if 1/√2 ≤ γ < √2

    Parameters
    ----------
    gamma : float
        GMC parameter (subcritical: 0 < γ < √2)

    Returns
    -------
    D : float
        Theoretical dimension
    """
    gamma_transition = 1.0 / np.sqrt(2)

    if gamma < gamma_transition:
        return 1.0 - gamma**2
    elif gamma < GAMMA_CRITICAL:
        return (np.sqrt(2) - gamma)**2
    else:
        return 0.0


def generate_log_correlated_field(
    n_points: int,
    n_modes: int = 1000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a (regularized) log-correlated Gaussian field on [0, 2π].

    The covariance is approximately -log|x-y| for |x-y| not too small,
    regularized at short distances by the mode cutoff.

    Parameters
    ----------
    n_points : int
        Number of discretization points
    n_modes : int
        Number of Fourier modes (acts as UV cutoff)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    X : ndarray
        The log-correlated Gaussian field values at each point
    """
    if seed is not None:
        np.random.seed(seed)

    # Points on [0, 2π)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Build field from Fourier modes
    # For log-correlated field, variance of mode k is ~ 1/|k|
    X = np.zeros(n_points)

    for k in range(1, n_modes + 1):
        # Variance ~ 1/k gives log-correlated covariance
        sigma_k = 1.0 / np.sqrt(k)

        # Random Fourier coefficients
        a_k = np.random.normal(0, sigma_k)
        b_k = np.random.normal(0, sigma_k)

        X += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)

    return X


def construct_gmc_measure(
    X: np.ndarray,
    gamma: float,
    normalize: bool = True
) -> np.ndarray:
    """
    Construct GMC measure from log-correlated field.

    μ_γ(dx) = exp(γX - γ²/2 * Var(X)) dx

    Parameters
    ----------
    X : ndarray
        Log-correlated Gaussian field
    gamma : float
        GMC parameter (subcritical: γ < √2 ≈ 1.41)
    normalize : bool
        Whether to normalize to a probability measure

    Returns
    -------
    mu : ndarray
        The GMC measure density at each point
    """
    # Compute variance for normalization
    var_X = np.var(X)

    # GMC density (Wick-ordered exponential)
    mu = np.exp(gamma * X - 0.5 * gamma**2 * var_X)

    if normalize:
        mu = mu / np.sum(mu)

    return mu


def estimate_correlation_dimension(
    mu: np.ndarray,
    theta: np.ndarray,
    r_values: Optional[np.ndarray] = None,
    n_samples: int = 5000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate correlation dimension from GMC measure.

    Uses the correlation integral:
        C(r) = ∫∫ μ(dx)μ(dy) 1(|x-y| < r)

    Correlation dimension D_C is the slope of log C(r) vs log r.

    Parameters
    ----------
    mu : ndarray
        GMC measure density
    theta : ndarray
        Angular positions
    r_values : ndarray, optional
        Distance thresholds to evaluate
    n_samples : int
        Number of point pairs to sample

    Returns
    -------
    D_C : float
        Estimated correlation dimension
    r_values : ndarray
        Distance thresholds used
    C_values : ndarray
        Correlation integral values
    """
    n = len(mu)

    if r_values is None:
        r_values = np.logspace(-2, 0, 30)

    # Sample points according to measure
    probs = mu / np.sum(mu)
    idx1 = np.random.choice(n, size=n_samples, p=probs)
    idx2 = np.random.choice(n, size=n_samples, p=probs)

    # Compute distances (on circle)
    d_theta = np.abs(theta[idx1] - theta[idx2])
    distances = np.minimum(d_theta, 2*np.pi - d_theta)

    # Compute correlation integral for each r
    C_values = np.array([np.mean(distances < r) for r in r_values])

    # Fit slope in log-log (avoiding zeros)
    valid = C_values > 0
    if np.sum(valid) < 5:
        return np.nan, r_values, C_values

    log_r = np.log(r_values[valid])
    log_C = np.log(C_values[valid])

    # Linear fit for dimension
    coeffs = np.polyfit(log_r, log_C, 1)
    D_C = coeffs[0]

    return D_C, r_values, C_values


def estimate_fourier_dimension(
    mu: np.ndarray,
    max_k: int = 100
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Fourier dimension from GMC measure.

    The Fourier dimension D_F is defined by |μ̂(n)| = O(|n|^{-D_F/2}).
    Computes Fourier coefficients and fits decay exponent.

    Parameters
    ----------
    mu : ndarray
        GMC measure density
    max_k : int
        Maximum mode number to consider

    Returns
    -------
    D_F : float
        Estimated Fourier dimension
    k_values : ndarray
        Mode numbers
    coeff_magnitudes : ndarray
        |μ̂(k)| at each mode
    """
    # Compute Fourier transform
    mu_hat = fft(mu)
    n = len(mu)

    # Fourier coefficient magnitudes
    coeff_mag = np.abs(mu_hat)

    # Use modes 1 to max_k
    k_values = np.arange(1, min(max_k, n//2))
    coeff_k = coeff_mag[k_values]

    # Fit decay: |μ̂(k)| ~ k^(-D_F/2)
    valid = coeff_k > 0
    if np.sum(valid) < 5:
        return np.nan, k_values, coeff_k

    log_k = np.log(k_values[valid])
    log_c = np.log(coeff_k[valid])

    coeffs = np.polyfit(log_k, log_c, 1)
    decay_exp = -coeffs[0]

    # Fourier dimension from decay: |μ̂(k)| ~ k^(-D_F/2)
    D_F = 2 * decay_exp

    return D_F, k_values, coeff_k


def simulate_gmc_ensemble(
    gamma_values: np.ndarray,
    n_realizations: int = 20,
    n_points: int = 4096,
    n_modes: int = 500
) -> dict:
    """
    Simulate GMC ensemble and compute dimensions for various gamma.

    Parameters
    ----------
    gamma_values : ndarray
        GMC parameters to test (should be < √2)
    n_realizations : int
        Number of independent realizations per gamma
    n_points : int
        Discretization points
    n_modes : int
        Fourier modes for field

    Returns
    -------
    results : dict
        Dictionary with gamma values, dimension estimates, and theory values
    """
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    D_C_all = []
    D_F_all = []
    D_C_std = []
    D_F_std = []
    D_theory = []

    for gamma in gamma_values:
        D_C_samples = []
        D_F_samples = []

        for i in range(n_realizations):
            # Generate field and measure
            X = generate_log_correlated_field(n_points, n_modes, seed=i*1000+int(gamma*100))
            mu = construct_gmc_measure(X, gamma)

            # Estimate dimensions
            D_C, _, _ = estimate_correlation_dimension(mu, theta)
            D_F, _, _ = estimate_fourier_dimension(mu)

            if not np.isnan(D_C):
                D_C_samples.append(D_C)
            if not np.isnan(D_F):
                D_F_samples.append(D_F)

        D_C_all.append(np.mean(D_C_samples) if D_C_samples else np.nan)
        D_F_all.append(np.mean(D_F_samples) if D_F_samples else np.nan)
        D_C_std.append(np.std(D_C_samples) if len(D_C_samples) > 1 else 0)
        D_F_std.append(np.std(D_F_samples) if len(D_F_samples) > 1 else 0)
        D_theory.append(theory_dimension(gamma))

        print(f"γ = {gamma:.2f}: D_C = {D_C_all[-1]:.3f} ± {D_C_std[-1]:.3f}, "
              f"D_F = {D_F_all[-1]:.3f} ± {D_F_std[-1]:.3f}, "
              f"D* = {D_theory[-1]:.3f}")

    return {
        'gamma': gamma_values,
        'D_C': np.array(D_C_all),
        'D_F': np.array(D_F_all),
        'D_C_std': np.array(D_C_std),
        'D_F_std': np.array(D_F_std),
        'D_theory': np.array(D_theory)
    }


if __name__ == "__main__":
    # Quick demo
    print("GMC Simulation Demo")
    print("=" * 50)
    print(f"Critical γ = √2 ≈ {GAMMA_CRITICAL:.4f}")
    print()

    n_points = 2048
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Generate for subcritical gamma values only
    gammas = [0.3, 0.6, 0.9, 1.2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, gamma in zip(axes.flat, gammas):
        X = generate_log_correlated_field(n_points, n_modes=500, seed=42)
        mu = construct_gmc_measure(X, gamma)

        ax.plot(theta, mu, 'b-', linewidth=0.5, alpha=0.7)
        ax.fill_between(theta, 0, mu, alpha=0.3)
        ax.set_title(f'GMC measure, γ = {gamma} (D* = {theory_dimension(gamma):.3f})')
        ax.set_xlabel('θ')
        ax.set_ylabel('μ_γ(θ)')
        ax.set_xlim(0, 2*np.pi)

    plt.tight_layout()
    plt.savefig('../figures/gmc_measures.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nDimension estimates vs theory:")
    X = generate_log_correlated_field(n_points, seed=42)
    for gamma in gammas:
        mu = construct_gmc_measure(X, gamma)
        D_C, _, _ = estimate_correlation_dimension(mu, theta)
        D_F, _, _ = estimate_fourier_dimension(mu)
        D_star = theory_dimension(gamma)
        print(f"γ = {gamma}: D_C = {D_C:.3f}, D_F = {D_F:.3f}, D* = {D_star:.3f}")
