import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares

F_MARGIN = 100e6

# --- Model definitions ---

def impedance_model_complex(f, R, L, C):
    omega = 2 * np.pi * f
    return R + 1j * (omega * L - 1 / (omega * C))

def mag_phase_residuals(params, f, Z_measured, phase_weight=1.0):
    """Residuals combining magnitude and phase differences."""
    R, L, C = params
    Z_pred = impedance_model_complex(f, R, L, C)
    
    mag_pred = np.abs(Z_pred)
    mag_meas = np.abs(Z_measured)
    
    phase_pred = np.angle(Z_pred)
    phase_meas = np.angle(Z_measured)
    
    # Unwrap phase to prevent discontinuities (e.g., jump from π to -π)
    phase_pred = np.unwrap(phase_pred)
    phase_meas = np.unwrap(phase_meas)
    
    # Normalize phase difference to roughly comparable scale to magnitude
    mag_diff = mag_pred - mag_meas
    phase_diff = (phase_pred - phase_meas) * phase_weight
    
    return np.concatenate([mag_diff, phase_diff])

def mag_phase_cost(params, f, Z_measured, phase_weight=1.0):
    """Scalar cost for global optimization."""
    r = mag_phase_residuals(params, f, Z_measured, phase_weight)
    return np.sum(r**2)

def parallel_RC(f, R, C):
    omega = 2 * np.pi * f
    return R / (1 + 1j * omega * R * C)


# --- Data setup ---
frequencies = np.array([315e6, 433.92e6, 868.35e6, 915e6])

Z_measured = np.array([
    parallel_RC(315e6, 150, 6.3e-12),
    parallel_RC(433.92e6, 120, 6.3e-12),
    parallel_RC(868.35e6, 90, 6.3e-12),
    parallel_RC(915e6, 80, 6.3e-12)
])

# --- Bounds and initial conditions ---
lower_bounds = [1e-3, 1e-12, 1e-15]
upper_bounds = [1e4, 1e-6, 1e-9]
phase_weight = 50.0  # relative weight for phase residual (adjust if needed)

# --- Global optimization ---
print("Running global optimization (Differential Evolution)...")
de_result = differential_evolution(
    mag_phase_cost,
    bounds=list(zip(lower_bounds, upper_bounds)),
    args=(frequencies, Z_measured, phase_weight),
    maxiter=5000,
    popsize=25,
    tol=1e-6,
    updating='deferred',
    workers=-1,
    disp=True
)

print("\nGlobal optimization result:")
print(f"  R = {de_result.x[0]:.4f} Ω")
print(f"  L = {de_result.x[1]:.6e} H")
print(f"  C = {de_result.x[2]:.6e} F")

# --- Local refinement ---
print("\nRefining with local least squares (magnitude + phase)...")
local_result = least_squares(
    mag_phase_residuals,
    de_result.x,
    bounds=(lower_bounds, upper_bounds),
    args=(frequencies, Z_measured, phase_weight),
    max_nfev=10000,
    xtol=1e-12,
    ftol=1e-12,
    gtol=1e-12,
    verbose=2
)

R_est, L_est, C_est = local_result.x

print("\nFinal estimated values (after local refinement):")
print(f"  R = {R_est:.4f} Ω")
print(f"  L = {L_est:.6e} H")
print(f"  C = {C_est:.6e} F")

# --- Frequency sweep for visualization ---
f_begin = frequencies[0] - F_MARGIN
f_end = frequencies[-1] + F_MARGIN
f_range = np.linspace(f_begin, f_end, num=1000)

Z_fit = impedance_model_complex(f_range, R_est, L_est, C_est)

# --- Plotting: magnitude + phase ---
fig, ax1 = plt.subplots(figsize=(8, 5))

# Magnitude
ax1.plot(f_range, np.abs(Z_fit), label="Fitted |Z|", color="blue", linewidth=2)
ax1.scatter(frequencies, np.abs(Z_measured), color="red", label="Measured |Z|", zorder=5)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("|Z| (Ω)", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')

# Phase (secondary axis)
ax2 = ax1.twinx()
ax2.plot(f_range, np.angle(Z_fit, deg=True), color="purple", label="Fitted ∠Z", linewidth=2, linestyle="--")
ax2.scatter(frequencies, np.angle(Z_measured, deg=True), color="magenta", label="Measured ∠Z", zorder=5)
ax2.set_ylabel("Phase ∠Z (degrees)", color="purple")
ax2.tick_params(axis='y', labelcolor='purple')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

ax1.grid(True, which="both", linestyle="--", alpha=0.6)
ax1.set_title("Series RLC Impedance Fit (Optimizing Magnitude + Phase)")
fig.tight_layout()
plt.show()

