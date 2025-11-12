import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Model definitions ---
def re_model(f, a_re, b_re):
    return b_re / (f ** a_re)

def im_model(f, a_im, b_im):
    return b_im / (f ** a_im)

def parallel_RC(f, R, C):
    omega = 2 * np.pi * f
    return R / (1 + 1j * omega * R * C)

# --- Example data ---
frequencies = np.array([315e6, 433.92e6, 868.35e6, 915e6])

Z_measured = np.array([
    parallel_RC(315e6, 150, 6.3e-12),
    parallel_RC(433.92e6, 120, 6.3e-12),
    parallel_RC(868.35e6, 90, 6.3e-12),
    parallel_RC(915e6, 80, 6.3e-12)
])

# Extract real and imaginary components
Z_re = Z_measured.real
Z_im = Z_measured.imag

# --- Fit real and imaginary parts independently ---
popt_re, _ = curve_fit(re_model, frequencies, Z_re, p0=[1.0, 1e10])
popt_im, _ = curve_fit(im_model, frequencies, Z_im, p0=[1.0, 1e10])

a_re, b_re = popt_re
a_im, b_im = popt_im

print("Fitted parameters:")
print(f"  a_re = {a_re:.6f}, b_re = {b_re:.6e}")
print(f"  a_im = {a_im:.6f}, b_im = {b_im:.6e}")

# --- Generate smooth fit curves ---
f_range = np.linspace(frequencies[0], frequencies[-1], 1000)
Z_re_fit = re_model(f_range, a_re, b_re)
Z_im_fit = im_model(f_range, a_im, b_im)

# --- Plot results ---
fig, ax1 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Real part
ax1[0].scatter(frequencies, Z_re, color="red", label="Measured Re(Z)", zorder=5)
ax1[0].plot(f_range, Z_re_fit, color="blue", label=f"Fit Re(Z) = b_re/f^a_re", linewidth=2)
ax1[0].set_ylabel("Re(Z) (Ω)")
ax1[0].legend()
ax1[0].grid(True, linestyle="--", alpha=0.6)

# Imaginary part
ax1[1].scatter(frequencies, Z_im, color="magenta", label="Measured Im(Z)", zorder=5)
ax1[1].plot(f_range, Z_im_fit, color="purple", label=f"Fit Im(Z) = b_im/f^a_im", linewidth=2)
ax1[1].set_xlabel("Frequency (Hz)")
ax1[1].set_ylabel("Im(Z) (Ω)")
ax1[1].legend()
ax1[1].grid(True, linestyle="--", alpha=0.6)

fig.suptitle("Independent Power-Law Fits for Re(Z) and Im(Z)")
plt.tight_layout()
plt.show()
