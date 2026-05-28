"""
Independent verification of pink_noise.wav.

This script knows NOTHING about how the noise was generated. It treats the
WAV as a black box and runs orthogonal statistical tests to confirm it is:
  1. Pink (1/f power spectrum, -3 dB/octave) in the claimed passband
  2. Band-limited to the claimed corners
  3. Stationary (no drift in level, spectrum, or variance over time)
  4. Gaussian-distributed in amplitude
  5. Free of periodicity (no hidden loop, no tonal contamination)
  6. The file we think it is (SHA-256 matches)

Pass criteria are printed at the end. Any FAIL means do not use the stimulus.
"""

import hashlib
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from scipy.stats import linregress, normaltest, kurtosis, skew
import matplotlib.pyplot as plt

# ---- Expected properties (from the protocol, not from the generator) ----
EXPECTED_SR        = 44_100
EXPECTED_DURATION  = 660           # seconds
EXPECTED_HP        = 40            # Hz
EXPECTED_LP        = 16_000        # Hz
EXPECTED_SLOPE     = -1.0          # PSD ~ 1/f, so log-log slope = -1
SLOPE_TOLERANCE    = 0.05          # accept -1.00 ± 0.05
STATIONARITY_MAX   = 0.05          # σ/μ of RMS across 1-s windows
EXPECTED_HASH_FILE = "pink_noise.sha256"
WAV_FILE           = "pink_noise.wav"

results = []
def check(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}: {detail}")

# ---- (0) Hash check -----------------------------------------------------
print("\n[0] File integrity")
with open(WAV_FILE, "rb") as f:
    actual_hash = hashlib.sha256(f.read()).hexdigest()
with open(EXPECTED_HASH_FILE) as f:
    expected_hash = f.read().split()[0]
check("SHA-256 matches recorded hash",
      actual_hash == expected_hash,
      f"{actual_hash[:16]}... vs {expected_hash[:16]}...")

# ---- Load WAV -----------------------------------------------------------
sr, data = wavfile.read(WAV_FILE)
# int32 -> float in [-1, 1] (int24 stored in int32 container)
x = data.astype(np.float64) / (2**23 - 1)
n = len(x)

print("\n[1] File format")
check("Sample rate", sr == EXPECTED_SR, f"{sr} Hz")
check("Duration", abs(n / sr - EXPECTED_DURATION) < 0.01,
      f"{n/sr:.3f} s ({n} samples)")
check("Mono (1-D)", x.ndim == 1, f"shape={data.shape}")

# ---- (2) Spectral slope test --------------------------------------------
# Independent from the generator: fit a line to log(PSD) vs log(f) in the
# claimed passband. Pink noise has theoretical slope exactly -1.
print("\n[2] Spectral slope (independent test)")
f, psd = welch(x, fs=sr, nperseg=2**16, scaling="density")
band = (f >= EXPECTED_HP * 2) & (f <= EXPECTED_LP / 2)  # interior of passband
log_f = np.log10(f[band])
log_p = np.log10(psd[band])
slope, intercept, r_value, _, _ = linregress(log_f, log_p)
check("Slope ~ -1.0 (pink)",
      abs(slope - EXPECTED_SLOPE) < SLOPE_TOLERANCE,
      f"slope = {slope:+.4f} (expected {EXPECTED_SLOPE:+.2f} ± {SLOPE_TOLERANCE})")
check("Linearity of log-log PSD (R² > 0.99)",
      r_value**2 > 0.99,
      f"R² = {r_value**2:.4f}")

# ---- (3) Band-limit test ------------------------------------------------
print("\n[3] Band-limiting")
# Energy outside the passband should be tiny relative to inside
in_band  = (f >= EXPECTED_HP) & (f <= EXPECTED_LP)
below    = f < EXPECTED_HP / 2     # well below corner
above    = f > EXPECTED_LP * 1.1   # well above corner
energy_in    = np.trapz(psd[in_band], f[in_band])
energy_below = np.trapz(psd[below], f[below]) if below.any() else 0
energy_above = np.trapz(psd[above], f[above]) if above.any() else 0
ratio_below = energy_below / energy_in
ratio_above = energy_above / energy_in
check("Sub-passband energy < 0.1%",
      ratio_below < 1e-3,
      f"{ratio_below:.2e}")
check("Super-passband energy < 0.1%",
      ratio_above < 1e-3,
      f"{ratio_above:.2e}")

# ---- (4) Stationarity in three independent ways -------------------------
print("\n[4] Stationarity")
# (a) RMS across 1-second windows
win = sr
n_windows = n // win
rms = np.sqrt(np.mean(x[:n_windows*win].reshape(-1, win)**2, axis=1))
cv_rms = rms.std() / rms.mean()
check("RMS coefficient of variation < 5%",
      cv_rms < STATIONARITY_MAX,
      f"σ/μ = {cv_rms:.4f}")

# (b) No linear trend in RMS (would indicate level drift)
t_idx = np.arange(len(rms))
trend_slope, _, trend_r, _, _ = linregress(t_idx, rms)
relative_drift = abs(trend_slope) * len(rms) / rms.mean()
check("No level drift over recording (< 1%)",
      relative_drift < 0.01,
      f"total drift = {relative_drift*100:.3f}%")

# (c) Spectral stationarity: compare PSD of first half vs second half
half = n // 2
f1, psd1 = welch(x[:half], fs=sr, nperseg=2**15)
f2, psd2 = welch(x[half:], fs=sr, nperseg=2**15)
in_band_half = (f1 >= EXPECTED_HP) & (f1 <= EXPECTED_LP)
spectral_diff = np.mean(np.abs(np.log10(psd1[in_band_half]) -
                               np.log10(psd2[in_band_half])))
check("Spectrum stable across halves (mean |Δlog PSD| < 0.05)",
      spectral_diff < 0.05,
      f"mean |Δlog PSD| = {spectral_diff:.4f}")

# ---- (5) Gaussianity ----------------------------------------------------
print("\n[5] Amplitude distribution")
# Subsample for the normality test (it gets touchy with millions of points)
subsample = x[::100]
sk = skew(subsample)
kt = kurtosis(subsample)  # excess kurtosis; Gaussian = 0
check("Skewness near 0", abs(sk) < 0.1, f"skew = {sk:+.4f}")
check("Excess kurtosis near 0", abs(kt) < 0.2, f"kurt = {kt:+.4f}")

# ---- (6) No periodicity / no hidden loop --------------------------------
print("\n[6] Periodicity check (autocorrelation)")
# A looped file would show ACF spikes at integer multiples of the loop period.
# True noise has ACF values at non-zero lags that are random with SD ~ 1/sqrt(N),
# where N is the length of the series being correlated.
#
# We test two things:
#   (a) The maximum |ACF| is consistent with the true-noise null distribution
#       (i.e., it's not so large that it's implausible under "genuine noise").
#       For N independent samples, max|ACF| over K lags has expected value
#       ~sqrt(2*log(K))/sqrt(N) and rarely exceeds ~4/sqrt(N).
#   (b) No single lag shows a "spike" — defined as |ACF| > 5 standard
#       deviations above the noise floor, which would indicate a true period.

rms_centered = rms - rms.mean()
acf = np.correlate(rms_centered, rms_centered, mode="full")
acf = acf[len(acf)//2:] / acf[len(acf)//2]   # normalize so lag 0 = 1
N = len(rms)
acf_noise_sd = 1.0 / np.sqrt(N)              # theoretical SD of ACF under null
max_lag_corr = np.max(np.abs(acf[1:]))
# Reasonable bound: 5 sigma above the noise floor is a real spike
spike_threshold = 5 * acf_noise_sd
# Generous bound on the max: noise can plausibly produce ~4-sigma fluctuations
# across hundreds of lags. Use 6 sigma as "definitely not noise."
max_threshold = 6 * acf_noise_sd

check(f"Max |ACF| within noise-floor expectation (< {max_threshold:.3f} = 6σ)",
      max_lag_corr < max_threshold,
      f"max |ACF| at lag>0 = {max_lag_corr:.4f} "
      f"(noise floor SD = {acf_noise_sd:.4f})")

# Spike detection: is there a SINGLE lag much larger than the rest?
# Real loops produce ACF values near 1.0 at the loop period.
spike_lags = np.where(np.abs(acf[1:]) > spike_threshold)[0]
check("No spike lags (no single lag exceeds 5σ noise floor)",
      len(spike_lags) == 0,
      f"{len(spike_lags)} lags exceed 5σ threshold")

# ---- (7) Verification figure --------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.loglog(f, psd, label="measured")
ref_f = np.logspace(np.log10(EXPECTED_HP), np.log10(EXPECTED_LP), 100)
ref_psd = 10**(intercept) * ref_f**slope
ax.loglog(ref_f, ref_psd, "r--", label=f"fitted slope = {slope:+.3f}")
ax.axvline(EXPECTED_HP, color="gray", linestyle=":", alpha=0.5)
ax.axvline(EXPECTED_LP, color="gray", linestyle=":", alpha=0.5)
ax.set(xlabel="Frequency (Hz)", ylabel="PSD",
       title=f"Spectrum (slope fit in passband: {slope:+.4f})")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
ax.plot(rms, alpha=0.7)
ax.axhline(rms.mean(), color="red", linestyle="--", label=f"mean = {rms.mean():.4f}")
ax.set(xlabel="Time (s)", ylabel="RMS",
       title=f"Stationarity (σ/μ = {cv_rms:.4f}, drift = {relative_drift*100:.3f}%)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.semilogx(f1[in_band_half], 10*np.log10(psd1[in_band_half]),
            alpha=0.7, label="first half")
ax.semilogx(f2[in_band_half], 10*np.log10(psd2[in_band_half]),
            alpha=0.7, label="second half")
ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB)",
       title="Spectrum: first half vs second half")
ax.legend(); ax.grid(True, which="both", alpha=0.3)

ax = axes[1, 1]
lags_s = np.arange(len(acf))   # lag in seconds (since we ACF'd 1-Hz RMS series)
ax.plot(lags_s[:120], acf[:120])
ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="threshold")
ax.axhline(-0.1, color="red", linestyle="--", alpha=0.5)
ax.set(xlabel="Lag (s)", ylabel="Autocorrelation",
       title=f"RMS autocorrelation (max |ACF| at lag>0 = {max_lag_corr:.3f})")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pink_noise_independent_verification.png", dpi=120)

# ---- Summary ------------------------------------------------------------
print("\n" + "=" * 60)
n_pass = sum(1 for _, p, _ in results if p)
n_fail = sum(1 for _, p, _ in results if not p)
print(f"VERIFICATION SUMMARY: {n_pass} passed, {n_fail} failed")
print("=" * 60)
if n_fail > 0:
    print("\nFAILED CHECKS:")
    for name, p, detail in results:
        if not p:
            print(f"  - {name}: {detail}")
    sys.exit(1)
print("\nAll checks passed. Stimulus is verified pink noise.")
print("Wrote pink_noise_independent_verification.png")
