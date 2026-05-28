"""
Generate pink noise stimulus for the acoustic-noise condition.

Method: FFT-based pink noise synthesis.
  1. Draw white Gaussian noise.
  2. FFT to frequency domain.
  3. Scale each bin by 1/sqrt(f) to impose a -3 dB/octave roll-off.
  4. IFFT back to time domain.
  5. Band-limit to 40 Hz - 16 kHz (speaker-realistic range).
  6. Apply 20 ms raised-cosine fades at start and end (no click on playback).
  7. Normalize to -6 dBFS peak (headroom for in-room SPL calibration).

Output: 24-bit WAV @ 44.1 kHz, 660 seconds (covers the 600 s control recording
with 60 s buffer; no looping required within any recording).

Reproducibility: PRNG is seeded. SHA-256 of the output file is written to
pink_noise.sha256. Re-running this script on any machine produces a
byte-identical WAV.
"""

import hashlib
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch
import matplotlib.pyplot as plt

# ---- Parameters ----------------------------------------------------------
SEED         = 20260526          # today's date; arbitrary but fixed
SAMPLE_RATE  = 44_100            # Hz
DURATION_S   = 660               # seconds; > longest recording (600 s)
HP_HZ        = 40                # high-pass corner (kill sub-bass rumble)
LP_HZ        = 16_000            # low-pass corner (speaker-realistic top)
FADE_MS      = 20                # fade in/out duration
PEAK_DBFS    = -6.0              # leave headroom for SPL calibration
OUT_WAV      = "pink_noise.wav"
OUT_HASH     = "pink_noise.sha256"
OUT_FIG      = "pink_noise_verification.png"

# ---- Generate pink noise via FFT shaping ---------------------------------
rng = np.random.default_rng(SEED)
n_samples = SAMPLE_RATE * DURATION_S
white = rng.standard_normal(n_samples)

# FFT, scale by 1/sqrt(f), inverse FFT
spectrum = np.fft.rfft(white)
freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
scaling = np.ones_like(freqs)
scaling[1:] = 1.0 / np.sqrt(freqs[1:])     # skip DC bin to avoid div-by-zero
pink = np.fft.irfft(spectrum * scaling, n=n_samples)

# ---- Band-limit ---------------------------------------------------------
# Second-order-sections form for numerical stability at long durations
sos_hp = butter(4, HP_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos")
sos_lp = butter(4, LP_HZ, btype="lowpass",  fs=SAMPLE_RATE, output="sos")
pink = sosfiltfilt(sos_hp, pink)
pink = sosfiltfilt(sos_lp, pink)

# ---- Fades ---------------------------------------------------------------
fade_n = int(SAMPLE_RATE * FADE_MS / 1000)
fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n)))
pink[:fade_n]  *= fade
pink[-fade_n:] *= fade[::-1]

# ---- Normalize to target peak -------------------------------------------
peak_lin = 10 ** (PEAK_DBFS / 20)
pink = pink / np.max(np.abs(pink)) * peak_lin

# ---- Write 24-bit WAV ----------------------------------------------------
# scipy.io.wavfile writes int24 if you pass int32 with values in int24 range
pink_int24 = (pink * (2**23 - 1)).astype(np.int32)
wavfile.write(OUT_WAV, SAMPLE_RATE, pink_int24)

# ---- SHA-256 -------------------------------------------------------------
with open(OUT_WAV, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
with open(OUT_HASH, "w") as f:
    f.write(f"{digest}  {OUT_WAV}\n")
print(f"SHA-256: {digest}")

# ---- Verification figure ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) Power spectral density — should show -3 dB/octave between 40 Hz and 16 kHz
f, psd = welch(pink, fs=SAMPLE_RATE, nperseg=2**16)
axes[0].loglog(f, psd)
# Reference line: -3 dB/octave = -10 dB/decade slope, normalized at 1 kHz
ref_f = np.logspace(np.log10(HP_HZ), np.log10(LP_HZ), 100)
ref_psd = psd[np.argmin(np.abs(f - 1000))] * (1000 / ref_f)
axes[0].loglog(ref_f, ref_psd, "r--", label="ideal -3 dB/octave")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("PSD")
axes[0].set_title("Power spectral density")
axes[0].legend()
axes[0].grid(True, which="both", alpha=0.3)

# (2) Stationarity check — RMS in 1 s windows should be ~constant
win = SAMPLE_RATE
rms = np.sqrt(np.mean(pink[:len(pink)//win*win].reshape(-1, win)**2, axis=1))
axes[1].plot(rms)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("RMS amplitude")
axes[1].set_title(f"Stationarity (σ/μ = {rms.std()/rms.mean():.4f})")
axes[1].grid(True, alpha=0.3)

# (3) Amplitude distribution — should be approximately Gaussian
axes[2].hist(pink, bins=200, density=True)
axes[2].set_xlabel("Sample amplitude")
axes[2].set_ylabel("Density")
axes[2].set_title("Amplitude distribution")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=120)
print(f"Wrote {OUT_WAV}, {OUT_HASH}, {OUT_FIG}")
