# Pink Noise Stimulus

Acoustic stimulus for the noise condition of the BCI noise-robustness protocol (see main protocol §3). This directory is the full reproducibility bundle: generator, verifier, stimulus file, integrity hash, verification figures.

## Bundle contents

| File | Purpose |
| --- | --- |
| `generate_pink_noise.py` | Generator. Seeded PRNG, FFT-based spectral shaping, Butterworth band-limiting, raised-cosine fades, peak normalization. |
| `verify_pink_noise.py` | Independent verifier. Treats the WAV as a black box; runs 15 statistical checks via orthogonal methods to the generator. |
| `pink_noise.wav` | The stimulus. 24-bit PCM, 44.1 kHz, mono, 660 s. |
| `pink_noise.sha256` | SHA-256 digest of `pink_noise.wav` for integrity checking. |
| `pink_noise_verification.png` | Generator-side figure (PSD, stationarity, amplitude distribution). |
| `pink_noise_independent_verification.png` | Verifier-side four-panel figure (fitted-slope PSD, stationarity with drift line, first-half vs second-half spectrum, RMS autocorrelation). |
| `README.md` | This file. |

## Stimulus parameters

- **Spectrum:** pink (1/f, −3 dB/octave), band-limited 40 Hz – 16 kHz
- **Sample rate:** 44.1 kHz
- **Bit depth:** 24-bit PCM
- **Duration:** 660 s — covers the 600 s control recording with 60 s headroom; no looping required within any recording in the protocol
- **Peak level:** −6 dBFS (headroom for in-room SPL calibration)
- **Fades:** 20 ms raised-cosine at start and end (prevents click artifacts on playback start/stop)
- **PRNG seed:** fixed in `generate_pink_noise.py` (file is byte-reproducible)

At playback, the stimulus is calibrated to 65 ± 3 dB(A) at the subject's ear position using the SPL-meter procedure described in the main protocol.

## Verification results

All 15 checks pass. Highlights from the most recent run:

| Property | Result | Threshold |
| --- | --- | --- |
| SHA-256 matches recorded hash | ✓ | exact |
| Sample rate / duration / mono | ✓ | as specified |
| Spectral slope (passband) | −0.9998 | −1.00 ± 0.05 |
| Log-log PSD linearity (R²) | 0.9985 | > 0.99 |
| Sub-passband energy fraction | 7.6 × 10⁻⁷ | < 10⁻³ |
| Super-passband energy fraction | 9.9 × 10⁻⁶ | < 10⁻³ |
| RMS coefficient of variation | 0.0123 | < 0.05 |
| Level drift across recording | 0.110% | < 1% |
| First-half vs second-half spectrum (mean ǀΔlog PSDǀ) | 0.0167 | < 0.05 |
| Skewness | −0.0003 | ǀ·ǀ < 0.1 |
| Excess kurtosis | −0.0060 | ǀ·ǀ < 0.2 |
| Max ǀACFǀ at non-zero lag | 0.1127 | < 6/√N = 0.234 |
| Spike lags (> 5σ noise floor) | 0 | 0 |

The verifier and generator share no code paths. The verifier applies linear regression for the slope test (the generator imposes 1/√f scaling without ever fitting a slope), energy integration for band-limiting (the generator applies Butterworth filters without ever computing residual energy), three orthogonal stationarity tests (RMS coefficient of variation, linear trend, and first-half vs second-half PSD), and an autocorrelation test against the theoretical 1/√N noise-floor distribution for periodicity. A bug or oversight in the generator would not propagate into a passing verification.

## Reproducing the bundle

```bash
python -m venv .venv
source .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install numpy scipy matplotlib

python generate_pink_noise.py
python verify_pink_noise.py
```

The SHA-256 emitted by `generate_pink_noise.py` must match the digest in `pink_noise.sha256` exactly. If it does not, do not use the stimulus — investigate first (likely causes: NumPy/SciPy version difference affecting filter coefficients, or a modified seed).

## Protocol reference

Drop-in description for the protocol's stimulus section:

> **Acoustic noise stimulus.** Pink noise (1/f spectrum), band-limited 40 Hz to 16 kHz, 660 s duration, 44.1 kHz sample rate, 24-bit WAV, generated via FFT-domain spectral shaping of seeded Gaussian white noise (`protocol/stimuli/generate_pink_noise.py`). Stimulus integrity verified by `protocol/stimuli/verify_pink_noise.py`: 15 statistical checks covering spectral slope, band-limiting, stationarity, amplitude distribution, and periodicity. SHA-256 of `pink_noise.wav` recorded in `protocol/stimuli/pink_noise.sha256`. Playback level calibrated to 65 ± 3 dB(A) at the subject's ear position using a phone-based SPL meter, identical procedure to other audio-condition drafts.
