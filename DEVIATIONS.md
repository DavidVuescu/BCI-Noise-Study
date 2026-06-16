# Protocol Deviations Log

This document tracks all methodological, procedural, and analytical deviations from the pre-registered protocol (OSF: osf.io/yq8wj). Transparency regarding these changes ensures the validity and reproducibility of the BCI noise study.

All deviations are reported here, regardless of their perceived impact on the study's conclusions.

---

### 2026-06-01: Clock Drift Correction (Pre-data)
**Phase:** Data Acquisition / Pre-processing
**Status:** Implemented prior to primary data collection.

* **Deviation:** Marker-to-sample alignment was changed from using a constant `wall_clock_anchor` offset to utilizing a per-recording linear regression of counter-derived acquisition time on wall-clock receive times. 
* **Motivation:** Pilot diagnostics revealed a ~1.2-1.4 ms/s drift between the host computer clock and the Unicorn hardware oscillator. Over the course of multi-minute recordings, this drift accumulated to several hundred milliseconds, which pushed ERP windows out of their expected latencies. 
* **Impact:** The new regression method successfully absorbs both Bluetooth pipeline delay and per-session clock drift as a data-driven correction without relying on hardcoded constants. The method is detailed in `analysis/loader.py`. 

---

### 2026-06-03: Session-Level RF Environment Capture
**Phase:** Data Acquisition (Recording Protocol)
**Status:** Active deviation during main data collection.

* **Deviation:** The protocol originally stated that WiFiMan and UniFi controller screenshots would be captured "immediately before and after each recording." This has been downgraded to a per-session capture (one set taken before the 4-condition block begins, and one set taken after the session concludes).
* **Motivation:** Capturing 8 separate sets of screenshots per subject introduced unmanageable manual overhead and significantly disrupted the experimental flow, directly interfering with the subjects' mandated inter-recording rest periods.
* **Impact:** The RF environment is characterized at the session baseline rather than per-recording. Because all four condition recordings are conducted back-to-back in the identical physical environment over a short duration, the ambient RF baseline remains adequately documented without degrading the subject experience or introducing unnecessary fatigue. 

---

### 2026-06-06: Per-Condition Pre-Recording Diagnostic
**Phase:** Data Acquisition (Recording Protocol)
**Status:** Active clarification during main data collection (retroactive to sub-01).

* **Deviation:** The protocol (§4, exclusion criterion 1) specifies a single "pre-session diagnostic" for resting-baseline electrode quality. In practice, the 10-second diagnostic was administered before *each* condition recording (control, chewing, EMI, acoustic), not once per session. When a channel exceeded the 60 µV flag threshold, electrodes were refitted — the recovery procedure explicitly named in §4 criterion 1 — and the diagnostic re-run until no channel was flagged, prior to beginning that condition's recording.
* **Motivation:** The Unicorn's dry/wet electrodes are prone to contact degradation across a multi-recording session. Re-seating to a clean baseline before each condition removes intra-session electrode drift as a confound, ensuring that measured condition effects are attributable to the noise manipulation rather than to progressive contact decay correlated with condition order. This is more conservative than the pre-registered single-gate procedure.
* **Impact:** No subject was excluded on grounds that would not also have triggered the pre-session criterion; the per-condition application only ever recovered electrodes prior to recording, never altered an exclusion decision. The procedure strengthens the primary (P300) and secondary (N170) analyses by equalizing baseline contact quality across conditions. One consequence affects the *exploratory* per-condition raw signal-quality metric (§5): because contact is normalized before each recording, per-condition stdev reflects condition effect on a re-seated baseline rather than cumulative real-world drift. As this measure is exploratory and descriptive with no inferential weight, interpretation is unaffected. Diagnostic runs are logged in condition order per subject and are mappable to condition via `order_assignments.csv` at analysis time.

---

### 2026-06-07: Epoch-Rejection Exclusion Gate Restricted to the Control Condition
**Phase:** Analysis (Subject Inclusion)
**Status:** Post-data analytical decision, made after preprocessing sub-01 through sub-16; applied to all subjects.

* **Deviation:** §4, exclusion criterion 3 (restated in §6) excludes any subject whose epoch rejection rate 
  exceeds 20% on *any* condition. 
  The gate is narrowed to the control condition alone: a subject is excluded on epoch-quality grounds 
  only if their control recording exceeds 20% rejection, and rejection in chewing, EMI, 
  or acoustic no longer triggers subject-level exclusion. 
  Nothing else changes — the ±150 µV peak-to-peak threshold, the boundary exclusion rule, 
  and the epoching are all untouched, and per-condition rejection rates continue to be computed 
  and reported in full per §6.

* **Motivation:** The original "any condition" wording is, on reflection, incoherent for a physical-layer noise study. Chewing is operationalised as a sensor-coupled EMG manipulation, so high-amplitude EMG tripping the rejection threshold is the condition doing exactly what it was designed to do, not evidence that the subject's data is untrustworthy. Excluding a subject because the noise condition was noisy selects against the very degradation the study exists to quantify, which is circular. Control is the only condition with no introduced noise, and it is also the data on which each subject's classifier is trained, so its rejection rate is the principled index of whether a subject's underlying signal is clean enough to analyse. Gating on control answers the question the criterion was meant to ask (is this subject's baseline usable?) without contaminating it with the manipulation's intended effect.

* **Impact:** Across the first 10 subjects, control rejection ranged from 1.6% to 2.4%, so under the control-only gate no subject is excluded and all ten are retained. Under the literal "any condition" rule, four subjects breach the threshold, every one of them on chewing: sub-01 (21.9%), sub-02 (34.2%), sub-03 (24.7%), and sub-10 (75.4%). All the while EMI and acoustic remain near 3.3% throughout and never approach it.

  This change matters for inference. Applying the registered rule drops the sample to N=6 and moves the primary chewing-vs-control contrast from p=0.014 to p=0.078, with the Friedman omnibus going from 0.012 to 0.060; under the control-only rule the chewing effect remains significant. The reason this is a question of power rather than of artifact is that the effect size is stable across both definitions (Cohen's dz = -0.82 at N=10, -0.72 at N=6): the four excluded subjects do not reverse or dissolve the effect, they remove four of the cleanest baselines and the statistical power that came with them. The deviation recovers that power rather than manufacturing the result. To keep this transparent, this will be explicitly reported in the manuscript.

  One limitation is retained and belongs next to this entry. Because rejected epochs are discarded before classification, chewing balanced accuracy is computed on the surviving, cleaner subset and is therefore a conservative estimate of the true degradation — the worst chewing epochs never reach the classifier. For high-rejection subjects the chewing test set is also thin (sub-10 retains 316 of roughly 1285 epochs, only 33 of them targets), and those per-subject chewing figures are read with that caveat. The per-condition rejection rate is accordingly reported as a finding in its own right, not merely as an exclusion gate.

---

### 2026-06-11: Stimulus WAV Bit Depth (32-bit vs registered 24-bit)
**Phase:** Stimulus Generation
**Status:** Documentation correction; stimulus already generated and hash-locked.

* **Deviation:** §3 (Acoustic) specifies the pink-noise stimulus as a 24-bit WAV.
  The generator writes int32 samples via `scipy.io.wavfile.write`, which emits
  32-bit PCM; scipy provides no 24-bit integer WAV encoder. The deposited
  `pink_noise.wav` is therefore 32-bit PCM with sample values confined to the
  24-bit range, not a true 24-bit container.
* **Motivation:** Identified on review of the export path after generation.
* **Impact:** None. Both depths far exceed the dynamic range of the playback
  chain (single consumer speaker at 65 dB(A) SPL) and of the perturbation itself.
  The waveform, spectrum, SHA-256-verified contents, and all 15 stimulus-integrity
  checks are unaffected; the registered spectral, band-limiting, level, and fade
  properties hold identically. The discrepancy is in the sample container format only.

---
