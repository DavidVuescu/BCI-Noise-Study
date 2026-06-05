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
* **Impact:** No subject was excluded on grounds that would not also have triggered the pre-session criterion; the per-condition application only ever recovered electrodes prior to recording, never altered an exclusion decision. The procedure strengthens the primary (P300) and secondary (N170) analyses by equalizing baseline contact quality across conditions. One consequence affects the *exploratory* per-condition raw signal-quality metric (§5): because contact is normalized before each recording, per-condition stdev reflects condition effect on a re-seated baseline rather than cumulative real-world drift. As this measure is exploratory and descriptive with no inferential weight, interpretation is unaffected. Diagnostic outputs were captured per run but not labeled by condition/attempt for sub-01 through sub-07; a labeled logging format is adopted from the point of this entry forward.

---
