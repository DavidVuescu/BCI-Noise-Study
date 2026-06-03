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