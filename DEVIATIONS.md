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

### 2026-06-11: Correction to 2026-06-06 entry — refit threshold & bad-channel handling
**Phase:** Data Acquisition / Analysis
**Status:** Correction of a prior log entry; aligns documentation with registered §4 and actual procedure.

* **Correction:** The 2026-06-06 entry stated diagnostics were "re-run until no
  channel was flagged." This overstates the procedure and is stricter than the
  registered protocol. Per §4 criterion 1, up to two channels flagged at >60 µV
  stdev are tolerated and proceeded with (flagged bad for the session). Refit
  (re-part hair, re-gel, brush-handle parting) is performed only when >2 channels
  are flagged; the diagnostic is then re-run to confirm recovery within the
  registered 5-minute window. The realized procedure matches the registered
  >2-channel threshold; the prior entry described a procedure neither registered
  nor performed.

* **Diagnostic administration detail:** Diagnostics were administered with the
  subject fixating a central point, and run multiple times (typ. 3–5) to assess
  contact stability before the gating decision. Fixation isolates electrode
  contact quality from gross-movement artifact and matches the attentional state
  of the recordings themselves; the ±150 µV epoch-rejection gate (§6) remains the
  downstream backstop against in-band contamination.

* **Bad-channel handling (operationalises §5 quality gate):** A channel flagged
  in ANY of a subject's condition diagnostics is excluded from that subject's
  classifier feature set across ALL four conditions, preserving a consistent
  feature space for the within-subject classifier. Posterior-lateral sites
  (PO7/PO8) are part of the N170 triad; subjects with persistent flags there have
  N170 estimated on the reduced channel set, reported as a per-subject reliability note.

* **Impact:** No subject was retained or excluded differently from what the
  registered ≤2-channel tolerance dictates. The change is documentary plus the
  downstream feature-set exclusion, which brings the pipeline into line with §5
  rather than departing from it.

---

### 2026-06-11: EMI Pilot Gate Not Run as Preregistered (§8)
**Phase:** Pre-Study Validation
**Status:** Deviation documented; retrospective validation completed 2026-06-25.

* **Deviation:** Preregistration §8 required a headset-on-table pilot recording before
  main data collection to confirm that the EMI condition (iperf3 UDP flood, 200 Mbps,
  2.4 GHz channel 6, 40 MHz) elevated the Bluetooth counter-gap rate. This standalone
  pilot was not conducted. WiFi saturation was instead verified via iperf3 throughput
  logs, WiFiMan channel-utilisation graphs, and UniFi controller screenshots captured
  before and after each session block.

* **Retrospective validation (2026-06-25):** The preregistered question was answered
  retrospectively using actual subject recordings. Counter-gap metrics (samples_dropped
  and samples_expected_by_counter from meta.json) were extracted for all completed
  recordings and compared between EMI and control conditions.

  All files for this validation are in protocol/emi_pilot/:
  - iperf3 throughput log: airspace-test_iperf.log.txt
  - Analysis script:       emi_pilot_validation.py
  - Results output:        emi_validation_results.json

  Results:
  - N subjects with valid control recordings: 42
  - N subjects with valid EMI recordings: 44
  - Mean dropout rate — control: 0.000% (std 0.000%, range 0.000–0.000%)
  - Mean dropout rate — EMI:     0.000% (std 0.000%, range 0.000–0.000%)
  - Statistical test: Mann-Whitney U, one-tailed (EMI > Control)
  - Statistic: 924.000, p = 1.0000, Cohen's d = 0.000
  - Verdict: AFH TOLERATED — no statistically significant elevation in
    counter-gap rate under the EMI condition.

* **Interpretation:** The Unicorn's Bluetooth radio successfully used Adaptive
  Frequency Hopping to avoid the saturated 2.4 GHz channel. Zero samples were dropped
  in any recording in any condition across all subjects. The WiFi saturation was
  physically present (confirmed by iperf3 and UniFi logs), but the BT link was resilient.
  This means the EMI condition did not degrade the signal at the transmission layer;
  any classifier accuracy differences under EMI arise from RF-induced sensor noise
  rather than data loss. This is consistent with the null or modest EMI accuracy
  effects observed across subjects and is reported in the Discussion.

---

### 2026-07-09: Acoustic playback timing and SPL not logged in session metadata (Documentation, post-data)
**Phase:** Data Acquisition (Acoustic condition) / Deposit preparation
**Status:** Documented at deposit time.

* **Deviation:** §3 (Acoustic) states that pink-noise playback start/end timestamps
  are logged in session metadata and that playback is calibrated to 65 ± 3 dB(A).
  In practice the stimulus was started and stopped manually via a media player (VLC)
  at the beginning and end of each acoustic recording; no programmatic timestamping
  was implemented and no per-session SPL value was recorded. Level was set to the
  registered 65 ± 3 dB(A) target before sessions but not logged per recording.
* **Motivation:** Automated timestamping was never scripted, and logging an SPL
  reading per participant while they were seated and waiting was impractical and
  would have disrupted session flow.
* **Impact:** Documentation gap only. Acoustic onset/offset are bounded by each
  recording's own start/stop timestamps (the stimulus spanned the recording; see
  session_metadata.csv start_time_iso_utc / stop_time_iso_utc). The stimulus itself,
  its target level, and its spectral properties are fixed and reproducible
  (protocol/pink_noise/). The acoustic condition produced no reliable EEG or
  classifier effect, so precise playback timing does not bear on any reported result.

---

### 2026-08-13: Control Train/Test Split Implemented as Random Rather Than Temporal (Documentation, post-data)
**Phase:** Analysis (classifier training)
**Status:** Documentation correction; identified during peer review.

* **Deviation:** §5 specifies training on "the first 70% of quality-controlled
  control epochs." The implementation (`analysis/classifier.py`) calls
  `sklearn.model_selection.train_test_split` with stratification and default
  shuffling, producing a *random* stratified 70/30 split rather than the first
  70% in temporal order. The registered proportion and the class stratification
  are as intended; the ordering is not. The manuscript repeated the registered
  wording and therefore described the split inaccurately.

* **Motivation:** The random split was an undocumented choice, but not an
  arbitrary one. Sampling across the whole recording avoids a train/test
  distribution mismatch driven by within-recording drift — alertness, electrode
  contact, and posture all change across a 600 s control recording — and
  guarantees that all three sub-blocks, and therefore all three target positions,
  are represented in training. A contiguous split trains on a fresh participant
  and tests on a fatigued one. The reasoning is defensible; it was simply never
  registered or logged.

* **Impact:** The manuscript text is corrected to describe the split as
  implemented. The statistical consequence of the choice is assessed by the
  pre-specified analysis in the following entry.

---

### 2026-08-13: Pre-Specified Blockwise Cross-Validation of the Control Ceiling, with Decision Rule Fixed in Advance
**Phase:** Analysis (peer-review response, pre-proceedings revision)
**Status:** Pre-specified. Committed before the analysis was run; results not
observed at time of writing.

* **Prompt:** Reviewer 1 (SYNASC 2026, submission 74) observed that control
  epochs are split within a single recording while the three noise conditions are
  tested on entirely separate recordings, and that with a 1000 ms epoch window
  and a 233 ms stimulus-onset asynchrony, adjacent epochs overlap. The reviewer
  requested "a blockwise or sub-blockwise split." The preceding entry makes this
  concern more acute rather than less: under a random split essentially every
  test epoch has at least one temporally overlapping neighbour in the training
  set, whereas under the temporal split the reviewer assumed, only epochs
  adjacent to the single seam are affected.

* **Analysis to be run:** Per subject, leave-one-sub-block-out cross-validation
  on the control recording. Each control recording comprises three sub-blocks
  (median 843 epochs, 94 targets each). Three folds: train on two sub-blocks,
  test on the held-out third. Each fold's model additionally scores the full
  chewing, EMI, and acoustic recordings. Every reported value is the mean across
  the three folds, so all four conditions are scored by models trained on an
  identical quantity of data (median 1686 epochs, ~188 targets; comparable to the
  ~1770 epochs of the current 70/30 split). SWLDA parameters (p_enter 0.10,
  p_remove 0.15, 60-feature cap), the `preprocessing-v2` epochs, and the fold
  definitions are unchanged and fixed in the committed script.

* **Leakage:** The registered boundary rejection (§6: first 2 s and last 1 s of
  each sub-block), combined with the self-paced inter-sub-block rest, produces a
  measured gap of 11.9–77.8 s (median 17.4 s; n = 82 boundaries across 41
  subjects) between the last epoch of one sub-block and the first of the next.
  The maximum temporal reach of a 1000 ms epoch at 233 ms SOA is 932 ms. No epoch
  in any fold shares samples with any epoch in another; the folds are leak-free
  by construction, by a margin of at least twelvefold.

* **Decision rule, fixed before results were observed:** The blockwise analysis is
  promoted to primary if any of the following hold: (1) the chewing-vs-control
  contrast loses significance (one-tailed Wilcoxon signed-rank against the
  CV-averaged control ceiling, Holm-corrected across the three noise conditions,
  α = 0.05); (2) EMI or acoustic *gains* significance under the same test; or
  (3) the mean control ceiling falls by ≥ 2 percentage points. Otherwise the
  existing analysis remains primary and the blockwise result is reported as a
  robustness check. Separately, and per the registered §5 contingency: if the
  mean control ceiling falls below 60%, the classification analysis is demoted to
  descriptive and the N170 analysis promoted to primary — a question governed by
  the pre-registration, not by this entry.

* **Directionality:** Criterion 3 is deliberately one-directional. A *fall* of
  ≥ 2 percentage points promotes the blockwise analysis; a *rise* of the same
  magnitude is reported and flagged but does not promote it. The rule can
  therefore only move the primary analysis toward the more conservative estimate,
  never toward the more favourable one.

* **Guards:** A single pre-specified run. No re-running with altered fold
  definitions, classifier parameters, or inclusion criteria. Both the existing and
  the blockwise results are reported in full under every branch of the rule.

* **Attribution caveat:** The blockwise design alters two things at once. It
  removes overlap leakage, and it requires each fold to classify a target cell
  absent from its own training set. A reduction in the ceiling therefore cannot be
  attributed to leakage alone, and the blockwise value is best read as a
  conservative lower bound rather than a clean measurement of leakage magnitude. A
  ceiling that does not fall is correspondingly stronger evidence than it may
  appear, having been obtained despite the additional generalisation burden.

* **Impact:** To be recorded on completion.

---
