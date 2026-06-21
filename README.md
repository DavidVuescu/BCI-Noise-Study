# BCI Noise Study

A reproducible research pipeline for measuring how environmental noise — mechanical artifact, electromagnetic interference, and broadband acoustic perturbation — degrades P300-based BCI signal quality on the Unicorn Hybrid Black consumer EEG headset.

This is the codebase for the master's dissertation of **Mihai-David Vuescu**, West University of Timișoara, conducted under pre-registration on OSF.

- **Pre-registration:** [osf.io/bng2z](https://osf.io/bng2z/files/ew4xd)
- **Hardware:** Unicorn Hybrid Black (g.tec medical engineering), 8-channel hybrid dry/wet EEG @ 250 Hz, Bluetooth 2.1+EDR
- **Stack:** Python 3.10, MNE-Python, pygame, scikit-learn, statsmodels, numpy/scipy

---

## Research question

To what extent do three categories of physical-layer environmental noise — mechanical artifact via chewing, 2.4 GHz electromagnetic interference, and acoustic perturbation via broadband pink noise — degrade P300 classifier accuracy and face-evoked N170 amplitude on a consumer EEG headset, relative to a controlled baseline condition?

The study is conducted in a residential testing environment as a deliberate ecological-validity choice rather than a compromise. The research question concerns consumer-grade BCI performance in the environments in which such hardware is actually used.

## Paradigm at a glance

- **Task:** fixed-target P300 face-oddball on a 3×3 grid
- **Stimulus:** 60 unique face images, 133 ms flash, 100 ms ISI, frame-locked at 60 Hz
- **Structure:** every recording is divided into three sub-blocks with rotating target cells {0, 4, 8}, decorrelating target identity from retinotopic position
- **Conditions:** control (10 min), chewing, EMI, acoustic (5 min each)
- **Within-subject design:** N = 50 target, up to N = 60 with overrecruitment buffer
- **Primary classifier:** SWLDA (Krusienski et al. 2008), trained on control, tested across conditions
- **Secondary measure:** N170 amplitude at PO7/Oz/PO8

Full design and analysis plan are in the registered protocol on OSF.

---

## Repository layout

```
bci-noise-study/
├── config.yaml                       # all paradigm parameters, one source of truth
├── requirements.txt                  # Python dependencies (pinned)
│
├── src/                              # RECORDING-SIDE code (runs during a session)
│   ├── config.py                     # YAML loader
│   ├── recorder.py                   # threaded UDP receiver for Unicorn UDP Interface
│   ├── sequence.py                   # 3-sub-block flash sequence generator
│   ├── stimulus.py                   # pygame paradigm runner with rest gates & count prompts
│   └── session.py                    # orchestrator: protocol-aware runner + single-recording mode
│
├── analysis/                         # ANALYSIS-SIDE code (runs offline)
│   ├── loader.py                     # Recording dataclass; clock-drift correction via linear fit
│   ├── preprocess.py                 # filter, epoch, baseline-correct, reject → MNE Epochs
│   ├── classifier.py                 # SWLDA stepwise feature selection + LDA
│   ├── plots.py                      # all standard figures (ERP, PSD, confusion matrix, summary)
│   └── _check_signal.py              # text-output ERP diagnostic
│
├── notebooks/
│   ├── 01_pipeline.ipynb             # single-subject workhorse: load → preprocess → classify → plot
│   └── 02_group_results.ipynb        # group-level aggregation across all processed subjects
│
├── protocol/
│   ├── order_assignments/
│   │   ├── generate_order_assignments.py   # reproducible cohort randomization
│   │   └── order_assignments.csv           # locked, audit-trail file (do not edit by hand)
│   └── pink_noise/                   # acoustic-condition stimulus + verifier + SHA-256
│
├── data/
│   ├── raw/                          # per-subject recordings (sub-XX/)
│   └── derived/                      # pipeline outputs, versioned by recipe
│       ├── preprocessing-v2/
│       └── classifier-v2/
│
├── DEVIATIONS.md                     # logged methodological deviations
├── pilot_observations.md             # noted-but-not-acted-on findings from piloting
└── README.md                         # you are here
```

## Architectural principles

The repo is structured around three discipline-imposing rules:

**1. Raw data is sacred.** The recorder writes whatever the hardware sent, in its native form, with no in-flight corrections. Clock-drift correction, filtering, epoching — all happen offline on the analysis side. This lets the same raw data be re-analyzed with different parameters without re-recording subjects.

**2. Folders earn themselves with multiple files.** Subject data lives in per-subject folders (BIDS-style). Derived data is shadowed by pipeline-version tags (`preprocessing-v2/`, `classifier-v2/`) so re-running with different parameters doesn't overwrite earlier outputs.

**3. Pre-registered parameters are constants, not arguments.** Filter cutoffs, epoch windows, rejection thresholds, SWLDA p-values, and similar are hardcoded as module-level constants with docstrings citing the relevant pre-registration section. Changing one requires editing the source and logging a deviation in `DEVIATIONS.md` — the friction is intentional.

## Timing: how marker alignment actually works

This is the load-bearing piece worth understanding.

The Unicorn streams via Bluetooth UDP. Two clocks are in play: the device's hardware counter (jitter-free 4 ms grid, ground truth for sample timing) and the host system clock (used to timestamp stimulus markers from pygame). These two clocks **drift relative to each other at ~1.2–1.4 ms/s**, accumulating to several hundred ms over a multi-minute recording.

A constant-offset alignment cannot absorb this drift. The fix, implemented in `analysis/loader.py`, fits a per-recording linear regression of acquisition time on wall-clock receive times. This absorbs both the Bluetooth pipeline delay and the per-session drift in one data-driven correction, with no hardcoded constants.

The fix is logged as a pre-data deviation in `DEVIATIONS.md`. This discovery was the single most important finding during pipeline development: without it, ERP windows looked at the wrong post-stimulus latencies and the classifier flatlined at chance.

## Quickstart

### Install

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; use source on Linux/Mac
pip install -r requirements.txt
```

### Verify the device pipeline (no subject required)

1. Launch UnicornUDP from Unicorn Suite DevTools, point at `127.0.0.1:1000`, click Open → Start
2. With the headset gelled and worn:
```bash
   python -m src._diagnostic
```
3. Confirm: 250 Hz packet rate, counter increments cleanly by 1, posterior channels (PO7/Oz/PO8) below 30 µV stdev

### Record a full session (all 4 conditions in protocol order)

```bash
python -m src.session --subject 01
```

This reads `protocol/order_assignments/order_assignments.csv`, runs the 4 recordings in the assigned order with ENTER gates between them, and automatically sets the correct target-cell permutation and duration (control = 600 s, all others = 300 s) for each condition. No manual permutation lookup required.

To resume after a crash mid-session:
```bash
python -m src.session --subject 01 --start-from 3
```

To record a single condition (targets and duration still auto-resolved from the CSV):
```bash
python -m src.session --subject 01 --condition control
```

Outputs land in `data/raw/sub-01/` as six files per condition:
`*_eeg.npy`, `*_acqtime.npy`, `*_timestamps.npy`, `*_meta.json`, `*_markers.csv`, `*_session.json`.

### EMI condition

Server (PC, leave running):
```bash
iperf3 -s
```

Client (laptop, start *before* the recording command, leave running through):
```bash
iperf3 -c <PC_IP> -u -b 200M -t 360
```

Confirm UniFi shows 2.4 GHz channel 6 at 40 MHz width with elevated channel utilization. Then run the recording.

### Analysis

The primary analysis interface is Jupyter notebooks:

```bash
jupyter lab
```

Open `notebooks/01_pipeline.ipynb`, set `SUBJECT_ID` at the top, and run all cells. It walks through load → PSD → preprocess all four conditions → ERP → classifier → confusion matrices → one-page summary figure saved to `data/derived/`. Run this the day each participant comes in.

Open `notebooks/02_group_results.ipynb` periodically as subjects accumulate. It auto-discovers all saved `_results.json` files and produces group-level accuracy plots, a subject × condition table, and sensitivity/specificity breakdowns.

For headless verification of the pipeline without a notebook:

```bash
python -m analysis._test_loader              # load + verify alignment
python -m analysis._test_preprocess          # filter, epoch, reject
python -m analysis._test_classifier          # SWLDA, train control, test all
python -m analysis._check_signal             # ERP SNR diagnostic, text output
```

Derived outputs land in `data/derived/preprocessing-v2/sub-XX/` and `data/derived/classifier-v2/sub-XX/`.

## Subject identity discipline

Subject IDs in this repo are pseudonyms: `sub-01`, `sub-02`, … The mapping from real identities to pseudo-IDs is held by the project lead in a private file outside the repository, used solely for subject-withdrawal requests, and destroyed at publication. This is per pre-registration §7 and GDPR.

No real names appear in commit messages, filenames, metadata, derived data, or logs.

## Reproducibility

- `protocol/order_assignments/generate_order_assignments.py` is byte-reproducible from a hardcoded `MASTER_SEED`. Regenerating the CSV from the seed yields an identical file (verifiable via SHA-256).
- `protocol/pink_noise/` contains the generator, independent verifier, generated WAV, and SHA-256 hash for the acoustic stimulus.
- All pre-registered parameters are constants in source, not arguments — methods sections can cite the source file.
- All methodological deviations are logged in `DEVIATIONS.md` with date, reason, and pre-data/post-data status.

## What this study is not

This study does **not** characterize the Unicorn Hybrid Black's absolute performance, generalize to other consumer EEG hardware, or measure cognitive effects of noise on subjects. The conditions are operationalized as *physical-layer* perturbations — sensor-coupled, transmission-coupled, or micro-movement-coupled — not as cognitive load manipulations.

The acoustic condition uses pink noise rather than music specifically to keep all conditions within the physical-layer category and avoid attentional/emotional confounds that lyrical or melodic content would introduce.

## License & contact

This project uses the MIT License. This project's author is...not quite sure what that means.

For questions, contact Mihai-David Vuescu via West University of Timișoara.

---

*Dedicated to my mother and her unyielding support.*
