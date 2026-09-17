# Beyond Keystroke Metrics

**A 3D Motion Capture Framework for Smartphone Typing Kinematics**

This repository contains the analysis code and materials for a methods paper introducing a smartphone motion-tracking pipeline for studying typing kinematics. Instead of treating a phone only as a keystroke logger, the pipeline uses the device's own inertial sensors to capture the micro-oscillations produced while typing.

The contribution here is the **pipeline**, not a substantive finding about any particular population. The included pilot study (N = 36, Digital Natives vs. Digital Immigrants) is a proof-of-concept illustration of what the framework can and cannot detect.

---

## What the pipeline extracts

Six kinematic descriptors are derived from smartphone micro-oscillations during typing:

| Signal | Descriptors |
| --- | --- |
| Motion | dominant frequency, RMS amplitude, peak-to-peak range |
| Tilt | dominant frequency, RMS amplitude, peak-to-peak range |

---

## Validation logic

The central validation argument is a dissociation between empirical and synthetic data:

| Level | Empirical | Synthetic |
| --- | --- | --- |
| Between-person (N = 36) | null | null |
| Within-person (N = 464 trials) | LOSO RF ≈ 55–56%, κ ≈ 0.12–0.13, significant | ≈ 51–52%, κ ≈ 0.03–0.04, non-significant |

The synthetic null is **engineered, not approximated**: kinematic ground-truth values are drawn from a single shared distribution with no reference to group assignment, and labels are attached by random permutation. This is a single-instance demonstration of appropriate null behaviour, not a repeated-sampling Type I error estimate.

---

## Repository structure

```
.
├── BM_BLMM_v2_0.R                      # Bayesian linear mixed models (single-predictor)
├── withinSubject_analyses_empirical.R  # PCA / LDA / LOSO random forest — empirical data
├── withinSubject_analyses_synth.R      # Same analyses on synthetic data (validation)
├── data/                               # Aggregated and trial-level CSVs
└── archive/                            # Superseded scripts, kept for provenance
    ├── BM_BLMM_v1_2.R
    ├── cfa_script_v2_0.R
    └── manifold_test.R
```

Scripts at the repository root are the current (v3.0) analysis set used in the manuscript. Everything under `archive/` is retained for transparency only and should not be re-run.

---

## Data files

Between- and within-person analyses use **different files** and must be loaded separately:

- `synth_aggregated.csv` — participant-level; the `file` column is an index 1–36.
- `synth_trial_level.csv` — trial-level; the `file` column holds trial strings and `participant_id` is numeric 1–36.

Two conventions worth knowing before touching the code:

- Participant ID is stored as `file`; platform is stored as `environment`.
- Empirical trial-level data is already z-scored — do not re-standardise it.

---

## Requirements

**R** (primary analysis language)

```r
install.packages(c("brms", "loo", "ggplot2", "randomForest", "caret", "dplyr", "tidyr"))
```

`brms` requires a working Stan toolchain (RStan or cmdstanr).

**MATLAB** is used upstream for signal processing and synthetic data generation; that pipeline is documented separately.

---

## Reproducing the analyses

Run the scripts from the repository root, in any order — they are independent:

```r
source("BM_BLMM_v2_0.R")
source("withinSubject_analyses_empirical.R")
source("withinSubject_analyses_synth.R")
```

Random forest seeds are set inside the scripts, so LOSO results are reproducible. Figures are written to the output directory defined at the top of each script.

---

## Status

Manuscript in preparation for submission. Some materials (OSF archive link, supplementary tables) will be added here on submission.

## Authors

- **Aaron** — R-based statistical analyses
- **Alireza** — MATLAB pipeline, synthetic data generation, recovery metrics, signal synthesis method

## Citation

A citation entry will be added once the preprint is posted.

## License

MIT — see `LICENSE`.
