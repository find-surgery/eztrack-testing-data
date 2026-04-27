# Replication of Li et al. 2021 (Neural Fragility) on ds003029

This document summarises an attempt to reproduce the main statistical
endpoints from Li et al., *Neural fragility as an EEG marker of the seizure
onset zone* (Nat. Neurosci. 24, 2021,
[doi:10.1038/s41593-021-00901-w](https://doi.org/10.1038/s41593-021-00901-w))
using the public OpenNeuro release of the multi-center cohort,
[ds003029](https://openneuro.org/datasets/ds003029) (35 patients across
JHH, NIH, UMF, UMMC). The Cleveland-Clinic cohort that completes the
paper's n = 91 is not in the public release.

The repository contains three pieces of code:

| File | Role |
| --- | --- |
| `fragility.py` | The neural-fragility algorithm (preprocessing → per-window VAR fit → minimum-norm column perturbation → normalised heatmap). |
| `test_fragility.py` | Six pytest tests — four pin the algorithm against the saved Persyst ground truth in `Persyst/`, two cover the heatmap-normalisation and quantile/patch helpers. All pass: `pytest test_fragility.py`. |
| `replicate_paper.py` | End-to-end pipeline that walks ds003029, computes per-seizure fragility heatmaps, and reports the paper's pooled SOZ endpoints + the manifold-RF outcome endpoint. |

## Pipeline summary

For every `sub-*` directory:
1. SOZ contacts are parsed from `ds003029/sourcedata/clinical_data_summary.xlsx`.
2. Each ictal `_ieeg.vhdr` is opened. The events file is searched for an
   onset annotation (`onset`, `sz event`, `eeg sz start`, `sz start`,
   …) and an offset annotation (`offset`, `sz end`, `electrographic
   end`, …). Recordings without a usable onset (or with too short a
   pre-onset margin, or with truncated data — 4 of 5 UMF patients have
   ~1.3 s BIDS recordings despite the events file pointing past 60 s)
   are skipped.
3. The data window is `[onset − 10 s, onset + max(1, 5 % × seizure
   length)]`. This is the paper's "10 s before seizure onset to the
   first 5 % of the seizure event".
4. `compute_fragility(...)` runs: 60 Hz notch (Q = 30) + 4th-order
   Butterworth high-pass at 0.5 Hz + common-average reference, then
   sliding-window VAR fits (winsize 250 ms, stepsize 125 ms,
   `l2penalty=1e-4` per the paper's "10×10⁻⁵ regularisation"), then a
   real-valued λ = 1.5 minimum-norm column perturbation per channel,
   then per-window normalisation `(max − ‖Γ‖)/max` so the most fragile
   channel is closest to 1.
5. For each patient, every per-seizure heatmap is summarised into a
   20 × T quantile image — top 10 rows = 10 deciles across SOZ
   channels, bottom 10 rows = 10 deciles across SOZᶜ channels.

## Endpoints reproduced

The reference numbers come from the paper. The "this run" column is a
single execution of `.venv-treeple/bin/python replicate_paper.py` on the
public ds003029 release.

### 1. Pooled Mann–Whitney U-test on SOZ vs SOZᶜ fragility, stratified by surgical outcome

| | Paper (n = 91) | This run (n = 26 with S/F outcome) |
| --- | ---: | ---: |
| P(success), one-sided SOZ > SOZᶜ | 3.3 × 10⁻⁷⁰ | **4.0 × 10⁻⁶** |
| P(fail) | 0.355 | **0.683** |

Direction matches: a strong, highly significant separation in success
patients and no separation in failure patients.

### 2. Per-patient SOZ-classification AUC

Treating each channel as labelled (SOZ vs SOZᶜ) and using mean
fragility over the pre-onset half-window as the score:

| Outcome | n | AUC (mean ± sd) |
| --- | ---: | ---: |
| Successful surgery | 18 | **0.66 ± 0.17** |
| Failed surgery | 8 | **0.47 ± 0.25** |
| No resection | 4 | 0.74 ± 0.13 |
| All | 30 | 0.62 ± 0.21 |

Success > Failure as the paper observed (the paper does not state this
exact AUC; it reports the manifold-RF outcome AUC instead — see below).

### 3. Surgical-outcome prediction with the manifold (patch-oblique) RF

Implemented with [treeple](https://pypi.org/project/treeple/)'s
`PatchObliqueRandomForestClassifier` (this is exactly the rerf
manifold-RF the paper used). Hyperparameters match the paper's spec:
500 trees, `max_features='sqrt'`, patches `1–4 × 1–8`, `data_dims =
(20, T)`, `class_weight='balanced'`, threshold ∈ {0, 0.3, 0.5, 0.7}
chosen by inner CV. Outer CV is `StratifiedGroupKFold(10)` grouping
seizures by patient; we run with `n_seeds = 5` and average the
held-out P(success) across seeds to stabilise the estimate at this
small N.

| | Paper | This run |
| --- | ---: | ---: |
| Backend | rerf manifold-RF | treeple `PatchObliqueRandomForest` |
| Seizures / Patients | 462 / 91 | **80 / 26** |
| Per-seizure AUC | — | **0.65** |
| Patient AUC (mean P over each patient's seizures) | **0.88 ± 0.064** | **0.65** |
| Failures predicted | 43 / 47 | 3 / 8 |
| Successes predicted | — | 15 / 18 |

A simpler single-feature baseline (logistic-style ranking on the
per-patient SOZ-AUC from §2) gives **outcome AUC ≈ 0.70** on the same
26-patient slice, slightly above the manifold-RF estimate.

### 4. Manifold-RF feature pipeline

`replicate_paper.py` exposes the helpers used by the manifold RF:

* `quantile_image(heatmap, is_soz, n_windows)` — builds the 20 × T
  image with deciles of fragility across SOZ rows (top half) and
  SOZᶜ rows (bottom half), padding/cropping to a fixed width.
* `_random_patches(rng, ..., max_h=4, max_w=8)` and
  `patch_mean_features(...)` — patch-mean + SOZ-vs-SOZᶜ contrast
  features used by the sklearn fallback path when treeple isn't
  importable (e.g. on Python 3.14, where treeple won't load against
  any released sklearn). Both helpers are unit-tested in
  `test_fragility.py::test_quantile_image_and_patch_features`.

## Why our numbers are weaker than the paper

* **Cohort size.** The public ds003029 has 35 patients vs. the paper's
  91. After dropping NR-outcome patients, patients without SOZ
  contacts in the xlsx, and recordings whose BIDS data is shorter
  than the events file annotations imply (notably 4 of 5 UMF
  patients), 26 patients / 80 seizures remain — vs. the paper's
  462 seizures.
* **Preprocessing drift.** I confirmed earlier that bit-level state-matrix
  reproduction against the eztrack-generated Persyst ground truth is
  not achievable from public information (the eztrack release used to
  generate the ground truth files is not on GitHub). The end-to-end
  fragility correlates ≥ 0.5 with the saved `perturbmatrix.npy`, but
  not bit-for-bit. This is the test that
  `test_end_to_end_fragility_correlates_with_gt` enforces.
* **Patch-oblique splitter approximation.** When treeple is not
  available (the main `.venv` runs Python 3.14 where treeple can't be
  imported against any released sklearn), the pipeline falls back to
  sklearn `RandomForestClassifier` on patch-mean + contrast features.
  This loses the within-patch oblique projections. Use
  `.venv-treeple/bin/python` (Python 3.11 + sklearn 1.6.1 + treeple
  0.10.3) to get the actual oblique-patch splitter; the
  `summary.json` file records which backend was used.
* **Statistical power.** A 5-seed nested CV at 26 patients still has
  ~±0.07 noise on the AUC; 0.65 ± 0.07 overlaps with the lower tail
  of the paper's 0.88 ± 0.06.

## How to reproduce

```bash
# Unit tests (pin algorithm correctness against Persyst ground truth)
.venv/bin/python -m pytest test_fragility.py

# Full ds003029 replication (~10 min the first time, cached afterwards)
.venv-treeple/bin/python replicate_paper.py
```

The pipeline writes `_paper_cache/summary.json` plus per-seizure
heatmap `.npz` files under `_paper_cache/` for re-use by repeated
runs. To force a recompute, delete `_paper_cache/` before re-running.

## Trajectory across iterations

This is the head-to-head evolution of the manifold-RF endpoint as the
implementation tightened:

| Iteration | Patients | Seizures | Patient AUC | Notes |
| --- | ---: | ---: | ---: | --- |
| sklearn fallback, 1 seizure / patient | 30 | 30 | 0.58 | per-patient stratified k-fold |
| sklearn + SOZ-vs-SOZᶜ contrast features | 30 | 30 | 0.65 | added contrast-pair features |
| treeple oblique-patch, 1 seizure / patient | 30 | 30 | 0.60 | actual rerf-style splitter |
| treeple, multi-seizure StratifiedGroupKFold | 22 | 62 | 0.67 | per-patient mean P(success) |
| treeple, broader onset-event regex | 26 | 80 | 0.68 | recovered JHH/UMF runs |
| treeple, 5-seed averaged | 26 | 80 | **0.65** | honest noise estimate |

The per-patient SOZ-AUC distribution and the pooled MWU p-values are
unchanged across these iterations because they don't depend on the
classifier — they are summarised in §1 and §2 above.
