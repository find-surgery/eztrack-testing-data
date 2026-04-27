"""Replicate the SOZ-discrimination endpoints from Li et al. 2021 on ds003029.

For each patient (1) parses SOZ contacts from the clinical Excel summary,
(2) loads the first ictal run, (3) extracts a [-10s, +10s] window around the
first electrographic seizure-onset annotation, (4) runs ``compute_fragility``,
(5) writes the per-channel mean fragility and SOZ label to a per-patient
parquet file. A summary step then reproduces the paper's two main pooled
endpoints:

* Pooled Mann-Whitney U-test on SOZ vs SOZ^C fragility, stratified by surgical
  outcome (paper: P(success) = 3.326e-70, P(fail) = 0.355).
* Per-patient AUC for SOZ-vs-SOZ^C from raw fragility scores (paper:
  AUC = 0.88 ± 0.064 over a 10-fold nested CV with a manifold RF).

Note: The paper's AUC is from a 10-fold nested-CV manifold-Random-Forest
classifier predicting **outcome** from heatmap quantile features; that exact
pipeline (rerf manifold RF) is out of scope here. Instead we report the
simpler per-patient SOZ-classification AUC from raw mean fragility, which
captures the same underlying signal the manifold RF exploits and whose
distribution should track the paper's reported value.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

import fragility as fr

ROOT = Path(__file__).resolve().parent
DS = ROOT / "ds003029"
CACHE = ROOT / "_paper_cache"
CACHE.mkdir(exist_ok=True)


# ---------- patient metadata ----------------------------------------------

def _normalize(s: str) -> str:
    s = re.sub(r"[\s_-]", "", str(s).lower())
    m = re.match(r"^([a-z]+)(\d+)$", s)
    return m.group(1) + str(int(m.group(2))) if m else s


def parse_soz(s) -> list[str]:
    """Expand 'PD1-4; AD1-4; ATT1-2' to individual channel names."""
    if not isinstance(s, str):
        return []
    out: list[str] = []
    for ch in re.split(r"[;,:]", s):
        ch = ch.strip()
        if not ch:
            continue
        m = re.match(r"^([A-Za-z]+)(\d+)\s*-\s*(?:[A-Za-z]+)?(\d+)$", ch)
        if m:
            prefix, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
            if lo <= hi:
                out.extend(f"{prefix}{i}" for i in range(lo, hi + 1))
        else:
            m2 = re.match(r"^([A-Za-z]+)(\d+)$", ch)
            if m2:
                out.append(ch)
    return out


def load_clinical() -> pd.DataFrame:
    df = pd.read_excel(DS / "sourcedata/clinical_data_summary.xlsx")
    df["_id"] = df["dataset_id"].astype(str).apply(_normalize)
    return df


def patient_records() -> list[dict]:
    df = load_clinical()
    sub_dirs = sorted(d for d in os.listdir(DS) if d.startswith("sub-"))
    records = []
    for sub in sub_dirs:
        n = _normalize(sub[4:])
        rows = df[df["_id"] == n]
        if not len(rows):
            continue
        row = rows.iloc[0]
        soz = parse_soz(row["soz_contacts"])
        records.append({
            "sub": sub,
            "outcome": str(row["outcome"]).strip(),
            "engel": row["engel_score"],
            "site": str(row["clinical_center"]).strip(),
            "soz": [s.lower() for s in soz],
        })
    return records


# ---------- per-patient feature extraction --------------------------------

ONSET_RE = re.compile(
    r"\bonset\b|"
    r"eeg\s*onset|"
    r"sz\s*onset|"
    r"sz\s*event|"
    r"eeg\s*sz\s*start|"
    r"\bsz\s*start\b",
    re.IGNORECASE)

OFFSET_RE = re.compile(
    r"\boffset\b|"
    r"sz\s*end|"
    r"eeg\s*sz\s*end|"
    r"electrographic\s*end|"
    r"\bend\b\s*$",  # bare 'end' at line end (rare)
    re.IGNORECASE)


def find_ictal_runs(sub: str) -> list[Path]:
    return sorted((DS / sub / "ses-presurgery/ieeg").glob(
        "*task-ictal*ieeg.vhdr"))


def find_onset_seconds(events_tsv: Path) -> float | None:
    if not events_tsv.exists():
        return None
    ev = pd.read_csv(events_tsv, sep="\t")
    mask = ev["trial_type"].astype(str).str.contains(ONSET_RE, na=False)
    if not mask.any():
        return None
    return float(ev.loc[mask, "onset"].iloc[0])


def find_seizure_window(events_tsv: Path) -> tuple[float, float] | None:
    """Return (onset_sec, offset_sec) — onset and earliest offset event after
    onset. ``None`` if no usable annotation pair is found."""
    if not events_tsv.exists():
        return None
    ev = pd.read_csv(events_tsv, sep="\t")
    tt = ev["trial_type"].astype(str)
    on = ev.loc[tt.str.contains(ONSET_RE, na=False), "onset"]
    off = ev.loc[tt.str.contains(OFFSET_RE, na=False), "onset"]
    if on.empty:
        return None
    onset = float(on.iloc[0])
    later_off = off[off > onset]
    if later_off.empty:
        return None
    return onset, float(later_off.iloc[0])


def good_analysis_channels(channels_tsv: Path) -> list[str]:
    chs = pd.read_csv(channels_tsv, sep="\t")
    keep = chs[(chs["type"].str.upper().isin(["ECOG", "SEEG"]))
               & (chs["status"] == "good")]
    return keep["name"].astype(str).tolist()


def load_window(vhdr: Path, t_onset: float, pre: float = 10.0,
                post: float = 10.0) -> tuple[np.ndarray, list[str], float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True,
                                          verbose=False)
    sfreq = float(raw.info["sfreq"])
    good = good_analysis_channels(Path(str(vhdr).replace(
        "_ieeg.vhdr", "_channels.tsv")))
    good = [c for c in good if c in raw.ch_names]
    raw.pick(good)
    start = max(0, int((t_onset - pre) * sfreq))
    stop = min(raw.n_times, int((t_onset + post) * sfreq))
    data = raw.get_data(start=start, stop=stop) * 1e6
    return data, raw.ch_names, sfreq


def _process_one_run(vhdr: Path, rec: dict, run_npz: Path,
                      pre_sec: float = 10.0,
                      seizure_frac: float = 0.05,
                      min_post_sec: float = 1.0,
                      ) -> dict | None:
    """Compute the fragility heatmap for the paper's analysis window:
    ``[onset - pre_sec, onset + max(min_post_sec, seizure_frac * length)]``.
    Falls back to ``[onset - pre_sec, onset + 5s]`` when no offset
    annotation is present."""
    ev_path = Path(str(vhdr).replace("_ieeg.vhdr", "_events.tsv"))
    win = find_seizure_window(ev_path)
    if win is None:
        onset_t = find_onset_seconds(ev_path)
        if onset_t is None or onset_t < pre_sec / 2:
            return None
        post = 5.0
    else:
        onset_t, offset_t = win
        if onset_t < pre_sec / 2:
            return None
        post = max(min_post_sec, seizure_frac * (offset_t - onset_t))
    try:
        data, ch_names, sfreq = load_window(vhdr, onset_t,
                                             pre=pre_sec, post=post)
    except Exception as e:
        print(f"    skip {vhdr.name}: load failed ({e})", flush=True)
        return None
    if data.shape[1] < 500:
        return None
    out = fr.compute_fragility(data, sfreq=sfreq)
    frag = out["fragility"].astype(np.float32)
    soz_set = set(rec["soz"])
    is_soz = np.array([c.lower() in soz_set for c in ch_names])
    if is_soz.sum() == 0 or (~is_soz).sum() == 0:
        return None
    np.savez(run_npz, heatmap=frag, is_soz=is_soz,
             ch_names=np.array(ch_names))
    return {"run": vhdr.name, "n_ch": int(frag.shape[0]),
            "n_win": int(frag.shape[1]), "n_soz": int(is_soz.sum()),
            "post_sec": float(post)}


def compute_patient_features(rec: dict) -> pd.DataFrame | None:
    """Run fragility over **every** ictal run with an onset annotation,
    cache one heatmap .npz per run, and return a per-channel summary
    parquet for the per-patient SOZ-AUC analysis (computed on the *first*
    valid run, matching the paper's "snapshot around onset")."""
    cache_path = CACHE / f"{rec['sub']}.parquet"
    runs = find_ictal_runs(rec["sub"])
    if not runs:
        print(f"  {rec['sub']}: no ictal runs", flush=True)
        return None

    run_infos = []
    first_npz = None
    for vhdr in runs:
        run_npz = CACHE / f"{rec['sub']}__{vhdr.stem}.npz"
        if run_npz.exists():
            with np.load(run_npz, allow_pickle=True) as z:
                run_infos.append({"run": vhdr.name,
                                   "n_ch": int(z["heatmap"].shape[0]),
                                   "n_win": int(z["heatmap"].shape[1]),
                                   "n_soz": int(z["is_soz"].sum())})
        else:
            info = _process_one_run(vhdr, rec, run_npz)
            if info is not None:
                run_infos.append(info)
        if first_npz is None and run_npz.exists():
            first_npz = run_npz

    if not run_infos:
        print(f"  {rec['sub']}: no usable ictal runs", flush=True)
        return None
    print(f"    {rec['sub']}: {len(run_infos)} usable runs", flush=True)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    with np.load(first_npz, allow_pickle=True) as z:
        frag = z["heatmap"]
        is_soz = z["is_soz"].astype(bool)
        ch_names = list(z["ch_names"])
    half = max(1, frag.shape[1] // 2)
    df = pd.DataFrame({
        "sub": rec["sub"],
        "outcome": rec["outcome"],
        "site": rec["site"],
        "ch": ch_names,
        "is_soz": is_soz,
        "fragility": frag[:, :half].mean(axis=1).astype(float),
        "run": first_npz.stem.split("__", 1)[1],
    })
    df.to_parquet(cache_path, index=False)
    # Also keep the legacy `<sub>_heatmap.npz` symlink-equivalent for the
    # existing per-patient pipeline / unit tests.
    legacy = CACHE / f"{rec['sub']}_heatmap.npz"
    if not legacy.exists():
        np.savez(legacy, heatmap=frag, is_soz=is_soz,
                 ch_names=np.array(ch_names))
    return df


# ---------- statistical endpoints -----------------------------------------

def pooled_mwu(features: pd.DataFrame) -> dict:
    from scipy.stats import mannwhitneyu

    out = {}
    for outcome in ("S", "F"):
        sub = features[features["outcome"] == outcome]
        if sub.empty:
            continue
        soz = sub.loc[sub["is_soz"], "fragility"].to_numpy()
        non = sub.loc[~sub["is_soz"], "fragility"].to_numpy()
        if soz.size == 0 or non.size == 0:
            continue
        u, p = mannwhitneyu(soz, non, alternative="greater")
        out[outcome] = {
            "n_soz": int(soz.size), "n_nonsoz": int(non.size),
            "mean_soz": float(soz.mean()), "mean_nonsoz": float(non.mean()),
            "U": float(u), "p_one_sided_greater": float(p),
        }
    return out


QUANTILES = np.arange(0.1, 1.001, 0.1)  # 10%, 20%, ..., 100%


def quantile_image(heatmap: np.ndarray, is_soz: np.ndarray,
                   n_windows: int) -> np.ndarray:
    """Build the 20×n_windows quantile image for one patient.

    Top 10 rows are 10 quantiles of fragility across SOZ channels,
    bottom 10 rows are 10 quantiles across SOZ^C channels. Time axis is
    truncated/zero-padded to ``n_windows`` so all patients share a fixed
    image shape (the manifold RF needs one).
    """
    soz = heatmap[is_soz]
    non = heatmap[~is_soz]
    if soz.size == 0 or non.size == 0:
        return None
    q_soz = np.quantile(soz, QUANTILES, axis=0)  # (10, n_win)
    q_non = np.quantile(non, QUANTILES, axis=0)  # (10, n_win)
    img = np.vstack([q_soz, q_non])               # (20, n_win)
    if img.shape[1] >= n_windows:
        img = img[:, :n_windows]
    else:
        pad = np.zeros((img.shape[0], n_windows - img.shape[1]),
                       dtype=img.dtype)
        img = np.hstack([img, pad])
    return img.astype(np.float32)


def thresholded_image(img: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(img >= threshold, img, 0.0).astype(np.float32)


def assemble_quantile_dataset(records: list[dict], n_windows: int = 80,
                              per_seizure: bool = True
                              ) -> tuple[np.ndarray, np.ndarray,
                                         list[str], list[str]]:
    """Stack quantile images and outcome labels (S=1, F=0).

    With ``per_seizure=True`` (default, matches the paper's training setup)
    every ictal run becomes its own sample with the patient's outcome
    label. ``per_seizure=False`` falls back to one sample per patient
    (using the first cached run's heatmap).

    Only patients with outcome S/F are included. Returns
    ``(X, y, sub_ids, run_ids)`` with ``X`` of shape (n_samples, 20*n_windows)
    and ``sub_ids`` the patient id per sample (used for group-aware CV).
    """
    X, y, subs, runs = [], [], [], []
    for rec in records:
        if rec["outcome"] not in ("S", "F"):
            continue
        if per_seizure:
            run_files = sorted(CACHE.glob(f"{rec['sub']}__*.npz"))
        else:
            legacy = CACHE / f"{rec['sub']}_heatmap.npz"
            run_files = [legacy] if legacy.exists() else []
        for npz in run_files:
            with np.load(npz, allow_pickle=True) as z:
                img = quantile_image(z["heatmap"], z["is_soz"].astype(bool),
                                     n_windows)
            if img is None:
                continue
            X.append(img.ravel())
            y.append(1 if rec["outcome"] == "S" else 0)
            subs.append(rec["sub"])
            runs.append(npz.stem)
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=int), subs, runs)


def _random_patches(rng: np.random.Generator, n_patches: int,
                    row_lo: int, row_hi: int, w: int,
                    max_h: int, max_w: int,
                    min_h: int = 1, min_w: int = 1
                    ) -> list[tuple[int, int, int, int]]:
    """Sample random rectangles ``(r0, r1, c0, c1)`` whose row range lies
    within ``[row_lo, row_hi)`` and column range within ``[0, w)``."""
    h = row_hi - row_lo
    patches = []
    for _ in range(n_patches):
        ph = int(rng.integers(min_h, min(max_h, h) + 1))
        pw = int(rng.integers(min_w, max_w + 1))
        r0 = int(rng.integers(0, h - ph + 1)) + row_lo
        c0 = int(rng.integers(0, w - pw + 1))
        patches.append((r0, r0 + ph, c0, c0 + pw))
    return patches


def patch_mean_features(images: np.ndarray, data_dims: tuple,
                         patches: list,
                         contrast_pairs: list | None = None
                         ) -> np.ndarray:
    """Patch-mean features plus optional SOZ-vs-SOZC contrast features.

    ``patches`` are evaluated as plain means. ``contrast_pairs`` are
    pairs of patches ``(p_soz, p_non)`` whose feature is
    ``mean(p_soz) - mean(p_non)`` — this lets the downstream RF see the
    SOZ-vs-SOZ^C contrast that an oblique split inside a single patch
    spanning both halves of the image would learn.
    """
    h, w = data_dims
    imgs = images.reshape(-1, h, w)
    feats = np.empty((imgs.shape[0],
                      len(patches) + (len(contrast_pairs) if contrast_pairs else 0)),
                     dtype=np.float32)
    for k, (r0, r1, c0, c1) in enumerate(patches):
        feats[:, k] = imgs[:, r0:r1, c0:c1].mean(axis=(1, 2))
    if contrast_pairs:
        off = len(patches)
        for k, (ps, pn) in enumerate(contrast_pairs):
            (sr0, sr1, sc0, sc1) = ps
            (nr0, nr1, nc0, nc1) = pn
            feats[:, off + k] = (imgs[:, sr0:sr1, sc0:sc1].mean(axis=(1, 2))
                                  - imgs[:, nr0:nr1, nc0:nc1].mean(axis=(1, 2)))
    return feats


def manifold_rf_nested_cv(records: list[dict], n_windows: int = 80,
                          n_outer: int = 10,
                          random_state: int = 0,
                          backend: str = "auto",
                          n_patches: int = 1000,
                          per_seizure: bool = True,
                          n_seeds: int = 1,
                          ) -> dict:
    """10-fold nested CV with the paper's manifold (patch-oblique) RF.

    ``backend='treeple'`` uses
    :class:`treeple.PatchObliqueRandomForestClassifier`, which the paper
    used (paper hyperparameters: 500 trees, max_features='sqrt' ≈ feature
    combinations 1.5, patch dims 1–4 × 1–8, image dims 20×T). When
    treeple isn't importable we fall back to a sklearn
    :class:`~sklearn.ensemble.RandomForestClassifier` on patch-mean +
    SOZ-vs-SOZ^C contrast features, which captures the same
    spatiotemporal structure but loses the within-patch oblique splits.

    Inner CV picks the heatmap threshold from {0, 0.3, 0.5, 0.7}.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    X, y, subs, runs = assemble_quantile_dataset(
        records, n_windows=n_windows, per_seizure=per_seizure)
    if len(X) < n_outer or y.sum() == 0 or (1 - y).sum() == 0:
        return {"error": "insufficient labeled data", "n": int(len(X))}
    subs_arr = np.asarray(subs)

    data_dims = (20, n_windows)
    treeple_clf = None
    if backend in ("auto", "treeple"):
        try:
            from treeple import PatchObliqueRandomForestClassifier
            treeple_clf = PatchObliqueRandomForestClassifier
        except Exception as e:
            if backend == "treeple":
                raise
            print(f"  treeple unavailable ({e!s}); falling back to "
                  f"sklearn RF on patch-mean features", flush=True)

    thresholds = [0.0, 0.3, 0.5, 0.7]
    if treeple_clf is not None:
        rf_kwargs = dict(
            n_estimators=500, max_features="sqrt", min_samples_split=2,
            max_patch_dims=(4, 8), min_patch_dims=(1, 1),
            dim_contiguous=(True, True), data_dims=data_dims,
            class_weight="balanced", n_jobs=-1,
        )

        def make_features(Xflat: np.ndarray, t: float) -> np.ndarray:
            return (np.where(Xflat >= t, Xflat, 0.0).astype(np.float32)
                    if t > 0 else Xflat)

        def make_clf(seed):
            return treeple_clf(random_state=seed, **rf_kwargs)
    else:
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(random_state)
        soz_patches = _random_patches(rng, n_patches // 3, 0, 10,
                                       n_windows, max_h=4, max_w=8)
        non_patches = _random_patches(rng, n_patches // 3, 10, 20,
                                       n_windows, max_h=4, max_w=8)
        contrast_pairs = list(zip(
            _random_patches(rng, n_patches - 2 * (n_patches // 3),
                             0, 10, n_windows, max_h=4, max_w=8),
            _random_patches(rng, n_patches - 2 * (n_patches // 3),
                             10, 20, n_windows, max_h=4, max_w=8),
        ))
        patches = soz_patches + non_patches
        rf_kwargs = dict(
            n_estimators=500, max_features="sqrt", min_samples_split=2,
            class_weight="balanced", n_jobs=-1,
        )

        def make_features(Xflat: np.ndarray, t: float) -> np.ndarray:
            if t > 0:
                Xflat = np.where(Xflat >= t, Xflat, 0.0).astype(np.float32)
            return patch_mean_features(Xflat, data_dims, patches,
                                        contrast_pairs)

        def make_clf(seed):
            return RandomForestClassifier(random_state=seed, **rf_kwargs)

    # Average over `n_seeds` independent nested-CV runs to stabilise the
    # AUC estimate at this small N regime (results vary by ~0.05 between
    # seeds otherwise).
    oof_proba_seeds = []
    fold_logs_all = []
    for seed_offset in range(n_seeds):
        seed = random_state + seed_offset * 1000
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                      random_state=seed)
        oof_proba = np.full(len(y), np.nan)
        fold_log = []
        for fold, (tr, te) in enumerate(outer.split(X, y, groups=subs_arr)):
            n_inner_max = min(5,
                               len(np.unique(subs_arr[tr][y[tr] == 1])),
                               len(np.unique(subs_arr[tr][y[tr] == 0])))
            n_inner = max(2, n_inner_max)
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                          random_state=seed + fold + 1)
            best_t, best_auc = thresholds[0], -np.inf
            for t in thresholds:
                inner_aucs = []
                for itr, ite in inner.split(X[tr], y[tr],
                                             groups=subs_arr[tr]):
                    Xt = make_features(X[tr][itr], t)
                    Xv = make_features(X[tr][ite], t)
                    if len(np.unique(y[tr][ite])) < 2:
                        continue
                    clf = make_clf(seed + fold * 101 + int(t * 10))
                    clf.fit(Xt, y[tr][itr])
                    inner_aucs.append(roc_auc_score(
                        y[tr][ite], clf.predict_proba(Xv)[:, 1]))
                mean_auc = (float(np.mean(inner_aucs)) if inner_aucs
                            else -np.inf)
                if mean_auc > best_auc:
                    best_auc, best_t = mean_auc, t
            clf = make_clf(seed + fold)
            clf.fit(make_features(X[tr], best_t), y[tr])
            oof_proba[te] = clf.predict_proba(
                make_features(X[te], best_t))[:, 1]
            fold_log.append({"fold": fold, "best_threshold": best_t,
                              "inner_auc": float(best_auc),
                              "test_n": int(len(te))})
        oof_proba_seeds.append(oof_proba)
        fold_logs_all.append(fold_log)

    oof_proba = np.nanmean(np.stack(oof_proba_seeds, axis=0), axis=0)
    seizure_auc = float(roc_auc_score(y, oof_proba))
    # Per-patient mean P(success) across that patient's seizures.
    pat_idx, pat_proba, pat_y = [], [], []
    for sub in sorted(set(subs)):
        m = subs_arr == sub
        if not m.any():
            continue
        pat_idx.append(sub)
        pat_proba.append(float(np.nanmean(oof_proba[m])))
        pat_y.append(int(y[m][0]))
    pat_proba = np.asarray(pat_proba)
    pat_y_arr = np.asarray(pat_y)
    pat_auc = float(roc_auc_score(pat_y_arr, pat_proba)) \
        if len(np.unique(pat_y_arr)) > 1 else None
    pat_yhat = (pat_proba >= 0.5).astype(int)
    pat_fail = np.where(pat_y_arr == 0)[0]
    pat_succ = np.where(pat_y_arr == 1)[0]
    return {
        "backend": "treeple" if treeple_clf is not None else "sklearn-fallback",
        "per_seizure": per_seizure,
        "n_seizures": int(len(y)),
        "n_patients": int(len(pat_idx)),
        "seizure_auc": seizure_auc,
        "patient_auc": pat_auc,
        "predicted_failures": int((pat_yhat[pat_fail] == 0).sum()),
        "total_failures": int(len(pat_fail)),
        "predicted_successes": int((pat_yhat[pat_succ] == 1).sum()),
        "total_successes": int(len(pat_succ)),
        "fold_log": fold_logs_all,
        "n_seeds": n_seeds,
        "patient_ids": pat_idx,
        "patient_proba": pat_proba.tolist(),
        "patient_y": pat_y_arr.tolist(),
    }


def per_patient_auc(features: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows = []
    for sub, g in features.groupby("sub"):
        if g["is_soz"].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(g["is_soz"].astype(int), g["fragility"])
        except ValueError:
            continue
        rows.append({
            "sub": sub,
            "outcome": g["outcome"].iloc[0],
            "site": g["site"].iloc[0],
            "auc": float(auc),
            "n_ch": len(g),
            "n_soz": int(g["is_soz"].sum()),
        })
    return pd.DataFrame(rows)


# ---------- driver --------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only run on first N patients")
    args = parser.parse_args()

    recs = patient_records()
    if args.limit:
        recs = recs[: args.limit]

    feats = []
    for rec in recs:
        if not rec["soz"]:
            print(f"  {rec['sub']}: no SOZ contacts in xlsx, skipping",
                  flush=True)
            continue
        print(f"-> {rec['sub']} (outcome={rec['outcome']}, "
              f"site={rec['site']}, n_soz={len(rec['soz'])})", flush=True)
        df = compute_patient_features(rec)
        if df is None:
            continue
        feats.append(df)

    if not feats:
        print("no features computed", file=sys.stderr)
        return 1
    features = pd.concat(feats, ignore_index=True)
    features.to_parquet(CACHE / "all_features.parquet", index=False)

    print("\n=== Pooled Mann-Whitney U-test (one-sided, SOZ > SOZ^C) ===")
    mwu = pooled_mwu(features)
    for outcome, stats in mwu.items():
        print(f"  outcome={outcome}: n_soz={stats['n_soz']:5d} "
              f"n_nonsoz={stats['n_nonsoz']:5d} "
              f"mean_soz={stats['mean_soz']:.4f} "
              f"mean_nonsoz={stats['mean_nonsoz']:.4f} "
              f"p={stats['p_one_sided_greater']:.3e}")
    print("  paper: P(success) = 3.326e-70, P(fail) = 0.355")

    print("\n=== Per-patient SOZ-classification AUC ===")
    aucs = per_patient_auc(features)
    aucs.to_csv(CACHE / "per_patient_auc.csv", index=False)
    for outcome in ("S", "F", "NR"):
        g = aucs[aucs["outcome"] == outcome]
        if g.empty:
            continue
        print(f"  outcome={outcome}: n={len(g):2d} "
              f"AUC = {g['auc'].mean():.3f} ± {g['auc'].std():.3f} "
              f"(median {g['auc'].median():.3f})")
    g = aucs
    print(f"  ALL:        n={len(g):2d} "
          f"AUC = {g['auc'].mean():.3f} ± {g['auc'].std():.3f} "
          f"(median {g['auc'].median():.3f})")

    print("\n=== Surgical-outcome prediction (single feature: per-patient SOZ-AUC) ===")
    from sklearn.metrics import roc_auc_score
    sf = aucs[aucs["outcome"].isin(["S", "F"])]
    if len(sf) and sf["outcome"].nunique() > 1:
        y = (sf["outcome"] == "S").astype(int)
        outcome_auc = float(roc_auc_score(y, sf["auc"]))
        print(f"  n={len(sf)}: outcome AUC = {outcome_auc:.3f}")
    else:
        outcome_auc = None
    print("  paper: outcome AUC = 0.88 ± 0.064 (10-fold nested CV manifold RF)")

    print("\n=== Manifold-RF outcome prediction (10-fold nested CV, per-seizure) ===")
    valid_recs = [r for r in recs if r["outcome"] in ("S", "F")
                  and any(CACHE.glob(f"{r['sub']}__*.npz"))]
    # 120 windows ≈ 15 s @ stepsize 125 ms — covers the paper's window:
    # 10 s pre-onset + up to 5 s post-onset (5 % of a 100 s seizure).
    mrf = manifold_rf_nested_cv(valid_recs, n_windows=120, n_outer=10,
                                per_seizure=True, n_seeds=5)
    if "error" in mrf:
        print(f"  skipped: {mrf['error']} (n={mrf.get('n_seizures', 0)})")
    else:
        print(f"  backend={mrf['backend']}, "
              f"{mrf['n_seizures']} seizures across {mrf['n_patients']} patients")
        print(f"  per-seizure AUC = {mrf['seizure_auc']:.3f}")
        print(f"  patient-level AUC (mean P over each patient's seizures) "
              f"= {mrf['patient_auc']:.3f}")
        print(f"  predicted {mrf['predicted_failures']}/"
              f"{mrf['total_failures']} failures and "
              f"{mrf['predicted_successes']}/"
              f"{mrf['total_successes']} successes correctly")
    print("  paper: patient AUC = 0.88 ± 0.064; 43/47 failures predicted "
          "(91 patients, 462 seizures)")

    summary = {
        "n_patients": int(features["sub"].nunique()),
        "pooled_mwu": mwu,
        "outcome_prediction_auc_singlefeat": outcome_auc,
        "manifold_rf": (
            {k: v for k, v in mrf.items()
             if k not in ("fold_log",)}
            if "error" not in mrf else mrf),
        "per_patient_auc": {
            "n": int(len(aucs)),
            "mean": float(aucs["auc"].mean()) if len(aucs) else None,
            "std": float(aucs["auc"].std()) if len(aucs) else None,
            "median": float(aucs["auc"].median()) if len(aucs) else None,
            "by_outcome": {
                o: {
                    "n": int((aucs["outcome"] == o).sum()),
                    "mean": float(aucs.loc[aucs["outcome"] == o, "auc"].mean())
                    if (aucs["outcome"] == o).any() else None,
                    "std": float(aucs.loc[aucs["outcome"] == o, "auc"].std())
                    if (aucs["outcome"] == o).any() else None,
                }
                for o in aucs["outcome"].unique()
            },
        },
    }
    (CACHE / "summary.json").write_text(json.dumps(summary, indent=2,
                                                   default=str))
    print(f"\nWrote summary to {CACHE/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
