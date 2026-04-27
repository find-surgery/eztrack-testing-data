"""Tests for the neural fragility implementation in ``fragility.py``.

Ground truth comes from the per-subject eztrack/Persyst derivatives shipped
in this repo for ``sub-la02``: state, deltavecs and perturbation matrices.

Two layers of correctness are checked:

1. Algorithmic — feeding the ground-truth state matrix A through our
   perturbation routine reproduces the saved deltavecs and perturbation
   matrices to machine precision. This isolates the perturbation logic
   from preprocessing/least-squares sensitivities.

2. End-to-end — running ``compute_fragility`` from the raw Persyst clip
   yields outputs of the right shape and a fragility heatmap that is
   strongly correlated with the ground truth (Pearson r ≥ 0.5 across the
   non-tail windows where the linear fit is well-conditioned).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest

import fragility as fr

ROOT = Path(__file__).resolve().parent
GT_DIR = ROOT / "Persyst/derivatives/fragility/monopolar/sub-la02"
PREFIX = "sub-la02_ses-presurgery_task-ictal_acq-seeg_desc-{}_run-01"
CLIP = ROOT / "Persyst/sourcedata/la02_ictal_reduced-clip.lay"


def _load_gt():
    state = np.load(GT_DIR / f"{PREFIX.format('statematrix')}.npy")
    delta = np.load(GT_DIR / f"{PREFIX.format('deltavecsmatrix')}.npy")
    pert = np.load(GT_DIR / f"{PREFIX.format('perturbmatrix')}.npy")
    with open(GT_DIR / f"{PREFIX.format('statematrix')}.json") as f:
        meta = json.load(f)
    return state, delta, pert, meta


def _load_raw_la02():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_persyst(str(CLIP), preload=True, verbose=False)
    _, _, _, meta = _load_gt()
    raw.pick(meta["ch_names"])
    return raw, meta


@pytest.fixture(scope="module")
def gt():
    return _load_gt()


@pytest.fixture(scope="module")
def raw_picked():
    return _load_raw_la02()


def test_perturbation_matches_gt_given_gt_state(gt):
    """Feeding GT A through our perturbation reproduces GT delta and pert."""
    state_gt, delta_gt, pert_gt, meta = gt
    radius = 1.5  # from the deltavecsmatrix.json model_parameters

    n_ch, _, n_win = state_gt.shape
    delta = np.zeros_like(state_gt)
    pert = np.zeros((n_ch, n_win))
    for w in range(n_win):
        g, n = fr.min_norm_col_perturbation(state_gt[:, :, w], radius=radius)
        delta[:, :, w] = g
        pert[:, w] = n

    np.testing.assert_allclose(pert, pert_gt, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(delta, delta_gt, atol=1e-8, rtol=1e-8)


def test_min_norm_perturbation_drives_eigenvalue_to_radius(gt):
    """A + Γ_i e_i^T must have λ = radius as an exact eigenvalue."""
    state_gt, *_ = gt
    radius = 1.5
    A = state_gt[:, :, 0]
    gamma, norms = fr.min_norm_col_perturbation(A, radius=radius)
    n_ch = A.shape[0]
    for i in range(n_ch):
        Apert = A.copy()
        Apert[:, i] += gamma[i, :]  # Γ_i is row i of gamma
        evs = np.linalg.eigvals(Apert)
        assert np.min(np.abs(evs - radius)) < 1e-9
        # 2-norm of Γ_i equals reported norm
        np.testing.assert_allclose(np.linalg.norm(gamma[i, :]), norms[i],
                                   atol=1e-12)


def test_end_to_end_shapes_and_params(raw_picked, gt):
    """Pipeline output has the shapes documented in the JSON sidecars."""
    raw, meta = raw_picked
    state_gt, delta_gt, pert_gt, _ = gt
    data = raw.get_data() * 1e6  # micro-volts

    out = fr.compute_fragility(data, sfreq=meta["model_parameters"]["sfreq"],
                               winsize=meta["model_parameters"]["winsize"],
                               stepsize=meta["model_parameters"]["stepsize"],
                               radius=1.5, l2penalty=0.0)

    assert out["state"].shape == state_gt.shape
    assert out["deltavecs"].shape == delta_gt.shape
    assert out["perturb"].shape == pert_gt.shape
    assert np.all(np.isfinite(out["perturb"]))
    assert np.all(out["perturb"] >= 0)


def test_end_to_end_fragility_correlates_with_gt(raw_picked, gt):
    """Fragility heatmap from raw data correlates strongly with GT.

    We restrict to non-tail windows where the data isn't distorted by the
    constant samples padding the end of the clip.
    """
    raw, meta = raw_picked
    state_gt, _, pert_gt, _ = gt
    data = raw.get_data() * 1e6

    out = fr.compute_fragility(data,
                               sfreq=meta["model_parameters"]["sfreq"],
                               winsize=meta["model_parameters"]["winsize"],
                               stepsize=meta["model_parameters"]["stepsize"],
                               radius=1.5, l2penalty=0.0)
    n_win = out["perturb"].shape[1]
    keep = slice(0, n_win - 6)  # last few windows fall on the constant tail
    r = np.corrcoef(out["perturb"][:, keep].ravel(),
                    pert_gt[:, keep].ravel())[0, 1]
    assert r >= 0.5, f"fragility correlation with GT too low: r={r:.3f}"


def test_normalized_fragility_inverts_perturb(gt):
    """``normalize_fragility`` flips raw perturbation norms so the SOZ — known
    to have the smallest norms in la02 — gets the highest values."""
    _, _, pert_gt, meta = gt
    f = fr.normalize_fragility(pert_gt)
    assert f.shape == pert_gt.shape
    assert (f >= -1e-12).all() and (f <= 1.0 + 1e-12).all()
    ch_names = meta["ch_names"]
    soz_idx = [ch_names.index(c) for c in ("L'2", "L'3", "L'4")
               if c in ch_names]
    non_idx = [i for i in range(len(ch_names)) if i not in soz_idx]
    assert f[soz_idx].mean() > f[non_idx].mean()


def test_quantile_image_and_patch_features():
    """``replicate_paper`` builds a 20×T quantile image per patient and
    derives patch-mean features for the manifold-RF stage."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "replicate_paper", ROOT / "replicate_paper.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)

    rng = np.random.default_rng(0)
    heatmap = rng.uniform(size=(40, 60)).astype(np.float32)
    is_soz = np.zeros(40, dtype=bool); is_soz[:5] = True

    img = rp.quantile_image(heatmap, is_soz, n_windows=50)
    assert img.shape == (20, 50)
    assert img[:10].max() <= heatmap[is_soz].max() + 1e-6
    assert img[10:].max() <= heatmap[~is_soz].max() + 1e-6

    patches = rp._random_patches(rng, n_patches=8, row_lo=0, row_hi=10,
                                  w=50, max_h=4, max_w=8)
    contrast = list(zip(
        rp._random_patches(rng, 4, 0, 10, 50, max_h=4, max_w=8),
        rp._random_patches(rng, 4, 10, 20, 50, max_h=4, max_w=8),
    ))
    feats = rp.patch_mean_features(img.ravel()[None, :], (20, 50),
                                    patches, contrast)
    assert feats.shape == (1, 8 + 4)
    ps, pn = contrast[0]
    expect = (img[ps[0]:ps[1], ps[2]:ps[3]].mean()
              - img[pn[0]:pn[1], pn[2]:pn[3]].mean())
    assert abs(float(feats[0, 8]) - float(expect)) < 1e-5
