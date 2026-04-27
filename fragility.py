"""Neural fragility algorithm from Li et al., Nat. Neurosci. 2021.

Implements the seizure-onset-zone biomarker described in
"Neural fragility as an EEG marker of the seizure onset zone"
(https://doi.org/10.1038/s41593-021-00901-w).

Pipeline:
  1. Preprocess each channel: 60 Hz notch (Q=30), 4th-order Butterworth
     high-pass at 0.5 Hz (zero-phase via filtfilt), then common average
     reference across channels.
  2. Slide a (winsize, stepsize) = (250, 125) ms window over the signal.
     In each window, fit a linear time-invariant model X(t+1) = A X(t) by
     least squares using the Moore-Penrose pseudoinverse.
  3. For each channel i in each window, compute the minimum-norm real
     column perturbation Γ_i such that A + Γ_i e_i^T has λ = r as an
     eigenvalue (default r = 1.5 to push the system unstable). The 2-norm
     of Γ_i is the channel's fragility for that window.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt


def preprocess(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Notch 60 Hz (Q=30), 4th-order Butterworth high-pass at 0.5 Hz, CAR."""
    b_n, a_n = iirnotch(60.0, Q=30.0, fs=sfreq)
    data = filtfilt(b_n, a_n, data, axis=-1)
    b_hp, a_hp = butter(4, 0.5, btype="highpass", fs=sfreq)
    data = filtfilt(b_hp, a_hp, data, axis=-1)
    return data - data.mean(axis=0, keepdims=True)


def estimate_state_matrices(
    data: np.ndarray, winsize: int, stepsize: int, l2penalty: float = 0.0
) -> np.ndarray:
    """Fit A per sliding window from X(t+1) ≈ A X(t).

    Parameters
    ----------
    data : (n_ch, n_times) array.
    winsize : window length in samples.
    stepsize : step between successive windows in samples.
    l2penalty : Tikhonov regularization on the normal equations. 0 disables
        regularization and a Moore-Penrose pseudoinverse is used instead.

    Returns
    -------
    A : (n_ch, n_ch, n_win) array of fitted state matrices.
    """
    n_ch, n_times = data.shape
    n_wins = (n_times - winsize) // stepsize + 1
    A = np.empty((n_ch, n_ch, n_wins), dtype=np.float64)
    eye = np.eye(n_ch)
    for w in range(n_wins):
        start = w * stepsize
        seg = data[:, start:start + winsize]
        Xt = seg[:, :-1]
        Xtp1 = seg[:, 1:]
        if l2penalty > 0:
            G = Xt @ Xt.T + l2penalty * eye
            A[:, :, w] = Xtp1 @ Xt.T @ np.linalg.inv(G)
        else:
            A[:, :, w] = Xtp1 @ np.linalg.pinv(Xt)
    return A


def min_norm_col_perturbation(
    A: np.ndarray, radius: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-norm column perturbations driving an eigenvalue to ±radius.

    For each channel i, finds the real vector Γ_i of smallest 2-norm such
    that A + Γ_i e_i^T has λ = radius as a real eigenvalue. The matrix
    determinant lemma reduces this to a single linear constraint
    b_i^T Γ_i = -1 with b_i = ((A - λI)^{-1})^T e_i, whose minimum-norm
    real solution is Γ_i = -b_i / (b_i^T b_i) and ||Γ_i|| = 1/||b_i||.

    Returns
    -------
    gamma : (n_ch, n_ch) array. Row i is Γ_i (so A + gamma[i].reshape(-1,1)
        @ e_i.reshape(1,-1) has λ = radius as an eigenvalue).
    norms : (n_ch,) array with ||Γ_i||_2.
    """
    n_ch = A.shape[0]
    M = np.linalg.inv(A - radius * np.eye(n_ch))
    sq = np.einsum("ij,ij->i", M, M)  # ||row i||^2 of M
    gamma = -M / sq[:, None]
    norms = 1.0 / np.sqrt(sq)
    return gamma, norms


def normalize_fragility(perturb: np.ndarray) -> np.ndarray:
    """Per-window normalization producing the paper's fragility heatmap.

    For each window the raw min-perturbation norm ||Γ_i|| is mapped to
    ``(max - ||Γ_i||) / max``, so that the most fragile channel (smallest
    norm) has a value near 1 and the most stable has a value near 0.
    This is the "neural fragility" that the paper's heatmaps display and
    the SOZ tends to have high values of.
    """
    mx = perturb.max(axis=0, keepdims=True)
    mx = np.where(mx > 0, mx, 1.0)
    return (mx - perturb) / mx


def compute_fragility(
    data: np.ndarray,
    sfreq: float,
    winsize: int = 250,
    stepsize: int = 125,
    radius: float = 1.5,
    l2penalty: float = 1e-4,
    do_preprocess: bool = True,
) -> dict:
    """End-to-end fragility computation.

    Returns a dict with ``state`` (n_ch, n_ch, n_win), ``deltavecs``
    (n_ch, n_ch, n_win) — the min-norm column perturbations Γ_i stored as
    rows (deltavecs[i, :, w] is Γ_i for channel i in window w), ``perturb``
    (n_ch, n_win) — the raw 2-norms of those vectors — and ``fragility``
    (n_ch, n_win) — ``perturb`` after per-window normalization, i.e. the
    paper's fragility heatmap where the SOZ tends to be high-valued.
    """
    if do_preprocess:
        data = preprocess(data, sfreq)
    A = estimate_state_matrices(data, winsize, stepsize, l2penalty=l2penalty)
    n_ch, _, n_win = A.shape
    delta = np.zeros_like(A)
    perturb = np.zeros((n_ch, n_win), dtype=np.float64)
    for w in range(n_win):
        g, n = min_norm_col_perturbation(A[:, :, w], radius=radius)
        delta[:, :, w] = g
        perturb[:, w] = n
    return {"state": A, "deltavecs": delta, "perturb": perturb,
            "fragility": normalize_fragility(perturb)}
