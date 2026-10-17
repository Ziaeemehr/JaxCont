"""
Mode tracking for spectral event detection.

An eigensolver returns each point's spectrum in an arbitrary order, so "the
eigenvalue nearest critical" picked independently at every point jumps
between modes whenever two of them trade places. ``track_modes`` reorders a
branch's spectra so that each column follows one continuous mode, and
``crossings`` finds the brackets where a per-mode test value changes sign.
Spectral events build their ``scan`` hook on these two functions.

Host-side NumPy only: this runs in ``detect_events``'s eager post-pass.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackedMode(NamedTuple):
    """The mode identity a spectral ``scan`` hands to ``refine``.

    ``column`` indexes the tracked mode; ``left``/``right`` are the bracket
    endpoints' full spectra in tracked order, so ``left[column]`` and
    ``right[column]`` are the same mode's eigenvalue at both ends.
    """

    column: int
    left: np.ndarray
    right: np.ndarray


def track_modes(spectra: np.ndarray) -> np.ndarray:
    """Reorder ``(N, n)`` spectra so that column ``k`` is one continuous mode.

    Consecutive points are matched by ``linear_sum_assignment`` on the cost
    ``|pred_i(t+1) - lambda_j(t+1)|``, where ``pred_i(t+1)`` is mode ``i``'s
    linear extrapolation from points ``t-1`` and ``t`` (just ``lambda_i(t)``
    at the first step). The extrapolation carries two real eigenvalues
    through a crossing: matching on ``|lambda_i(t) - lambda_j(t+1)|`` alone
    makes them bounce off each other whenever the samples straddle the
    crossing. Row ``0`` keeps its input order.
    """
    spectra = np.asarray(spectra)
    out = np.array(spectra, dtype=np.result_type(spectra.dtype, np.complex64), copy=True)
    for t in range(1, out.shape[0]):
        pred = out[t - 1] if t == 1 else 2.0 * out[t - 1] - out[t - 2]
        cost = np.abs(pred[:, None] - out[t][None, :])
        # A non-finite eigenvalue can be matched, but never preferred.
        cost = np.where(np.isfinite(cost), cost, np.finfo(cost.dtype).max / 4)
        _, cols = linear_sum_assignment(cost)
        out[t] = out[t][cols]
    return out


def crossings(
    values: np.ndarray, where: np.ndarray, *, eligible: np.ndarray | None = None
) -> list[int]:
    """Bracket starts ``i`` where some column of ``values`` crosses zero.

    A bracket ``(i, i+1)`` counts for column ``k`` when
    ``where[i, k] & where[i+1, k]`` and ``values[:, k]`` changes sign or is
    exactly zero at exactly one end. An exact zero at sample ``i`` is owned by
    bracket ``i-1`` (or bracket ``0`` when ``i == 0``): bracket ``i`` is then
    not opened for that column. ``values``/``where`` may be ``(N,)`` for a
    single column. Optional ``eligible`` masks adjacent brackets before zero
    ownership: an excluded incoming bracket cannot own a zero. Returns sorted,
    unique starts.
    """
    values = np.asarray(values)
    where = np.asarray(where, dtype=bool)
    if values.ndim == 1:
        values, where = values[:, None], where[:, None]
    a, b = values[:-1], values[1:]
    # An exact zero counts at one end only: a mode that is identically zero
    # (e.g. a center's real part) never opens a bracket.
    hit = ((np.sign(a) != np.sign(b)) & ((a != 0) | (b != 0)))
    hit &= np.isfinite(a) & np.isfinite(b)
    owned = hit & where[:-1] & where[1:]
    if eligible is not None:
        bracket_mask = np.asarray(eligible, dtype=bool)
        if bracket_mask.ndim == 1:
            bracket_mask = bracket_mask[:, None]
        owned &= bracket_mask
    # Bracket i starts on an exact zero that bracket i-1 already ends on.
    owned[1:] &= ~((a[1:] == 0) & owned[:-1])
    return np.flatnonzero(np.any(owned, axis=1)).tolist()


def mode_brackets(
    spectra: np.ndarray, values: np.ndarray, where: np.ndarray
) -> list[tuple[int, int, Any]]:
    """Brackets for ``Event.scan``: one ``(i, i+1, TrackedMode)`` per crossing mode.

    ``spectra`` is the ``track_modes`` output; ``values``/``where`` are the
    per-mode test values and validity masks on the same columns. Distinct modes in the same bracket retain their own identities.
    """
    brackets = []
    for k in range(spectra.shape[1]):
        for i in crossings(values[:, k], where[:, k]):
            brackets.append((i, i + 1, TrackedMode(k, spectra[i], spectra[i + 1])))
    brackets.sort(key=lambda b: b[0])
    return brackets
