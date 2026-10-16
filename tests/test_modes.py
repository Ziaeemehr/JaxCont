"""Mode tracking and bracket detection (jaxcont.bifurcations.modes)."""

import numpy as np
import pytest

from jaxcont.bifurcations.modes import crossings, track_modes


@pytest.mark.parametrize("n_points", [40, 41])
def test_track_modes_follows_crossing_eigenvalues(n_points):
    # Two real eigenvalues p and -p cross at p = 0. An even grid straddles the
    # crossing, an odd one samples it exactly; in both cases each tracked
    # column must stay on one line instead of bouncing off the other.
    p = np.linspace(-1.0, 1.0, n_points)
    spectra = np.stack([p, -p], axis=1).astype(complex)
    spectra[1::2] = spectra[1::2, ::-1]  # eigensolver order is arbitrary
    tracked = track_modes(spectra)
    for k in range(2):
        diffs = np.diff(tracked[:, k].real)
        assert np.all(diffs > 0) or np.all(diffs < 0)


def test_track_modes_handles_conjugate_pairs():
    rng = np.random.default_rng(0)
    p = np.linspace(-1.0, 1.0, 50)
    pair = p + 1j * (1.0 + 0.2 * p)
    spectra = np.stack([pair, np.conj(pair), -0.5 + 0.0 * p, 0.3 * p - 2.0], axis=1)
    shuffled = np.array([row[rng.permutation(4)] for row in spectra])
    tracked = track_modes(shuffled)
    for k in range(4):
        col = tracked[:, k]
        # One mode per column: no jumps between consecutive points, and a
        # conjugate member keeps the sign of its imaginary part.
        assert np.max(np.abs(np.diff(col))) < 0.1
        assert np.all(np.sign(col.imag) == np.sign(col[0].imag))
    np.testing.assert_allclose(
        np.sort_complex(tracked[0]), np.sort_complex(spectra[0])
    )


def test_crossings_sign_change_exact_zero_and_mask():
    values = np.array([-2.0, -1.0, 1.0, 2.0, 0.0, 3.0, -1.0])
    where = np.ones_like(values, dtype=bool)
    # Sign change in bracket 1; exact zero at sample 4 owns brackets 3 and 4.
    assert crossings(values[:6], where[:6]) == [1, 3, 4]
    # A masked endpoint removes every bracket that touches it.
    where[6] = False
    assert crossings(values, where) == [1, 3, 4]
    where[2] = False
    assert crossings(values, where) == [3, 4]


def test_crossings_per_column():
    values = np.array([[-1.0, 1.0], [1.0, 2.0], [2.0, -1.0]])
    where = np.ones_like(values, dtype=bool)
    assert crossings(values, where) == [0, 1]
    where[:, 1] = False
    assert crossings(values, where) == [0]
