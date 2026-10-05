"""Sanity-check fixture for symbolic/sax.py (FastShapelets path).

The worked example from the implementation prompt:

    Class A spike   [2, 6, 9, 6] -> z-norm ~[-1.508, 0.100, 1.306, 0.100]
                                  -> PAA(2) ~[-0.70, 0.70] -> "ac"
    Class A flat    [1, 1, 1, 1] -> "bb" (zero variance -> middle symbol)
    Class B oscill. [1, 2, 1, 2] -> z-norm [-1, 1, -1, 1] -> PAA [0, 0] -> "bb"
"""
import numpy as np
import pytest

from symtsfm.symbolic.sax import (
    FastShapeletSelector,
    paa,
    sax_breakpoints,
    sax_word,
    sax_words_for_series,
    sliding_windows,
    znormalize,
)

SPIKE = np.array([2.0, 6.0, 9.0, 6.0])
FLAT = np.array([1.0, 1.0, 1.0, 1.0])
OSC = np.array([1.0, 2.0, 1.0, 2.0])


# --------------------------------------------------------------------------- fixture
def test_znormalize_spike():
    np.testing.assert_allclose(znormalize(SPIKE), [-1.508, 0.100, 1.306, 0.100], atol=1e-3)


def test_znormalize_oscillating():
    np.testing.assert_allclose(znormalize(OSC), [-1.0, 1.0, -1.0, 1.0], atol=1e-12)


def test_znormalize_flat_is_zero():
    np.testing.assert_array_equal(znormalize(FLAT), np.zeros(4))


def test_paa_spike():
    np.testing.assert_allclose(paa(znormalize(SPIKE), 2), [-0.70, 0.70], atol=5e-3)


def test_breakpoints_alphabet3():
    np.testing.assert_allclose(sax_breakpoints(3), [-0.4307, 0.4307], atol=1e-4)


@pytest.mark.parametrize(
    "window,expected",
    [(SPIKE, "ac"), (FLAT, "bb"), (OSC, "bb")],
)
def test_fixture_words_exact(window, expected):
    assert sax_word(window, word_len=2, alphabet_size=3) == expected


def test_fixture_collision_is_real():
    # The documented failure mode of a symbolic-only design: flat and oscillating
    # windows collide on the same SAX word.
    assert sax_word(FLAT, 2, 3) == sax_word(OSC, 2, 3)
    assert sax_word(SPIKE, 2, 3) != sax_word(OSC, 2, 3)


# --------------------------------------------------------------------------- helpers
def test_paa_non_divisible_length_preserves_mean():
    x = np.arange(10, dtype=float)
    p = paa(x, 3)
    assert p.shape == (3,)
    assert np.isclose(p.mean(), x.mean())
    assert np.all(np.diff(p) > 0)


def test_sliding_windows_shapes_and_starts():
    x = np.arange(10, dtype=float)
    w, starts = sliding_windows(x, 4, step=2)
    assert w.shape == (4, 4)
    np.testing.assert_array_equal(starts, [0, 2, 4, 6])
    np.testing.assert_array_equal(w[1], [2, 3, 4, 5])


def test_sax_words_for_series_matches_single_window_encoder():
    x = np.concatenate([SPIKE, FLAT, OSC])
    words, starts = sax_words_for_series(x, window=4, word_len=2, alphabet_size=3, step=4)
    assert list(words) == ["ac", "bb", "bb"]
    np.testing.assert_array_equal(starts, [0, 4, 8])


# --------------------------------------------------------------------------- selector
def _toy_dataset(n_per_class=20, length=64, seed=0):
    """Class 0 contains a spike at a random position, class 1 a sine-ish oscillation."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        s = rng.normal(0, 0.1, length)
        p = rng.integers(5, length - 12)
        s[p : p + 8] += np.array([0, 2, 5, 8, 5, 2, 0, 0])
        X.append(s)
        y.append(0)
    for _ in range(n_per_class):
        s = rng.normal(0, 0.1, length) + np.sin(np.arange(length) * np.pi / 2)
        X.append(s)
        y.append(1)
    return np.stack(X), np.array(y)


def test_fastshapelet_selector_finds_discriminative_words_and_valid_locations():
    X, y = _toy_dataset()
    sel = FastShapeletSelector(
        window=8, word_len=4, alphabet_size=4, top_k=5, n_projections=10,
        mask_size=1, n_candidates=30, seed=0,
    )
    sel.fit(X, y)
    assert len(sel.words_) == 5
    assert all(ig > 0 for ig in sel.info_gain_)
    m = sel.match(X)
    for key in ("presence", "frequency", "distance", "location"):
        assert m[key].shape == (len(X), 5)
    assert m["location"].min() >= 0
    assert m["location"].max() <= X.shape[1] - 8
    # best shapelet should separate the classes on real distance
    d0 = m["distance"][y == 0, 0].mean()
    d1 = m["distance"][y == 1, 0].mean()
    assert abs(d0 - d1) > 0.5


def test_fastshapelet_selector_is_deterministic():
    X, y = _toy_dataset()
    kw = dict(window=8, word_len=4, alphabet_size=4, top_k=3, n_projections=5, seed=1)
    a = FastShapeletSelector(**kw).fit(X, y)
    b = FastShapeletSelector(**kw).fit(X, y)
    assert a.words_ == b.words_


def test_flat_words_are_never_selected_and_flat_inputs_flag_uninformative_locations():
    # telemetry-like data: long constant stretches plus a class-specific bump
    rng = np.random.default_rng(0)
    X, y = [], []
    for i in range(40):
        s = np.full(96, 1.5)
        p = rng.integers(10, 70)
        s[p : p + 10] += (np.hanning(10) * 3) if i % 2 else np.linspace(0, 3, 10)
        X.append(s)
        y.append(i % 2)
    X, y = np.stack(X), np.array(y)
    for labels in (y, None):
        sel = FastShapeletSelector(window=12, word_len=4, alphabet_size=4, top_k=4, n_projections=5, seed=0).fit(X, labels)
        assert "cccc" not in sel.words_  # the SAX word of an all-zero z-normalised window
        assert (sel.shapelets_.std(1) > 1e-6).all()
    m = sel.match(np.full((2, 96), 3.0))
    assert not m["informative"].any()  # all windows equidistant -> location is arbitrary
    assert sel.match(X[:2])["informative"].all()


def test_flat_windows_are_never_the_match_location():
    # mostly-flat input with one bump: the best match must sit on the bump, not on flat filler
    X = np.full((1, 200), 2.0)
    X[0, 150:162] += np.hanning(12) * 5
    sel = FastShapeletSelector(window=12, word_len=4, alphabet_size=4, top_k=2, n_projections=3, seed=0)
    sel.fit(np.vstack([X, X + np.random.default_rng(0).normal(0, 0.01, X.shape)]), None)
    m = sel.match(X)
    assert ((m["location"] > 150 - 12) & (m["location"] < 162)).all()
    assert m["informative"].all()
