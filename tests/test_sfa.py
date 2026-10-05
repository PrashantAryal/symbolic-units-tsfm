"""Sanity checks for symbolic/sfa.py (BOSS-ST path) on the prompt's worked example.

The SAX fixture shows flat [1,1,1,1] and oscillating [1,2,1,2] colliding on "bb".
SFA works in the frequency domain, so with enough coefficients it separates them:
the oscillation puts all its energy in the Nyquist coefficient (Re X2 = -4).
"""
import numpy as np
import pytest

from symtsfm.symbolic.sfa import SFA, BOSSSTSelector, dft_coefficients, mcb_edges, sfa_symbols
from symtsfm.symbolic.sax import znormalize
from tests.test_sax import FLAT, OSC, SPIKE, _toy_dataset

# fixed symmetric edges so the words are deterministic (MCB would learn them from data)
EDGES_3 = np.tile([-0.5, 0.5], (4, 1))


def _word(window, word_len=4):
    return SFA(word_len=word_len, alphabet_size=3).set_edges(EDGES_3[:word_len]).words(window[None])[0]


def test_dft_coefficients_worked_example():
    np.testing.assert_allclose(dft_coefficients(znormalize(SPIKE), 4), [-2.814, 0.0, -0.402, 0.0], atol=2e-3)
    np.testing.assert_allclose(dft_coefficients(znormalize(OSC), 4), [0.0, 0.0, -4.0, 0.0], atol=1e-9)
    np.testing.assert_array_equal(dft_coefficients(znormalize(FLAT), 4), np.zeros(4))


def test_dft_matches_numpy_on_random_window():
    x = np.random.default_rng(0).normal(size=32)
    F = np.fft.rfft(x)
    np.testing.assert_allclose(dft_coefficients(x, 6), [F[1].real, F[1].imag, F[2].real, F[2].imag, F[3].real, F[3].imag])


@pytest.mark.parametrize("window,expected", [(SPIKE, "abbb"), (FLAT, "bbbb"), (OSC, "bbab")])
def test_fixture_sfa_words_exact(window, expected):
    assert _word(window) == expected


def test_flat_window_maps_to_middle_symbol():
    assert set(_word(FLAT)) == {"b"}


def test_sfa_separates_flat_from_oscillating_where_sax_collides():
    assert _word(FLAT) != _word(OSC)
    assert _word(SPIKE) != _word(OSC)


def test_short_word_collides_like_sax():
    # with only the first coefficient pair, flat and oscillation collide again
    assert _word(FLAT, 2) == _word(OSC, 2)


def test_mcb_edges_are_equi_depth():
    coefs = np.random.default_rng(1).normal(size=(10_000, 4))
    edges = mcb_edges(coefs, 4)
    assert edges.shape == (4, 3)
    syms = sfa_symbols(coefs, edges)
    for i in range(4):
        counts = np.bincount(syms[:, i], minlength=4) / len(coefs)
        np.testing.assert_allclose(counts, 0.25, atol=0.01)


def test_bossst_selector_scores_and_locations():
    X, y = _toy_dataset()
    sel = BOSSSTSelector(window=8, word_len=4, alphabet_size=4, top_k=5, seed=0).fit(X, y)
    assert len(sel.words_) == 5
    assert sel.info_gain_[0] > 0.5  # the classes are trivially separable symbolically
    m = sel.match(X)
    for key in ("presence", "frequency", "distance", "location"):
        assert m[key].shape == (len(X), 5)
    assert (m["distance"] >= 0).all() and (m["distance"] <= 1).all()
    assert m["location"].min() >= 0 and m["location"].max() <= X.shape[1] - 8
    # exact presence <=> zero symbolic distance
    np.testing.assert_array_equal(m["presence"] > 0, m["distance"] == 0)


def test_bossst_unsupervised_mode():
    X, _ = _toy_dataset()
    sel = BOSSSTSelector(window=8, word_len=4, alphabet_size=4, top_k=3).fit(X, None)
    assert sel.info_gain_ == [0.0, 0.0, 0.0]
    assert sel.word_class_ == [None, None, None]
    assert sel.match(X[:3])["rarity"].shape == (3,)


def test_bossst_excludes_the_flat_word():
    X = np.full((30, 80), 2.0)
    X[:, 30:40] += np.hanning(10)[None] * np.linspace(1, 3, 30)[:, None]
    sel = BOSSSTSelector(window=10, word_len=4, alphabet_size=4, top_k=3).fit(X, None)
    flat = sel.sfa.words(np.zeros((1, 10)))[0]
    assert flat not in sel.words_
    m = sel.match(np.full((1, 80), 5.0))
    assert not m["informative"].any()


def test_bossst_flat_windows_are_never_the_match_location():
    X = np.full((20, 120), 1.0)
    X[:, 80:92] += np.hanning(12)[None] * np.linspace(1, 2, 20)[:, None]
    sel = BOSSSTSelector(window=12, word_len=4, alphabet_size=4, top_k=2).fit(X, None)
    loc = sel.match(X[:3])["location"]
    assert ((loc > 80 - 12) & (loc < 92)).all()


def test_multivariate_with_dead_channels():
    """Real multivariate data (SMD, MSL) contains constant channels. Those yield few or no
    words; the feature block must stay rectangular and must not invent evidence."""
    import numpy as np
    from symtsfm.symbolic.features import SymbolicFeatureExtractor

    rng = np.random.default_rng(0)
    n, T = 30, 200
    X = np.zeros((n, 4, T), dtype=np.float32)
    X[:, 0] = rng.normal(0, 1, (n, T)).cumsum(1)          # normal signal
    X[:, 1] = 3.7                                          # dead: constant everywhere
    X[:, 2] = rng.normal(0, 1, (n, T))                     # noise
    X[:, 3, :] = np.arange(T)[None] * 0 + 1.0              # dead: another constant
    y = np.array([i % 2 for i in range(n)])

    for method in ("fastshapelets", "bossst"):
        ext = SymbolicFeatureExtractor(method=method, window=20, word_len=4, alphabet_size=4,
                                       top_k=5, seed=0)
        feats, meta = ext.fit_transform(X, y)
        assert feats.shape == (n, 4, ext.dim_per_channel), feats.shape
        assert np.isfinite(feats).all()
        assert ext.K == 5
        # dead channels contribute placeholder words, never fabricated ones
        for c in (1, 3):
            words = ext.words(c)
            assert len(words) == ext.K
            assert all(w["word"] == "<none>" for w in words[ext.k_[c]:])
        assert ext.k_[0] > 0 and ext.k_[2] > 0             # the live channels still work
        assert ext.transform(X[:3])[0].shape == (3, 4, ext.dim_per_channel)
