import numpy as np

from symtsfm.symbolic.behavior_rules import ContrastiveBehaviorRules


def _features(n=40):
    # miner dimensions 3*K+1 = 4, then four ordered SAX words
    x = np.zeros((n, 1, 8), dtype=np.float32)
    x[:, 0, -4:] = np.array([0, 1, 2, 3], dtype=np.float32)
    x[: n // 2, 0, :3] = np.array([1, 1, 0], dtype=np.float32)
    x[n // 2:, 0, -3:] = np.array([3, 2, 1], dtype=np.float32)
    return x


def test_contrast_rules_detect_nonpermuted_behaviour():
    x = _features()
    m = ContrastiveBehaviorRules(ordered_words=4, ngram=2, min_support=2,
                                 min_confidence=0.6, min_growth=1.1, seed=3)
    got = m.fit_apply(x[:10], x[10:30], np.r_[np.ones(10), np.zeros(10)],
                      x[30:], np.zeros(10), behavior="high_forecast_error")
    assert got["summary"]["n_rules"] > 0
    assert got["summary"]["test_target_never_used_for_mining_or_threshold_selection"]


def test_event_vocabulary_requires_ordered_suffix():
    m = ContrastiveBehaviorRules(ordered_words=4)
    try:
        m.fit_event_vocabulary(np.zeros((2, 1, 3), dtype=np.float32))
    except ValueError:
        return
    assert False, "missing SAX suffix must be rejected"
