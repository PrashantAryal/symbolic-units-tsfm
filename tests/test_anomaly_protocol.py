from __future__ import annotations

import numpy as np

from symtsfm.evaluation.anomaly_protocol import _raw_fpr, best_f1_threshold, point_adjust


def test_fpr_is_non_negative_and_uses_normal_points_only():
    labels = np.array([0, 0, 1, 1])
    prediction = np.array([1, 0, 1, 0])
    assert _raw_fpr(prediction, labels) == 0.5


def test_best_f1_threshold_matches_returned_precision_and_recall():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    f1, precision, recall, threshold = best_f1_threshold(scores, labels)
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert 0.2 < threshold <= 0.8


def test_point_adjust_marks_complete_detected_event():
    labels = np.array([0, 1, 1, 1, 0])
    prediction = np.array([0, 0, 1, 0, 0])
    assert point_adjust(prediction, labels).tolist() == [False, True, True, True, False]

