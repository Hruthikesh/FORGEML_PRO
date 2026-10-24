import numpy as np
import pytest

from forgeml.policies import CostPolicy, precision_constrained_threshold, top_k_review


def test_exact_budget_stable_ties():
    assert top_k_review([0.5, 0.5, 0.1], 1).tolist() == [True, False, False]
    assert top_k_review([0.5], 0).sum() == 0
    with pytest.raises(ValueError):
        top_k_review([0.5], 2)


def test_cost_assumptions_and_precision_constraint():
    assert CostPolicy().threshold == pytest.approx(5 / 105)
    with pytest.raises(ValueError):
        _ = CostPolicy(0).threshold
    threshold = precision_constrained_threshold([1, 0, 1], [0.9, 0.8, 0.7], 1)
    assert threshold == 0.9
    assert np.isinf(precision_constrained_threshold([0, 0], [0.9, 0.8], 0.5))
