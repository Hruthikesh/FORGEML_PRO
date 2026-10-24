"""Operational policies fitted on development predictions, never final holdout."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostPolicy:
    false_positive_cost: float = 5.0
    false_negative_cost: float = 100.0
    version: str = "cost-fp5-fn100-v1"

    @property
    def threshold(self) -> float:
        if min(self.false_positive_cost, self.false_negative_cost) <= 0:
            raise ValueError("Costs must be positive assumed units")
        return self.false_positive_cost / (self.false_positive_cost + self.false_negative_cost)


def top_k_review(probabilities, budget):
    probabilities = np.asarray(probabilities, dtype=float)
    if budget < 0 or budget > len(probabilities):
        raise ValueError("Budget must be between zero and transaction count")
    selected = np.zeros(len(probabilities), dtype=bool)
    selected[np.argsort(-probabilities, kind="stable")[:budget]] = True
    return selected


def precision_constrained_threshold(labels, probabilities, minimum_precision=0.8):
    """Choose greatest recall subject to empirical development precision constraint."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not 0 < minimum_precision <= 1 or len(labels) != len(probabilities):
        raise ValueError("Invalid precision constraint or mismatched inputs")
    eligible = []
    for threshold in np.unique(probabilities):
        mask = probabilities >= threshold
        precision = labels[mask].mean()
        recall = labels[mask].sum() / max(1, labels.sum())
        if precision >= minimum_precision:
            eligible.append((recall, float(threshold)))
    return max(eligible, default=(0, float("inf")))[1]
