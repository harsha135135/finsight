"""Cluster bootstrap confidence intervals.

Questions drawn from the same filing are not independent, so resampling is done
over filing groups, not individual questions. Paired differences resample the
same groups for both configurations.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

DEFAULT_B = 2000
DEFAULT_SEED = 20261106


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def cluster_bootstrap_mean(values: Mapping[str, float], groups: Mapping[str, str],
                           b: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Mean of per-item values with a 95% CI from resampling groups. Returns (mean, low, high)."""
    by_group: dict[str, list[float]] = defaultdict(list)
    for item, value in values.items():
        by_group[groups[item]].append(value)
    keys = sorted(by_group)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    total = sum(sum(v) for v in by_group.values())
    count = sum(len(v) for v in by_group.values())
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        sample = [by_group[rng.choice(keys)] for _ in keys]
        n = sum(len(s) for s in sample)
        means.append(sum(sum(s) for s in sample) / n)
    means.sort()
    return total / count, _percentile(means, 0.025), _percentile(means, 0.975)


def paired_difference(a: Mapping[str, float], b_values: Mapping[str, float], groups: Mapping[str, str],
                      b: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Mean of (a - b) over shared items, with a group-resampled 95% CI."""
    shared = sorted(set(a) & set(b_values))
    return cluster_bootstrap_mean({i: a[i] - b_values[i] for i in shared}, groups, b, seed)
