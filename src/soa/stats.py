"""Confidence intervals for eval numbers. Pure functions, no model calls.

Wilson score interval for a count out of n (validity); percentile bootstrap, seeded, for a statistic over documents
(field accuracy). All results are percentages (0-100).
"""

import math
import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")

Z_95 = 1.959963984540054  # two-sided 95% normal quantile
DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 0


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for `successes` out of `n`, in percent. Needs n >= 1 and 0 <= successes <= n."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError(f"need n >= 1 and 0 <= successes <= n, got successes={successes}, n={n}")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def bootstrap_interval(
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval of `statistic` over `items` (resampled with replacement, same size).

    Seeded, so the same items always give the same interval. The documents are the unit that is resampled.
    """
    if not items:
        raise ValueError("bootstrap_interval needs at least one item")
    rng = random.Random(seed)
    size = len(items)
    values = sorted(statistic([items[rng.randrange(size)] for _ in range(size)]) for _ in range(resamples))
    low = values[round(0.025 * resamples)]
    high = values[min(resamples - 1, round(0.975 * resamples) - 1)]
    return low, high
