"""Linear shrinkage of the sample covariance matrix towards a multiple of the identity.

Ledoit and Wolf, "A well-conditioned estimator for large-dimensional covariance
matrices" (working paper of February 2001; published in the Journal of Multivariate
Analysis 88, 2004). Lemma and equation numbers are those of the working paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Shrinkage:
    """The shrunk covariance matrix and the scalars it was built from.

    ``mean_variance`` is ``m``, ``dispersion`` is ``d^2``, ``error`` is ``b^2`` and
    ``intensity`` is ``b^2 / d^2``, the weight on the identity target.
    """

    covariance: FloatArray
    intensity: float
    mean_variance: float
    dispersion: float
    error: float


def ledoit_wolf(returns: FloatArray, demean: bool = True) -> Shrinkage:
    """Ledoit and Wolf's estimator ``S* = (b^2/d^2) m I + (1 - b^2/d^2) S``. Eq. (14).

    With ``S`` the sample covariance of the ``T x N`` matrix ``returns`` (normalised by
    ``T``) and ``||A||^2 = tr(A A') / N``:

    - ``m = tr(S) / N`` (Lemma 3.2),
    - ``d^2 = ||S - m I||^2`` (Lemma 3.3),
    - ``b^2 = min(d^2, (1/T^2) sum_t ||x_t x_t' - S||^2)`` (Lemma 3.4).

    The sum over observations is computed without forming the ``T`` outer products:
    ``sum_t ||x_t x_t' - S||^2 = (sum_t |x_t|^4 - T tr(S^2)) / N``.

    The paper assumes a known zero mean. With ``demean=True`` the sample mean is
    subtracted first, which the paper does not cover.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns must be a T x N matrix")
    t, n = x.shape
    if t < 2:
        raise ValueError("need at least two observations")
    if demean:
        x = x - x.mean(axis=0)
    sample = x.T @ x / t
    m = float(np.trace(sample) / n)
    centred = sample - m * np.eye(n)
    d2 = float(np.sum(centred**2) / n)
    if d2 <= 0.0:
        return Shrinkage(np.asarray(sample, dtype=np.float64), 0.0, m, 0.0, 0.0)
    fourth = float(np.sum(np.sum(x**2, axis=1) ** 2))
    b2_bar = (fourth - t * float(np.sum(sample**2))) / (n * t**2)
    b2 = min(max(b2_bar, 0.0), d2)
    intensity = b2 / d2
    shrunk = intensity * m * np.eye(n) + (1.0 - intensity) * sample
    return Shrinkage(np.asarray(shrunk, dtype=np.float64), intensity, m, d2, b2)
