"""Sample correlation matrices, their spectra, and eigenvalue clipping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from static.mp import fit_noise_variance, mp_bounds

FloatArray = NDArray[np.float64]


def sample_covariance(returns: FloatArray, demean: bool = True) -> FloatArray:
    """Covariance of a ``T x N`` return matrix, normalised by ``T``.

    With ``demean=False`` this is Pafka and Kondor (2002) Eq. (2): the mean is known to
    be zero and is not estimated.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns must be a T x N matrix")
    if demean:
        x = x - x.mean(axis=0)
    return np.asarray(x.T @ x / x.shape[0], dtype=np.float64)


def correlation_from_covariance(cov: FloatArray) -> FloatArray:
    """Rescale a covariance matrix to unit diagonal."""
    vol = np.sqrt(np.diag(cov))
    if np.any(vol <= 0.0):
        raise ValueError("an asset has zero variance")
    corr = cov / np.outer(vol, vol)
    return np.asarray(0.5 * (corr + corr.T), dtype=np.float64)


def sample_correlation(returns: FloatArray, demean: bool = True) -> FloatArray:
    """Correlation matrix of a ``T x N`` return matrix. Laloux Eq. (1) after rescaling."""
    return correlation_from_covariance(sample_covariance(returns, demean=demean))


def sorted_eigh(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Eigenvalues in descending order with matching eigenvector columns."""
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=np.float64), np.asarray(
        vectors[:, order], dtype=np.float64
    )


def count_signal(eigenvalues: FloatArray, q: float, fit: bool = False) -> int:
    """Number of eigenvalues above the upper Marchenko-Pastur edge.

    ``fit=False`` uses the edge for ``sigma2 = 1``. ``fit=True`` uses the self-consistent
    edge from :func:`static.mp.fit_noise_variance`. When ``Q < 1`` the matrix is
    singular; the edge ``(1 + sqrt(N/T))^2`` is still the top of the nonzero noise
    eigenvalues, so the same formula is used.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    if q >= 1.0:
        if fit:
            return fit_noise_variance(ev, q).n_signal
        return int(np.sum(ev > mp_bounds(q)[1]))
    edge = (1.0 + np.sqrt(1.0 / q)) ** 2
    return int(np.sum(ev > edge))


def clip_correlation(
    corr: FloatArray,
    q: float,
    *,
    fit: bool = False,
    n_keep: int | None = None,
    unit_diagonal: bool = True,
    spectrum: tuple[FloatArray, FloatArray] | None = None,
) -> FloatArray:
    """Eigenvalue clipping. Potters, Bouchaud and Laloux (2005) Eq. (17).

    The ``k*`` largest eigenvalues are kept. Every other eigenvalue is replaced by one
    common value chosen so the trace is preserved, which is their average. Eigenvectors
    are unchanged. ``k*`` is the number of eigenvalues above the Marchenko-Pastur edge
    unless ``n_keep`` is given.

    The clipped matrix no longer has an exactly unit diagonal. With
    ``unit_diagonal=True`` it is rescaled to one; that step is not in the source.

    ``spectrum`` may hold the output of :func:`sorted_eigh` for ``corr`` when the
    caller already has it.
    """
    values, vectors = sorted_eigh(corr) if spectrum is None else spectrum
    n = values.size
    k = count_signal(values, q, fit=fit) if n_keep is None else n_keep
    if not 0 <= k <= n:
        raise ValueError("n_keep out of range")
    cleaned = values.copy()
    if k < n:
        cleaned[k:] = values[k:].mean()
    out = (vectors * cleaned) @ vectors.T
    out = 0.5 * (out + out.T)
    if unit_diagonal:
        out = correlation_from_covariance(out)
    return np.asarray(out, dtype=np.float64)


def clip_covariance(
    cov: FloatArray,
    q: float,
    *,
    fit: bool = False,
    spectrum: tuple[FloatArray, FloatArray] | None = None,
) -> FloatArray:
    """Clip the correlation matrix inside a covariance matrix, keeping sample variances.

    ``spectrum`` is passed on to :func:`clip_correlation`.
    """
    vol = np.sqrt(np.diag(cov))
    cleaned = clip_correlation(correlation_from_covariance(cov), q, fit=fit, spectrum=spectrum)
    return np.asarray(cleaned * np.outer(vol, vol), dtype=np.float64)
