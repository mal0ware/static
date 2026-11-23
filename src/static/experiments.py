"""The experiments behind every table in ``results/``.

Each function takes a :class:`Config` and returns a table. Random draws come from
generators seeded by the configuration and the experiment's own key, so tables are
identical from run to run and independent of the order the experiments run in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from static.absorption import default_components, rolling_absorption_ratio, standardized_shift
from static.cleaning import (
    clip_covariance,
    sample_correlation,
    sample_covariance,
    sorted_eigh,
)
from static.feasibility import (
    es_critical_ratio,
    minimise_expected_shortfall,
    minimise_maximal_loss,
    ml_feasible_probability,
    ml_feasible_probability_normal,
)
from static.mp import fit_noise_variance, ks_distance, mp_bounds
from static.portfolio import (
    market_mode_covariance,
    min_variance_weights,
    portfolio_variance,
    q0_theory,
    q1_theory,
    simulate_noise_ratios,
)

FloatArray = NDArray[np.float64]
Frame = Any  # pandas is treated as untyped

TRADING_DAYS = 252
ES_BETA = 0.7

#: Episodes named in the README, as (label, first day, last day).
EPISODES: tuple[tuple[str, str, str], ...] = (
    ("2008", "2007-07-02", "2009-06-30"),
    ("2020", "2020-01-02", "2020-12-31"),
    ("2022", "2022-01-03", "2022-12-30"),
)


@dataclass(frozen=True)
class Config:
    """Run settings. ``fast`` shrinks every simulation for the CI smoke run."""

    fast: bool = False
    seed: int = 20261123

    def rng(self, *key: int) -> np.random.Generator:
        """A generator tied to this seed and an experiment key."""
        return np.random.default_rng([self.seed, *key])


# --------------------------------------------------------------------------------------
# 1. Out-of-sample risk against N/T (Pafka and Kondor 2002; Kondor, Pafka and Nagy 2007)
# --------------------------------------------------------------------------------------


def divergence_table(cfg: Config) -> Frame:
    """Mean ``q0``, ``q1`` and ``q2`` against ``N/T`` for unit covariance."""
    if cfg.fast:
        sizes = {50: 20, 100: 20}
        ratios = [0.2, 0.5, 0.8, 0.9, 0.95]
    else:
        sizes = {100: 200, 200: 100, 400: 50}
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    rows = []
    for n, reps in sizes.items():
        for j, ratio in enumerate(ratios):
            t = round(n / ratio)
            rng = cfg.rng(1, n, j)
            draws = [simulate_noise_ratios(n, t, rng) for _ in range(reps)]
            q0 = np.array([d.q0 for d in draws])
            q1 = np.array([d.q1 for d in draws])
            q2 = np.array([d.q2 for d in draws])
            rows.append(
                {
                    "N": n,
                    "T": t,
                    "N/T": n / t,
                    "samples": reps,
                    "q0_mean": q0.mean(),
                    "q0_sd": q0.std(ddof=1),
                    "q0_theory": q0_theory(n / t),
                    "q0_mean/theory": q0.mean() / q0_theory(n / t),
                    "q1_mean": q1.mean(),
                    "q1_theory": q1_theory(n / t),
                    "q2_mean": q2.mean(),
                }
            )
    return pd.DataFrame(rows)


#: Pafka and Kondor (2002) Table 1 as printed: model, n, T, q0, q1, q2, q2/q1.
PAFKA_KONDOR_TABLE1: tuple[tuple[str, int, int, float, float, float, float], ...] = (
    ("I", 100, 600, 1.09, 0.92, 1.09, 1.19),
    ("I", 500, 3000, 1.09, 0.91, 1.10, 1.20),
    ("I", 500, 1500, 1.22, 0.81, 1.22, 1.49),
    ("I", 500, 750, 1.73, 0.57, 1.74, 3.00),
    ("II", 500, 3000, 1.09, 0.91, 1.10, 1.20),
    ("II", 500, 750, 1.72, 0.58, 1.72, 2.97),
)


def pafka_kondor_table(cfg: Config) -> Frame:
    """Re-run Pafka and Kondor's Table 1. In fast mode ``n`` is divided by five."""
    scale = 5 if cfg.fast else 1
    reps = 10 if cfg.fast else 50
    rows = []
    for j, (model, n_paper, t_paper, p_q0, p_q1, p_q2, p_ratio) in enumerate(PAFKA_KONDOR_TABLE1):
        n, t = n_paper // scale, t_paper // scale
        cov = market_mode_covariance(n) if model == "II" else None
        rng = cfg.rng(2, j)
        draws = [simulate_noise_ratios(n, t, rng, true_cov=cov) for _ in range(reps)]
        q0 = np.array([d.q0 for d in draws])
        q1 = np.array([d.q1 for d in draws])
        q2 = np.array([d.q2 for d in draws])
        rows.append(
            {
                "model": model,
                "n": n,
                "T": t,
                "samples": reps,
                "q0": q0.mean(),
                "q0_sd": q0.std(ddof=1),
                "q0_paper": p_q0,
                "q1": q1.mean(),
                "q1_sd": q1.std(ddof=1),
                "q1_paper": p_q1,
                "q2": q2.mean(),
                "q2_sd": q2.std(ddof=1),
                "q2_paper": p_q2,
                "q2/q1": (q2 / q1).mean(),
                "q2/q1_paper": p_ratio,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Marchenko-Pastur against simulated spectra (Laloux et al. 1999)
# --------------------------------------------------------------------------------------


def noise_eigenvalues(n: int, t: int, samples: int, rng: np.random.Generator) -> FloatArray:
    """Pooled eigenvalues of ``samples`` pure-noise correlation matrices."""
    pooled = [
        sorted_eigh(sample_correlation(rng.standard_normal((t, n))))[0] for _ in range(samples)
    ]
    return np.concatenate(pooled)


def mp_edge_table(cfg: Config) -> Frame:
    """How fast the finite-size spectrum settles onto the density, at Laloux's Q."""
    q = 1309 / 406
    sizes = [25, 50, 100, 200] if cfg.fast else [25, 50, 100, 200, 400, 800]
    total = 2000 if cfg.fast else 10000
    lo, hi = mp_bounds(q)
    rows = []
    for n in sizes:
        t = round(q * n)
        samples = max(4, total // n)
        rng = cfg.rng(3, n)
        largest, smallest, outside, ks = [], [], [], []
        for _ in range(samples):
            values = sorted_eigh(sample_correlation(rng.standard_normal((t, n))))[0]
            lo_n, hi_n = mp_bounds(t / n)
            largest.append(values[0])
            smallest.append(values[-1])
            outside.append(np.mean((values < lo_n) | (values > hi_n)))
            ks.append(ks_distance(values, t / n))
        rows.append(
            {
                "N": n,
                "T": t,
                "samples": samples,
                "lambda_max_theory": hi,
                "largest_mean": np.mean(largest),
                "largest_max": np.max(largest),
                "lambda_min_theory": lo,
                "smallest_mean": np.mean(smallest),
                "share_outside_edges": np.mean(outside),
                "ks_distance_mean": np.mean(ks),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Real spectra
# --------------------------------------------------------------------------------------


def spectrum_summary(returns: FloatArray) -> dict[str, float]:
    """Laloux-style summary of one ``T x N`` return window."""
    t, n = returns.shape
    q = t / n
    values, _ = sorted_eigh(sample_correlation(returns))
    _, plain_hi = mp_bounds(q)
    fit = fit_noise_variance(values, q)
    noise = values[fit.n_signal :]
    return {
        "N": n,
        "T": t,
        "Q": q,
        "lambda_1": values[0],
        "lambda_max(sigma2=1)": plain_hi,
        "lambda_1/lambda_max": values[0] / plain_hi,
        "lambda_1_share_of_variance": values[0] / n,
        "n_above_plain_edge": int(np.sum(values > plain_hi)),
        "sigma2_fitted": fit.sigma2,
        "n_above_fitted_edge": fit.n_signal,
        "share_of_eigenvalues_in_fitted_band": float(
            np.mean((values >= fit.lambda_min) & (values <= fit.lambda_max))
        ),
        "share_of_eigenvalues_below_fitted_band": float(np.mean(values < fit.lambda_min)),
        "variance_share_above_fitted_edge": float(values[: fit.n_signal].sum() / n),
        "ks_distance_of_bulk": ks_distance(noise, q, fit.sigma2),
    }


def real_spectrum_table(universes: dict[str, Frame]) -> Frame:
    """Spectra of the French portfolios over Laloux's years and recent windows."""
    rows = []
    for name, frame in universes.items():
        last = frame.index[-1]
        windows = [
            ("1991-01-01", "1996-12-31"),
            (str(frame.index[-6 * TRADING_DAYS].date()), str(last.date())),
            (str(frame.index[-TRADING_DAYS].date()), str(last.date())),
        ]
        for start, end in windows:
            window = frame.loc[start:end]
            window = window.loc[:, window.notna().all(axis=0)]
            row: dict[str, Any] = {"universe": name, "start": start, "end": end}
            row.update(spectrum_summary(window.to_numpy(dtype=np.float64)))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Clipping and out-of-sample risk
# --------------------------------------------------------------------------------------


def _clip_plain(cov: FloatArray, q: float) -> FloatArray:
    return clip_covariance(cov, q)


def clipping_sim_table(cfg: Config) -> Frame:
    """``q0`` with and without clipping when the truth is one market mode plus noise."""
    n = 50 if cfg.fast else 100
    reps = 10 if cfg.fast else 100
    ratios = [0.2, 0.5, 0.8, 0.95] if cfg.fast else [0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
    cov = market_mode_covariance(n)
    rows = []
    for j, ratio in enumerate(ratios):
        t = round(n / ratio)
        rng = cfg.rng(4, j)
        raw = np.array([simulate_noise_ratios(n, t, rng, true_cov=cov).q0 for _ in range(reps)])
        clipped = np.array(
            [
                simulate_noise_ratios(n, t, rng, true_cov=cov, cleaner=_clip_plain).q0
                for _ in range(reps)
            ]
        )
        rows.append(
            {
                "N": n,
                "T": t,
                "N/T": n / t,
                "samples": reps,
                "q0_raw": raw.mean(),
                "q0_raw_sd": raw.std(ddof=1),
                "q0_clipped": clipped.mean(),
                "q0_clipped_sd": clipped.std(ddof=1),
                "q0_theory_raw": q0_theory(n / t),
            }
        )
    return pd.DataFrame(rows)


CLIP_METHODS = ("equal weight", "raw", "clipped", "clipped, fitted edge")


def clipping_real_table(
    universes: dict[str, Frame],
    windows: tuple[int, ...] = (63, 126, 252, 504, 1260),
    hold: int = 21,
    start: str = "1969-07-01",
) -> Frame:
    """Minimum-variance portfolios on real returns, estimated on trailing windows.

    Every ``hold`` days the covariance of the last ``T`` days is estimated, cleaned or
    not, and the fully invested minimum-variance portfolio is held for the next ``hold``
    days at fixed weights. All window lengths are scored on the same days, which begin
    once the longest window is available.

    With fewer days than assets the raw matrix cannot be inverted and the fitted edge is
    not defined, so those two rows are empty. One missing value in the industry file is
    set to zero; size and book-to-market portfolios with gaps are left out.
    """
    rows = []
    longest = max(windows)
    for name, frame in universes.items():
        window_frame = frame.loc[start:]
        window_frame = window_frame.loc[
            :, window_frame.notna().sum(axis=0) >= len(window_frame) - 5
        ]
        x = window_frame.fillna(0.0).to_numpy(dtype=np.float64)
        total, n = x.shape
        rebalance = range(longest, total - hold + 1, hold)
        for t in windows:
            realised: dict[str, list[FloatArray]] = {m: [] for m in CLIP_METHODS}
            predicted: dict[str, list[float]] = {m: [] for m in CLIP_METHODS}
            gross: dict[str, list[float]] = {m: [] for m in CLIP_METHODS}
            for now in rebalance:
                past, future = x[now - t : now], x[now : now + hold]
                cov = sample_covariance(past)
                estimates: dict[str, FloatArray | None] = {
                    "equal weight": None,
                    "raw": cov if t > n else None,
                    "clipped": clip_covariance(cov, t / n),
                    "clipped, fitted edge": clip_covariance(cov, t / n, fit=True)
                    if t > n
                    else None,
                }
                for method in CLIP_METHODS:
                    estimate = estimates[method]
                    if method == "equal weight":
                        w = np.full(n, 1.0 / n)
                        predicted[method].append(portfolio_variance(w, cov))
                    elif estimate is None:
                        continue
                    else:
                        w = min_variance_weights(estimate)
                        predicted[method].append(portfolio_variance(w, estimate))
                    realised[method].append(future @ w)
                    gross[method].append(float(np.abs(w).sum()))
            for method in CLIP_METHODS:
                base = {
                    "universe": name,
                    "N": n,
                    "T": t,
                    "N/T": n / t,
                    "method": method,
                    "first_day": str(window_frame.index[longest].date()),
                    "last_day": str(window_frame.index[rebalance[-1] + hold - 1].date()),
                }
                if not realised[method]:
                    rows.append(base | {"rebalances": 0})
                    continue
                returns = np.concatenate(realised[method])
                realised_var = float(np.mean(returns**2) - np.mean(returns) ** 2)
                predicted_var = float(np.mean(predicted[method]))
                rows.append(
                    base
                    | {
                        "rebalances": len(realised[method]),
                        "realised_vol_pct": 100.0 * np.sqrt(TRADING_DAYS * realised_var),
                        "predicted_vol_pct": 100.0 * np.sqrt(TRADING_DAYS * predicted_var),
                        "realised/predicted": np.sqrt(realised_var / predicted_var),
                        "gross_leverage": float(np.mean(gross[method])),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Absorption ratio (Kritzman et al. 2011)
# --------------------------------------------------------------------------------------


def absorption_series(industries: Frame, start: str = "1969-07-01") -> Frame:
    """Daily absorption ratio and its standardised shift for the 49 industries."""
    frame = industries.loc[start:].fillna(0.0)
    ar = rolling_absorption_ratio(frame.to_numpy(dtype=np.float64))
    return pd.DataFrame(
        {"absorption_ratio": ar, "standardized_shift": standardized_shift(ar)}, index=frame.index
    )


def _drawdown(market: Frame) -> tuple[float, Any]:
    wealth = (1.0 + market).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()), drawdown.idxmin()


def absorption_events_table(series: Frame, market: Frame) -> Frame:
    """The absorption ratio around the three episodes named in the README."""
    ar, shift = series["absorption_ratio"], series["standardized_shift"]
    rows = []
    for label, start, end in EPISODES:
        window_ar, window_shift = ar.loc[start:end], shift.loc[start:end]
        depth, trough = _drawdown(market.loc[start:end])
        above = window_shift[window_shift > 1.0]
        rows.append(
            {
                "episode": label,
                "window": f"{start} to {end}",
                "AR_at_start": window_ar.iloc[0],
                "AR_max": window_ar.max(),
                "AR_max_date": str(window_ar.idxmax().date()),
                "AR_at_end": window_ar.iloc[-1],
                "shift_max": window_shift.max(),
                "shift_max_date": str(window_shift.idxmax().date()),
                "first_day_shift_above_1": str(above.index[0].date()) if len(above) else "never",
                "market_drawdown_pct": 100.0 * depth,
                "market_trough_date": str(trough.date()),
            }
        )
    valid = ar.dropna()
    rows.append(
        {
            "episode": "all days",
            "window": f"{valid.index[0].date()} to {valid.index[-1].date()}",
            "AR_at_start": valid.iloc[0],
            "AR_max": valid.max(),
            "AR_max_date": str(valid.idxmax().date()),
            "AR_at_end": valid.iloc[-1],
            "shift_max": shift.max(),
            "shift_max_date": str(shift.idxmax().date()),
        }
    )
    return pd.DataFrame(rows)


#: Kritzman et al. Exhibit 9, annualised return after a one-sigma increase / decrease.
KRITZMAN_EXHIBIT9: dict[int, tuple[float, float]] = {
    1: (-8.28, 9.27),
    5: (-8.44, 10.06),
    21: (-5.86, 12.16),
}


def absorption_returns_table(series: Frame, market: Frame) -> Frame:
    """Market return after one-sigma shifts in the absorption ratio (Exhibit 9 analogue).

    The signal is known at the close of day ``t``; the return runs over days ``t + 1`` to
    ``t + h`` and is annualised as ``mean * 252 / h``.
    """
    shift = series["standardized_shift"]
    market = market.reindex(shift.index)
    log_wealth = np.log1p(market).cumsum()
    periods = [
        ("paper sample", "1998-01-01", "2010-05-10"),
        ("after the paper", "2010-05-11", str(shift.index[-1].date())),
        ("all days", str(shift.dropna().index[0].date()), str(shift.index[-1].date())),
    ]
    rows = []
    for label, start, end in periods:
        for horizon, (paper_up, paper_down) in KRITZMAN_EXHIBIT9.items():
            forward = np.expm1(log_wealth.shift(-horizon) - log_wealth).loc[start:end]
            signal = shift.loc[start:end]
            ok = forward.notna() & signal.notna()
            up, down = forward[ok & (signal > 1.0)], forward[ok & (signal < -1.0)]
            scale = 100.0 * TRADING_DAYS / horizon
            everything = forward[ok]
            diff = up.mean() - down.mean()
            # days overlap by h - 1 and signals cluster, so this standard error is too small
            se = np.sqrt(up.var(ddof=1) / len(up) + down.var(ddof=1) / len(down))
            rows.append(
                {
                    "period": label,
                    "start": start,
                    "end": end,
                    "horizon_days": horizon,
                    "days_shift_above_1": len(up),
                    "days_shift_below_-1": len(down),
                    "after_increase_pct": scale * up.mean(),
                    "after_decrease_pct": scale * down.mean(),
                    "difference_pct": scale * diff,
                    "naive_t_of_difference": diff / se,
                    "all_days_pct": scale * everything.mean(),
                    "paper_after_increase_pct": paper_up if label == "paper sample" else np.nan,
                    "paper_after_decrease_pct": paper_down if label == "paper sample" else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 6. Feasibility (Kondor, Pafka and Nagy 2007; Ciliberti, Kondor and Mezard 2007)
# --------------------------------------------------------------------------------------


def ml_feasibility_table(cfg: Config) -> Frame:
    """Share of maximal-loss problems with a solution against Kondor et al. Eq. (26)."""
    sizes = [10, 20] if cfg.fast else [10, 20, 40, 80]
    reps = 60 if cfg.fast else 400
    ratios = [0.35, 0.5, 0.65] if cfg.fast else [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    rows = []
    for n in sizes:
        for j, ratio in enumerate(ratios):
            t = round(n / ratio)
            rng = cfg.rng(5, n, j)
            hits = sum(
                minimise_maximal_loss(rng.standard_normal((t, n))).bounded for _ in range(reps)
            )
            exact = ml_feasible_probability(n, t)
            rows.append(
                {
                    "N": n,
                    "T": t,
                    "N/T": n / t,
                    "samples": reps,
                    "simulated": hits / reps,
                    "eq26_exact": exact,
                    "eq27_normal": ml_feasible_probability_normal(n, t),
                    "z_score": (hits / reps - exact)
                    / max(np.sqrt(exact * (1.0 - exact) / reps), 1e-12),
                }
            )
    return pd.DataFrame(rows)


def es_feasibility_table(cfg: Config) -> Frame:
    """Share of expected-shortfall problems with a solution at ``beta = 0.7``."""
    sizes = {15: 30, 30: 30} if cfg.fast else {25: 300, 50: 300, 100: 150}
    ratios = (
        [0.34, 0.46, 0.58] if cfg.fast else [0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62]
    )
    critical = es_critical_ratio(ES_BETA)
    rows = []
    for n, reps in sizes.items():
        for j, ratio in enumerate(ratios):
            t = round(n / ratio)
            rng = cfg.rng(6, n, j)
            hits = sum(
                minimise_expected_shortfall(rng.standard_normal((t, n)), ES_BETA).bounded
                for _ in range(reps)
            )
            rows.append(
                {
                    "beta": ES_BETA,
                    "N": n,
                    "T": t,
                    "N/T": n / t,
                    "samples": reps,
                    "simulated": hits / reps,
                    "critical_N/T": critical,
                }
            )
    return pd.DataFrame(rows)


def es_half_crossings(table: Frame) -> Frame:
    """``N/T`` where the simulated feasibility share crosses one half, by interpolation."""
    rows = []
    for n, group in table.groupby("N"):
        group = group.sort_values("N/T")
        x, p = group["N/T"].to_numpy(), group["simulated"].to_numpy()
        crossing = np.nan
        for i in range(len(x) - 1):
            if p[i] >= 0.5 > p[i + 1]:
                crossing = x[i] + (p[i] - 0.5) / (p[i] - p[i + 1]) * (x[i + 1] - x[i])
                break
        rows.append(
            {
                "beta": ES_BETA,
                "N": n,
                "N/T_at_half": crossing,
                "critical_N/T": group["critical_N/T"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def es_boundary_table() -> Frame:
    """The expected-shortfall phase boundary from Ciliberti et al. Eqs. (21), (22), (24)."""
    betas = [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]
    return pd.DataFrame({"beta": betas, "critical_N/T": [es_critical_ratio(b) for b in betas]})


def n_components_used(n_assets: int) -> int:
    """Number of eigenvectors in the absorption ratio for ``n_assets``."""
    return default_components(n_assets)
