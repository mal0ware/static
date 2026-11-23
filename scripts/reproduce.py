"""Rebuild every figure and table.

    uv run python scripts/fetch_data.py
    uv run python scripts/reproduce.py          # full run, writes results/
    uv run python scripts/reproduce.py --fast   # smoke run, writes results_fast/

The full run takes several minutes. The fast run uses smaller simulations and finishes in
well under two minutes; its numbers are noisier and are not the ones quoted in the README.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from static import experiments as ex
from static.cleaning import sample_correlation, sorted_eigh
from static.data import FILES, load, vintage
from static.feasibility import ml_feasible_probability
from static.mp import mp_bounds, mp_cdf, mp_pdf
from static.portfolio import q0_theory, q1_theory

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 150,
    }
)


# ------------------------------------------------------------------------------ output


def save_table(table: Any, out: Path, name: str) -> None:
    table.to_csv(out / f"{name}.csv", index=False, float_format="%.6g", lineterminator="\n")


def markdown(table: Any, digits: int = 3) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{value:.{digits}f}"
        return str(value)

    head = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in table.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def finish(fig: Any, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------------------- figures


def fig_divergence(table: Any, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    grid = np.linspace(0.0, 0.985, 400)
    ax.plot(grid, [q0_theory(r) for r in grid], color=INK, lw=1.6, label="theory, 1/√(1 − N/T)")
    ax.plot(
        grid, [q1_theory(r) for r in grid], color=INK, lw=1.6, ls="--", label="theory, √(1 − N/T)"
    )
    for k, (n, group) in enumerate(table.groupby("N")):
        se = group["q0_sd"] / np.sqrt(group["samples"])
        ax.errorbar(
            group["N/T"],
            group["q0_mean"],
            yerr=2 * se,
            fmt=MARKERS[k],
            color=SERIES[k],
            mec=SURFACE,
            mew=0.8,
            capsize=2,
            elinewidth=1,
            label=f"true risk, N = {n}",
        )
        ax.plot(
            group["N/T"],
            group["q1_mean"],
            MARKERS[k],
            mfc=SURFACE,
            mec=SERIES[k],
            mew=1.4,
            label=f"predicted risk, N = {n}",
        )
    ax.axvline(1.0, color=MUTED, lw=1, ls=":")
    ax.text(0.992, 2.4, "N = T", color=MUTED, fontsize=9, ha="right", va="center", rotation=90)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 8.0)
    ax.set_xlabel("assets per observation, N/T")
    ax.set_ylabel("risk of the estimated portfolio / risk of the true optimum")
    ax.set_title("True risk diverges as N approaches T while predicted risk goes to zero")
    ax.legend(loc="upper left", ncol=2)
    finish(fig, out, "fig1_divergence")


def fig_mp_simulated(cfg: ex.Config, out: Path) -> None:
    samples = 4 if cfg.fast else 20
    cases = [
        (406, 1309, "N = 406, T = 1309 (Laloux et al.'s sizes)"),
        (400, 420, "N = 400, T = 420"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for k, (ax, (n, t, title)) in enumerate(zip(axes, cases, strict=True)):
        values = ex.noise_eigenvalues(n, t, samples, cfg.rng(7, k))
        lo, hi = mp_bounds(t / n)
        top = hi * 1.08
        ax.hist(
            values,
            bins=np.linspace(0.0, top, 61),
            density=True,
            color="#9ec5f4",
            edgecolor=SURFACE,
            linewidth=0.5,
            label=f"simulated, {samples} matrices",
        )
        grid = np.linspace(max(lo, 1e-4), hi, 2000)
        ax.plot(grid, mp_pdf(grid, t / n), color=INK, lw=1.6, label="Marchenko-Pastur")
        ax.set_xlim(0.0, top)
        ax.set_ylim(0.0, 1.25 if k == 0 else 2.5)
        ax.set_xlabel("eigenvalue λ")
        ax.set_ylabel("density ρ(λ)")
        ax.set_title(title)
        ax.legend(loc="upper right")
    finish(fig, out, "fig2_mp_simulated")


def fig_mp_real(universes: dict[str, Any], out: Path) -> None:
    cases = [
        ("industries49", "49 industry portfolios", "1991-01-01", "1996-12-31"),
        ("size_bm_100", "100 size and book-to-market portfolios", None, None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (name, label, start, end) in zip(axes, cases, strict=True):
        frame = universes[name]
        window = frame.iloc[-ex.TRADING_DAYS :] if start is None else frame.loc[start:end]
        window = window.loc[:, window.notna().all(axis=0)]
        x = window.to_numpy(dtype=np.float64)
        t, n = x.shape
        values = sorted_eigh(sample_correlation(x))[0]
        lo, hi = mp_bounds(t / n)
        grid = np.linspace(lo, hi, 4001)
        ranks = np.arange(1, n + 1)
        noise = np.interp(1.0 - (ranks - 0.5) / n, mp_cdf(grid, t / n), grid)
        ax.axhspan(lo, hi, color="#e9e8e4", lw=0, label="Marchenko-Pastur band")
        ax.plot(ranks, noise, color=INK, lw=1.6, label="pure noise")
        ax.plot(ranks, values, "o", color=SERIES[0], ms=4, mec=SURFACE, mew=0.4, label="measured")
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue rank")
        ax.set_ylabel("eigenvalue λ (log scale)")
        ax.set_title(f"{label}\n{window.index[0].date()} to {window.index[-1].date()}, T = {t}")
        ax.legend(loc="upper right")
    finish(fig, out, "fig3_mp_real")


def fig_clipping_real(table: Any, out: Path) -> None:
    titles = {
        "industries49": "49 industry portfolios",
        "size_bm_100": "size and book-to-market portfolios",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, (name, group) in zip(axes, table.groupby("universe", sort=False), strict=True):
        equal = group[group["method"] == "equal weight"]["realised_vol_pct"].iloc[0]
        ax.axhline(equal, color=MUTED, lw=1.4, ls=":", label="equal weight")
        for k, method in enumerate(["raw", "clipped", "clipped, fitted edge"]):
            rows = group[(group["method"] == method) & (group["rebalances"] > 0)]
            ax.plot(
                rows["N/T"],
                rows["realised_vol_pct"],
                marker=MARKERS[k],
                color=SERIES[k],
                mec=SURFACE,
                mew=0.8,
                label=method,
            )
        ax.set_xscale("log")
        ticks = sorted(group["N/T"].unique())
        ax.set_xticks(ticks, [f"{v:.2f}" for v in ticks])
        ax.minorticks_off()
        ax.set_xlabel("assets per observation, N/T (log scale)")
        ax.set_title(f"{titles[name]}, N = {group['N'].iloc[0]}")
    axes[0].set_ylabel("realised volatility, % a year")
    axes[0].legend(loc="upper left")
    finish(fig, out, "fig4_clipping_real")


def fig_clipping_sim(table: Any, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid = np.linspace(0.0, 0.985, 300)
    ax.plot(grid, [q0_theory(r) for r in grid], color=INK, lw=1.6, label="theory without cleaning")
    ax.plot(table["N/T"], table["q0_raw"], MARKERS[0], color=SERIES[0], mec=SURFACE, label="raw")
    ax.plot(
        table["N/T"],
        table["q0_clipped"],
        marker=MARKERS[1],
        color=SERIES[1],
        mec=SURFACE,
        label="clipped",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 8.0)
    ax.set_xlabel("assets per observation, N/T")
    ax.set_ylabel("true risk / risk of the true optimum, q0")
    ax.set_title(f"One market mode plus noise, N = {table['N'].iloc[0]}")
    ax.legend(loc="upper left")
    finish(fig, out, "fig5_clipping_sim")


def fig_absorption(series: Any, out: Path) -> None:
    window = series.loc["1996-01-01":]
    fig, axes = plt.subplots(
        2, 1, figsize=(9.2, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    for ax in axes:
        for label, start, end in ex.EPISODES:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#e9e8e4", lw=0)
            if ax is axes[0]:
                ax.text(
                    pd.Timestamp(start),
                    0.97,
                    f" {label}",
                    color=MUTED,
                    fontsize=9,
                    va="top",
                    transform=ax.get_xaxis_transform(),
                )
    axes[0].plot(window.index, window["absorption_ratio"], color=SERIES[0], lw=1.4)
    axes[0].set_ylabel("absorption ratio")
    axes[0].set_title(
        "Share of variance in the top 10 of 49 eigenvectors (500-day window, 250-day half-life)"
    )
    axes[1].plot(window.index, window["standardized_shift"], color=SERIES[0], lw=1.0)
    for level in (-1.0, 1.0):
        axes[1].axhline(level, color=MUTED, lw=1, ls=":")
    axes[1].set_ylabel("standardised shift, σ")
    axes[1].set_xlabel("date")
    finish(fig, out, "fig6_absorption")


def fig_feasibility(ml: Any, es: Any, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    ax = axes[0]
    for k, (n, group) in enumerate(ml.groupby("N")):
        ts = np.arange(round(n / 0.72), round(n / 0.28) + 1)
        ax.plot(n / ts, [ml_feasible_probability(n, int(t)) for t in ts], color=SERIES[k], lw=1.4)
        ax.plot(
            group["N/T"],
            group["simulated"],
            MARKERS[k],
            color=SERIES[k],
            mec=SURFACE,
            mew=0.8,
            label=f"N = {n}",
        )
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("assets per observation, N/T")
    ax.set_ylabel("share of samples with a solution")
    ax.set_title("Maximal loss: points simulated, lines Eq. (26)")
    ax.legend(loc="upper right")
    ax = axes[1]
    for k, (n, group) in enumerate(es.groupby("N")):
        ax.plot(
            group["N/T"],
            group["simulated"],
            marker=MARKERS[k],
            color=SERIES[k],
            mec=SURFACE,
            mew=0.8,
            lw=1.4,
            label=f"N = {n}",
        )
    critical = es["critical_N/T"].iloc[0]
    ax.axvline(critical, color=INK, lw=1.4, ls="--", label=f"boundary, N/T = {critical:.3f}")
    ax.set_xlabel("assets per observation, N/T")
    ax.set_title(f"Expected shortfall, β = {ex.ES_BETA}: simulated")
    ax.legend(loc="upper right")
    finish(fig, out, "fig7_feasibility")


def fig_es_boundary(crossings: Any, out: Path) -> None:
    from static.feasibility import es_critical_ratio

    betas = np.linspace(0.01, 0.995, 300)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        betas, [es_critical_ratio(float(b)) for b in betas], color=INK, lw=1.6, label="boundary"
    )
    for k, row in enumerate(crossings.itertuples(index=False)):
        if np.isfinite(row[2]):
            ax.plot(
                row[0],
                row[2],
                MARKERS[k],
                color=SERIES[k],
                mec=SURFACE,
                label=f"simulated, N = {row[1]}",
            )
    ax.text(0.25, 0.42, "no solution", color=MUTED, fontsize=10)
    ax.text(0.62, 0.12, "solution exists", color=MUTED, fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.6)
    ax.set_xlabel("expected-shortfall level β")
    ax.set_ylabel("assets per observation, N/T")
    ax.set_title("Where expected-shortfall optimisation stops having a solution")
    ax.legend(loc="upper left")
    finish(fig, out, "fig8_es_boundary")


# -------------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="small simulations for a smoke run")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    cfg = ex.Config(fast=args.fast)
    out: Path = args.out or (ROOT / ("results_fast" if args.fast else "results"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def step(message: str) -> None:
        print(f"[{time.perf_counter() - started:6.1f}s] {message}", flush=True)

    try:
        industries = load("industries49", args.data_dir)
        size_bm = load("size_bm_100", args.data_dir)
        factors = load("factors", args.data_dir)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    universes = {"industries49": industries, "size_bm_100": size_bm}
    market = factors["Mkt-RF"] + factors["RF"]
    vintages = [
        f"{name}: {vintage((args.data_dir / csv).read_text(encoding='latin-1'))} "
        f"Last day {load(name, args.data_dir).index[-1].date()}."
        for name, (_, csv) in FILES.items()
    ]
    (out / "data_vintage.txt").write_text(
        "\n".join(vintages) + "\n", encoding="utf-8", newline="\n"
    )

    tables: list[tuple[str, str, Any]] = []

    step("1/6 out-of-sample risk against N/T")
    divergence = ex.divergence_table(cfg)
    pafka = ex.pafka_kondor_table(cfg)
    fig_divergence(divergence, out)
    tables += [
        ("table1_divergence", "Risk ratios against N/T, unit covariance", divergence),
        ("table2_pafka_kondor", "Pafka and Kondor (2002) Table 1 re-run", pafka),
    ]

    step("2/6 Marchenko-Pastur against simulated spectra")
    edges = ex.mp_edge_table(cfg)
    fig_mp_simulated(cfg, out)
    tables.append(("table3_mp_edges", "Finite-size spectra against the density, Q = 3.22", edges))

    step("3/6 real spectra")
    spectra = ex.real_spectrum_table(universes)
    fig_mp_real(universes, out)
    tables.append(("table4_real_spectra", "Spectra of French portfolio correlations", spectra))

    step("4/6 clipping")
    clip_sim = ex.clipping_sim_table(cfg)
    clip_real = ex.clipping_real_table(universes)
    fig_clipping_sim(clip_sim, out)
    fig_clipping_real(clip_real, out)
    tables += [
        ("table5_clipping_sim", "Clipping when the truth is one market mode", clip_sim),
        ("table6_clipping_real", "Minimum-variance portfolios on real returns", clip_real),
    ]

    step("5/6 absorption ratio")
    series = ex.absorption_series(industries)
    events = ex.absorption_events_table(series, market)
    after = ex.absorption_returns_table(series, market)
    monthly = series.dropna(subset=["absorption_ratio"]).resample("ME").last().reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
    fig_absorption(series, out)
    save_table(monthly, out, "absorption_ratio_monthly")
    tables += [
        ("table7_absorption_events", "Absorption ratio around three episodes", events),
        ("table8_absorption_returns", "Market return after one-sigma shifts", after),
    ]

    step("6/6 feasibility under maximal loss and expected shortfall")
    ml = ex.ml_feasibility_table(cfg)
    es = ex.es_feasibility_table(cfg)
    crossings = ex.es_half_crossings(es)
    boundary = ex.es_boundary_table()
    fig_feasibility(ml, es, out)
    fig_es_boundary(crossings, out)
    tables += [
        ("table9_ml_feasibility", "Maximal loss: share of samples with a solution", ml),
        ("table10_es_feasibility", "Expected shortfall: share of samples with a solution", es),
        ("table11_es_crossings", "Expected shortfall: where the share crosses one half", crossings),
        ("table12_es_boundary", "Expected-shortfall phase boundary", boundary),
    ]

    mode = "fast (smoke run, small simulations)" if cfg.fast else "full"
    lines = [f"# Results\n\nMode: {mode}. Seed {cfg.seed}.\n", "```", *vintages, "```\n"]
    for name, title, table in tables:
        save_table(table, out, name)
        lines += [f"## {title}\n", f"`{name}.csv`\n", markdown(table), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    step(f"done, wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
