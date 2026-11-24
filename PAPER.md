# Sources, formulas and comparison

Every formula in `src/static/` is listed here with where it was read. Section 3 compares
each claim in the sources with what this repository measured. Section 4 lists what
differs and why. Section 5 lists the variants tried and what was chosen after seeing
results.

All numbers come from `results/`, written by `uv run python scripts/reproduce.py` with
seed 20261123.

## 1. Sources and what was read

| Short name | Citation | Copy read |
| --- | --- | --- |
| MP | V. A. Marchenko and L. A. Pastur, "Distribution of eigenvalues for some sets of random matrices", Math. USSR-Sbornik 1(4), 457-483, 1967 | **Not read.** The density is taken from LCBP Eq. (3) and PK Eq. (3), which agree with each other. |
| LCBP | L. Laloux, P. Cizeau, J.-P. Bouchaud and M. Potters, "Noise Dressing of Financial Correlation Matrices", Phys. Rev. Lett. 83(7), 1467-1470, 1999 | arXiv:cond-mat/9810255v1, all 3 pages |
| PBL | M. Potters, J.-P. Bouchaud and L. Laloux, "Financial Applications of Random Matrix Theory: Old Laces and New Pieces", Acta Physica Polonica B 36(9), 2005 | arXiv:physics/0507111v1, section on cleaning. Read because LCBP does not define clipping. |
| PK | S. Pafka and I. Kondor, "Noisy covariance matrices and portfolio optimization II", Physica A 319, 487-494, 2003 | arXiv:cond-mat/0205119v2, all 9 pages. KPN cites it for the formula and it holds the table re-run here. |
| KPN | I. Kondor, S. Pafka and G. Nagy, "Noise sensitivity of portfolio selection under various risk measures", J. Banking and Finance 31(5), 1545-1573, 2007 | arXiv:physics/0611027v1, sections 2.3, 2.4, 3.2, 3.3 and the expected-shortfall feasibility section |
| CKM | S. Ciliberti, I. Kondor and M. Mezard, "On the feasibility of portfolio optimization under expected shortfall", Quantitative Finance 7(4), 389-396, 2007 | arXiv:physics/0606015, sections II to IV |
| LW | O. Ledoit and M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices", J. Multivariate Analysis 88(2), 365-411, 2004 | The working paper of February 2001 (ledoit.net/ole1a.pdf), section 3.3. The journal version was not read; numbering may differ. |
| KLPR | M. Kritzman, Y. Li, S. Page and R. Rigobon, "Principal Components as a Measure of Systemic Risk", J. Portfolio Management 37(4), 112-126, 2011 | The 2010 working-paper version (web.mit.edu/finlunch/Fall10/PCASystemicRisk.pdf). The journal version was not read; exhibit numbers may differ. |

Equation numbers below are those of the copies read. Journal versions may number
differently.

## 2. Formulas implemented

### 2.1 Marchenko-Pastur density. `static.mp`

N assets, T observations, Q = T/N ≥ 1, entries of variance σ².

- Sample correlation matrix, LCBP Eq. (1): C = (1/T) M Mᵀ, with returns demeaned and
  rescaled to unit volatility (LCBP footnote). `cleaning.sample_correlation`.
- Density, LCBP Eq. (3):

  ρ(λ) = Q / (2πσ²) · √((λ_max − λ)(λ − λ_min)) / λ,  λ_max,min = σ²(1 + 1/Q ± 2√(1/Q)).

  `mp.mp_pdf`, `mp.mp_bounds`. PK Eq. (3) is the same with r = 1/Q and σ² = 1:
  ρ(λ) = √((λ₊ − λ)(λ − λ₋)) / (2πrλ), λ± = (1 ± √r)².
- Noise variance once the market mode is removed, LCBP text after Eq. (3):
  σ² = 1 − λ₁/N. `mp.fit_noise_variance` iterates this idea: see 4.1.

### 2.2 Eigenvalue clipping. `static.cleaning`

- PBL Eq. (17): λ_c^k = 1 − δ for k > k*, λ_c^k = λ_E^k for k ≤ k*, with δ chosen so the
  trace is preserved, and k* set by the Marchenko-Pastur edge. Preserving the trace
  makes the common value the average of the replaced eigenvalues.
  `cleaning.clip_correlation`.

### 2.3 Minimum-variance portfolio and its risk. `static.portfolio`

- Weights, PK Eq. (1): w*_i = Σ_j σ⁻¹_ij / Σ_jk σ⁻¹_jk. `min_variance_weights`.
- Sample covariance without mean subtraction, PK Eq. (2): σ⁽¹⁾_ij = (1/T) Σ_t y_it y_jt.
  `cleaning.sample_covariance(demean=False)`.
- Risk ratios, PK section 2 items 1 to 4 and KPN Eq. (6). With w⁽⁰⁾ the optimum under
  the true covariance σ⁽⁰⁾ and w⁽¹⁾ the optimum under the sample covariance σ⁽¹⁾:

  q0² = w⁽¹⁾ᵀσ⁽⁰⁾w⁽¹⁾ / w⁽⁰⁾ᵀσ⁽⁰⁾w⁽⁰⁾ (true risk),
  q1² = w⁽¹⁾ᵀσ⁽¹⁾w⁽¹⁾ / w⁽⁰⁾ᵀσ⁽⁰⁾w⁽⁰⁾ (predicted risk),
  q2² = w⁽¹⁾ᵀσ⁽²⁾w⁽¹⁾ / w⁽⁰⁾ᵀσ⁽⁰⁾w⁽⁰⁾ (realised risk in a second sample).

  `simulate_noise_ratios`.
- Limit for large N and T at fixed r = N/T, PK Eq. (4):
  q0 = √(∫ρ(λ)/λ² dλ) / ∫ρ(λ)/λ dλ. PK text after Eq. (4) and KPN Eq. (7):
  q0 = 1/√(1 − N/T). PK text after Eq. (4): q1 = √(1 − N/T).
  `q0_from_density`, `q0_theory`, `q1_theory`.
- Model II, PK section 2: one eigenvalue "about 25 times larger than the rest" along
  (1, …, 1), identity elsewhere. `market_mode_covariance`.

- Exact law at finite size. Not from the sources. For Gaussian returns with a known
  zero mean, the sample covariance of T observations is Wishart with T degrees of
  freedom, and for any true covariance

  q0² = 1 + (N − 1)/(T − N + 2) · F, F distributed as F(N − 1, T − N + 2).

  So the mean of q0² is (T − 1)/(T − N), which tends to 1/(1 − N/T), and its variance
  is infinite unless T > N + 2. `q0_exact_rms`, `q0_exact_quantile`, `q0_exact_mean`.
  The law was written down from the standard distribution of quadratic forms in an
  inverse Wishart matrix and then checked by Kolmogorov-Smirnov tests against
  simulation (`tests/test_portfolio.py`). No paper was read for it.

### 2.4 Linear shrinkage. `static.shrinkage`

- LW Eq. (14): S* = (b²/d²) m I + (1 − b²/d²) S, with S the sample covariance and
  ‖A‖² = tr(AAᵀ)/N. LW Lemma 3.2: m = tr(S)/N. Lemma 3.3: d² = ‖S − mI‖². Lemma 3.4:
  b² = min(d², (1/T²) Σ_t ‖x_t x_tᵀ − S‖²). `ledoit_wolf`. The sum over observations
  is computed as (Σ_t |x_t|⁴ − T tr(S²))/N, and the tests compare it with the sum
  written out.

### 2.5 Absorption ratio. `static.absorption`

- KLPR Eq. (1): AR = Σ_{i=1..n} σ²_Ei / Σ_{j=1..N} σ²_Aj, the variance of the first n
  eigenvectors over the total variance of the N assets. `absorption_ratio`.
- KLPR Part III: 500-day window, n about one fifth of N (51 industries, 10 eigenvectors),
  variances exponentially weighted with a half-life of 250 days.
  `rolling_absorption_ratio`.
- KLPR Eq. (2): ΔAR = (AR_15day − AR_1year) / σ, with 15-day and one-year moving
  averages and σ the standard deviation of the one-year absorption ratio.
  `standardized_shift`.

### 2.6 Feasibility under maximal loss and expected shortfall. `static.feasibility`

- Expected-shortfall programme, KPN Eqs. (16) to (19):
  min v + Σ_t u_t / ((1 − β)T) subject to u_t ≥ −v − Σ_i w_i x_it, u_t ≥ 0, Σ_i w_i = 1.
  `minimise_expected_shortfall`.
- Maximal-loss programme, KPN Eqs. (21) to (23): min u subject to u ≥ −Σ_i w_i x_it for
  all t, Σ_i w_i = 1. `minimise_maximal_loss`.
- Probability that the maximal-loss problem has a solution, KPN Eq. (26):
  p = 2^−(T−1) Σ_{k=N−1..T−1} C(T−1, k). `ml_feasible_probability`.
- Its approximation, KPN Eqs. (27) to (29): p ≈ 1 − Φ(z), z = 2(N/T − 1/2)√T.
  `ml_feasible_probability_normal`.
- Expected-shortfall phase boundary, CKM Eqs. (20) to (22) and (24). With
  g(x) = 0 for x ≥ 0, x² for −1 ≤ x < 0, −2x − 1 for x < −1, and t = T/N:

  (21) 1 − β + (1/2√π) ∫ds e^(−s²) g′(v + s√(2q0)) = 0
  (22) −1 + t/√(2πq0) ∫ds e^(−s²) s g′(v + s√(2q0)) = 0
  (24) t(1 − β)v − q0/2 + (t/2√π) ∫ds e^(−s²) g(v + s√(2q0)) ≥ 0

  The boundary is where (24) holds with equality. `es_critical_ratio`. How the three
  equations are solved is in 4.6.
- Limit, CKM Eq. (25): 1/t* = 1/2 − O((1 − β)³ e^(−1/(4π(1 − β)²))).

## 3. Claim by claim

### 3.1 Pure-noise spectra follow the density (LCBP Eq. (3))

Holds. At LCBP's ratio Q = 3.22 the Kolmogorov distance between one simulated spectrum
and the density falls from 0.067 at N = 25 to 0.004 at N = 800 (`table3_mp_edges.csv`).
The largest eigenvalue sits below the edge 2.424 on average at every size: 2.184 at
N = 25, 2.402 at N = 800. LCBP say the edges are "somewhat blurred" at finite N with a
small probability of eigenvalues outside them. Measured share outside: 0.5% at N = 25,
under 0.15% from N = 200.

### 3.2 Real correlation matrices (LCBP's empirical result)

LCBP, 406 S&P 500 stocks, 1309 days in 1991 to 1996: the top eigenvalue is 25 times
λ_max, σ² = 0.74 fits 94% of the spectrum, and the 6% of eigenvalues above the edge
carry 26% of the variance.

Partly reproduced on stocks (`table13_stock_spectra.csv`). Five back-to-back windows of
1309 days from 2000-10-19 to 2026-10-29, each with the stocks that were index members
on its last day and have a return on every day: 267, 316, 368, 440 and 490 stocks.

- The top eigenvalue is 34 to 72 times λ_max for σ² = 1 and holds 27 to 45% of the
  variance. Between 9 and 18 eigenvalues clear that edge.
- With LCBP's first step, σ² = 1 − λ₁/N (0.55 to 0.73 here), 71 to 77% of the
  eigenvalues lie inside the band, 17 to 23% lie below it and 21 to 27 eigenvalues
  (5 to 8%) lie above it. Those above carry 53 to 68% of the variance.
- The self-consistent fit of 2.1 ends at σ² between 0.18 and 0.36 with 74 to 140
  eigenvalues called signal and 66 to 74% inside the band.
- Scanning σ² from 0.02 to 2 for the band that holds the most eigenvalues gives 74 to
  80%. This scan was added after the first two fits were seen and is an upper bound
  only: it places the band wherever it catches most, whether or not the density fits.

So a large majority of the eigenvalues are compatible with a noise band, and the
market mode stands far above it, as LCBP found. Their 94% and 26% are not matched in
any window. The bulk here is wider than one Marchenko-Pastur density. Possible reasons
that this repository cannot separate: a different period, a universe that loses its
failed members (4.2), unfiltered price data, and LCBP's fit by eye.

Portfolios are further from noise (`table4_real_spectra.csv`). For the 100 size and
book-to-market portfolios in 1991 to 1996 (T = 1518) the top eigenvalue is 34.2 times
λ_max and holds 54% of the variance, 3 eigenvalues are above the edge for σ² = 1, and
no band holds more than 72% of the eigenvalues. For the 49 industry portfolios the top
eigenvalue is 15.4 times λ_max with 44% of the variance, and no band holds more than
45%. Each portfolio is already an average over many stocks, so idiosyncratic noise is
mostly gone before the matrix is formed.

### 3.3 Out-of-sample risk diverges as 1/√(1 − N/T) (PK, KPN Eq. (7))

Holds as a limit. `table1_divergence.csv`, unit covariance, 33 combinations of
N = 100, 200, 400 and N/T from 0.1 to 0.98, with 2000, 1000 and 500 samples.

Ratio of the simulated mean of q0 to the formula, lowest and highest over the three
sizes:

| N/T | lowest | highest |
| --- | --- | --- |
| up to 0.7 | 0.990 | 1.000 |
| 0.8 | 0.987 | 0.999 |
| 0.9 | 0.977 | 0.992 |
| 0.95 | 0.956 | 0.996 |
| 0.98 | 0.881 | 0.952 |

Predicted risk q1 follows √(1 − N/T) within 0.01 up to N/T = 0.9, and realised risk q2
tracks q0.

The shortfall near the critical point is a finite-size effect that the exact law of 2.3
gives in full. The simulated mean is within 2.3 standard errors of the exact mean in
all 33 cells. At N/T = 0.98:

| N | T | mean | s.e. | exact mean | median (95% interval) | exact median | r.m.s. | exact r.m.s. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 102 | 6.29 | 0.07 | 6.31 | 5.49 (5.38 to 5.60) | 5.50 | 7.02 | 7.11 |
| 200 | 204 | 6.60 | 0.07 | 6.70 | 6.08 (5.97 to 6.20) | 6.17 | 7.00 | 7.12 |
| 400 | 408 | 6.80 | 0.07 | 6.92 | 6.54 (6.37 to 6.71) | 6.61 | 6.99 | 7.13 |

The limit is 7.14. The median and the mean rise towards it with N, and the bootstrap
intervals for the three medians do not overlap. At N = 100, T = 102 the F variable has
4 denominator degrees of freedom, so q0² has no finite variance and the sample root
mean square has no usable standard error. KPN note that the standard deviation of q0
grows faster than its mean near the critical point.

### 3.4 PK Table 1

Holds. `table2_pafka_kondor.csv`, 50 samples a row. The paper does not say how many
samples it used.

| model | n | T | q0 here | q0 paper | q1 here | q1 paper | q2/q1 here | q2/q1 paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | 100 | 600 | 1.094 | 1.09 | 0.913 | 0.92 | 1.206 | 1.19 |
| I | 500 | 3000 | 1.095 | 1.09 | 0.913 | 0.91 | 1.202 | 1.20 |
| I | 500 | 1500 | 1.226 | 1.22 | 0.819 | 0.81 | 1.495 | 1.49 |
| I | 500 | 750 | 1.734 | 1.73 | 0.585 | 0.57 | 2.981 | 3.00 |
| II | 500 | 3000 | 1.095 | 1.09 | 0.914 | 0.91 | 1.196 | 1.20 |
| II | 500 | 750 | 1.730 | 1.72 | 0.576 | 0.58 | 3.013 | 2.97 |

The largest gap is q1 in the fourth row, 0.585 against 0.57, with a sample standard
deviation of 0.027 here and 0.02 in the paper. The formula gives 0.577.

### 3.5 Clipping lowers realised risk (PK's reading of Laloux et al. 2000; PBL)

Holds against the raw matrix when N/T is large. It does not beat linear shrinkage.

Simulation with one market mode plus unit noise (`table5_clipping_sim.csv`, N = 100,
200 samples a row, all estimators on the same samples): clipping holds q0 between 1.03
and 1.11 at every N/T up to 0.98, where the raw estimate averages 6.76. The paired
difference, clipped minus raw, runs from −0.024 (s.e. 0.0004) at N/T = 0.1 to −5.6
(s.e. 0.27) at 0.98. Ledoit-Wolf holds q0 between 1.05 and 1.21 and is above clipping
in every row, by 0.02 to 0.11 with standard errors under 0.004. This model is the best
case for clipping, because the truth is exactly what clipping assumes.

Stocks (`table14_stock_min_variance.csv`, 261 rebalances every 21 days from 2005-01-10
to 2026-10-22, 253 to 490 stocks, 370 on average). Realised volatility, % a year:

| T | mean N/T | equal weight | raw | clipped | clipped, fitted edge | Ledoit-Wolf |
| --- | --- | --- | --- | --- | --- | --- |
| 252 | 1.47 | 20.5 | none | 12.4 | none | 12.8 |
| 504 | 0.73 | 20.5 | 24.8 | 12.7 | 12.0 | 12.8 |
| 1260 | 0.29 | 20.5 | 12.8 | 13.2 | 12.1 | 12.2 |

Ratio of realised volatility to that of Ledoit-Wolf, with 95% intervals from a block
bootstrap over holding periods (blocks of 6): clipped 0.97 (0.92 to 1.01), 1.00 (0.93
to 1.07) and 1.08 (1.03 to 1.15) at the three windows; raw 1.94 (1.64 to 2.33) and
1.05 (1.04 to 1.06); fitted edge 0.93 (0.90 to 0.97) and 0.99 (0.97 to 1.02). Clipping
keeps 6.4, 8.5 and 11.7 eigenvalues on average, the fitted edge 73 and 102. The mean
shrinkage weight is 0.065, 0.036 and 0.019.

Portfolios (`table6_clipping_real.csv`, 627 rebalances from 1974-06-27 to 2026-09-18).
For the 49 industries at T = 63 (N/T = 0.78) realised volatility is 11.8% clipped
against 19.3% raw and 11.4% for Ledoit-Wolf. At T = 126 clipped and raw are level
(12.0% and 12.2%). At T = 252, 504 and 1260 the clipped portfolio is worse than the
raw one by 0.8 to 1.2 points. The size and book-to-market portfolios show the same
pattern: 11.6% against 15.5% at T = 126, then worse than raw by 0.5 to 1.0 points.
Ledoit-Wolf is lower than clipping in all ten rows. The ratio of clipped to Ledoit-Wolf
runs from 1.04 to 1.18, and its interval excludes one in nine rows (the exception is
industries at T = 63, 0.998 to 1.084). With these portfolios clipping keeps 1.3 to 2.7
eigenvalues, and 3.2 shows the ones below the edge are not noise.

Predicted risk is too low in every row of both tables. On stocks realised over
predicted volatility is 6.9 raw, 2.0 clipped and 2.4 Ledoit-Wolf at T = 504. On
industries it is 5.3 raw at T = 63 and still 1.20 raw at T = 1260, where the formula
for independent Gaussian returns would give 1.04.

The fitted-edge variant is this repository's own procedure. It is not in the sources
and it was added after seeing the portfolio spectra. On stocks, which it had not seen,
it is the lowest row at T = 504 and level with Ledoit-Wolf at T = 1260. That is one
sample and the variant was not fixed in writing beforehand. See section 5.

### 3.6 Absorption ratio (KLPR)

KLPR Exhibit 7 is a chart and gives no table of levels, so levels are compared by eye
only: their ratio for 51 MSCI USA industries rises through 2007 and 2008 and stays high
into 2010. Here, with 49 French industries, it goes from 0.804 on 2007-07-02 to a peak
of 0.927 on 2008-12-02, the highest value in a sample that starts in 1971, and is
0.914 at the end of June 2009 (`table7_absorption_events.csv`).

In 2020 the ratio goes from 0.804 to 0.914 by 2020-04-09. The standardised shift first
passes one on 2020-03-09, when the market was already falling; the trough is
2020-03-23. In 2022 the ratio does not rise at all: it starts the year at 0.870, its
high for the year is 0.870 on 2022-01-05, and it ends at 0.866, while the market falls
25.4% to 2022-10-14.

KLPR Exhibit 9, annualised stock return after a one-sigma shift, 1998-01-01 to
2010-05-10, against this repository on the same dates (`table8_absorption_returns.csv`):

| horizon | after increase, here | paper | after decrease, here | paper |
| --- | --- | --- | --- | --- |
| 1 day | -4.7% | -8.28% | 8.2% | 9.27% |
| 1 week | -6.1% | -8.44% | 8.4% | 10.06% |
| 1 month | -2.5% | -5.86% | 6.2% | 12.16% |

Same signs, smaller gaps. From 2010-05-11 to 2026-09-30, after the paper's sample, the
sign reverses: returns after an increase are 34.7%, 35.1% and 34.4% a year at the three
horizons, against 9.5%, 10.0% and 10.2% after a decrease. Over all days since 1972
returns are also higher after increases at every horizon.

Forward returns overlap and the signal days come in runs, so days are not independent
observations. The table gives a 95% interval for the difference from a moving-block
bootstrap over days (blocks of 126 days, 2000 resamples) and counts the runs:

| period | horizon | difference, points a year | 95% interval | runs above +1 | runs below −1 |
| --- | --- | --- | --- | --- | --- |
| paper sample | 1 day | −13.0 | −39.0 to 15.8 | 7 | 8 |
| paper sample | 1 week | −14.5 | −42.0 to 14.8 | 7 | 8 |
| paper sample | 1 month | −8.6 | −36.0 to 19.8 | 7 | 8 |
| after the paper | 1 day | 25.2 | 3.4 to 47.3 | 8 | 9 |
| after the paper | 1 week | 25.0 | 4.5 to 46.8 | 8 | 9 |
| after the paper | 1 month | 24.3 | 4.6 to 49.0 | 8 | 9 |
| all days | 1 day | 7.0 | −6.9 to 20.9 | 31 | 35 |
| all days | 1 week | 6.5 | −7.4 to 21.0 | 31 | 35 |
| all days | 1 month | 9.5 | −4.2 to 22.8 | 31 | 35 |

On the paper's dates the difference cannot be told from zero here. After the paper the
interval excludes zero, on 8 runs of increases. Over all days it includes zero. An
earlier version of the table printed a t-statistic that treated days as independent
(9.99 at one month after the paper); it has been removed.

KLPR Exhibit 8 (share of drawdowns preceded by a spike) and Exhibit 10 (a trading rule)
were not re-run.

### 3.7 Maximal loss has a solution with probability KPN Eq. (26)

Holds. `table9_ml_feasibility.csv`: 36 combinations, 400 samples each. 32 are within
two standard errors of Eq. (26), three are between 2 and 2.3, and one is 3.5 away
(N = 20, T = 44: 0.887 against 0.820). `reproduce.py` re-runs every cell that is 3 or
more standard errors out with ten times the samples and a separate random stream
(`table16_ml_recheck.csv`). That cell comes back at 0.816 from 4,000 samples, 0.6
standard errors below the formula, so the first value is read as a chance outlier and
is left in the table. The rule was written after the outlier was seen. A test repeats
the cell with 2,000 samples of its own.

The normal approximation, Eqs. (27) to (29), is at or below the exact sum in every row
and by a wide margin at small sizes: at N = 10, T = 20 it gives 0.500 against 0.676.
Eq. (26) puts the probability at exactly one half on T = 2N - 2, and the approximation
puts it on T = 2N. The two agree as T grows (0.500 against 0.563 at N = 80, T = 160).

### 3.8 Expected shortfall stops having a solution on a line in the (β, N/T) plane (CKM)

Holds where checked. Solving CKM Eqs. (21), (22) and (24) gives the boundary in
`table12_es_boundary.csv`: N/T = 0.099 at β = 0.2, 0.258 at 0.4, 0.408 at 0.6, 0.494 at
0.8, and 0.500 from 0.9 up. CKM give no table. The curve matches their Fig. 2 by eye
(flat at 0.5 above β of about 0.85, through roughly 0.4 at β = 0.6, into the origin).

At β = 0.7 the boundary is N/T = 0.463. The simulated share of solvable samples crosses
one half at N/T = 0.483, 0.470 and 0.469 for N = 25, 50 and 100
(`table11_es_crossings.csv`), and the drop steepens with N as CKM Eq. (28) says. Only
β = 0.7 was simulated.

CKM Eq. (25) says the gap to 1/2 is of order (1 - β)³ e^(-1/(4π(1 - β)²)). The solved
boundary gives a gap of 5.6e-5 at β = 0.880 and 6.8e-3 at β = 0.793, which is 8.3 and
5.0 times that expression. The order of magnitude agrees. The paper does not give the
constant.

## 4. Differences from the sources

### 4.1 Fitting σ²

LCBP fit σ² by eye. `fit_noise_variance` repeats their first step until it stops
changing: call eigenvalues above the edge signal, set σ² to the mean of the rest,
recompute the edge. It divides by the number of remaining eigenvalues where LCBP's
σ² = 1 - λ₁/N divides by N. With one market mode and N = 200 the two differ by under 1%
(tested). Q is kept at T/N throughout.

### 4.2 Data

LCBP use 406 stocks of the S&P 500 and PK about 400 US stocks. The stock panel here is
G. Lim's Zenodo deposit 10.5281/zenodo.23933497, version 1.0.0: daily log returns for
628 tickers from 2000-01-03 to 2026-10-29 and a daily index-membership flag for each.
The deposit states that prices came from Yahoo Finance through FinanceDataReader, that
membership came from a public history of S&P 500 constituents, and that its processed
panels are under CC BY 4.0. It was found by a search of Zenodo after Stooq's downloads
turned out to sit behind a browser check. None of its code or results are used.

The panel has survivorship. It holds the members whose prices could still be
downloaded in 2026. By the deposit's own count (`table15_stock_coverage.csv`) that is
47% of index members in 2000, 53% in 2005, 65% in 2010, 74% in 2015, 87% in 2020 and
99% in 2026. Companies that went bankrupt or were bought are mostly absent. The effect
on levels is plain: realised volatility of every portfolio is understated. The effect
on the comparison between estimators is not known. Membership is used as of each date,
so a stock enters a window or a rebalance only while it was in the index.

Log returns are turned back into simple returns. The first day has no previous price
and is dropped. Nothing else is filtered: bad prints and unadjusted events in the
source would pass through. A stock with no return on a day in the holding period is
given zero for that day.

KLPR use 51 MSCI USA industries. This repository uses the 49 French industries, built
from CRSP with different industry definitions. French files are the vintage built from
the 202609 CRSP database, last day 2026-09-30.

The assets in the portfolio tests are chosen at each rebalance from data up to that
day: a column is used if it has a return on each of the last 1260 days. One day in the
industry file has a missing value for one industry after 1969, so that industry is out
for the rebalances whose window contains it (48 or 49 industries). Between 96 and 100
of the size and book-to-market portfolios qualify. The first version of this test
chose the columns once from the whole sample, which used later data, and kept 95 of
the size and book-to-market portfolios. Fixing it left the industry rows unchanged to
the digits quoted and moved the size and book-to-market rows by up to 0.9 points (raw
at T = 126 went from 14.6% to 15.5%).

### 4.3 Clipping

PBL's Eq. (17) acts on the correlation matrix. Here the clipped matrix is then rescaled
to a unit diagonal and multiplied back by the sample volatilities. The rescaling is not
in PBL. With fewer days than assets the same edge formula (1 + √(N/T))² is used for the
nonzero eigenvalues.

### 4.4 Shrinkage

LW assume a known zero mean. On real returns the sample mean is subtracted first. In
simulation it is not, to match 4.7. Only the identity target of this one paper is
implemented.

### 4.5 Absorption ratio

A year is 252 trading days for KLPR's one-year averages; the paper does not give a
number. The standard deviation uses the same 252 days. Returns after a signal start the
next day and are annualised as mean × 252 / h; KLPR do not state their convention. The
market return is the French value-weighted market (Mkt-RF plus RF), where KLPR use the
MSCI USA index.

### 4.6 Solving the CKM boundary

Their integrals ∫ds e^(-s²) f(v + s√(2q0)) / √π are expectations of f(v + √q0 · Z) for a
standard normal Z, and g is piecewise polynomial, so each has a closed form in the
normal density and distribution (`feasibility._saddle`, checked against quadrature in
the tests). After integrating (22) by parts it reads t · P(-1 < v + √q0 Z < 0) = 1. So
for given v and q0, Eq. (21) gives β and Eq. (22) gives t directly. For each q0 the code
finds the v at which (24) is an equality, which traces the boundary as a curve, and
then finds the q0 whose β is the one asked for. CKM fix β, solve (21) and (22) for each
t, and interpolate the sign change of (24). The two should agree; only the curve's
shape could be compared (3.8). Above β of about 0.92 the returned value is within 2e-8
of 1/2 and is not resolved further. β below 0.002 is not supported.

Whether a sample has a solution is decided by the solver's status: HiGHS reporting
"unbounded" or "infeasible or unbounded" counts as no solution, since equal weights are
always feasible. CKM bound the weights by a large cutoff and let it grow.

### 4.7 Covariance convention

Simulations follow PK Eq. (2) and do not subtract the sample mean. Real-data work
subtracts it.

## 5. Variants tried

Everything that was run is in `results/`. Nothing was dropped. The git history shows
the order: the portfolio results were committed first, and the stock panel, the
Ledoit-Wolf benchmark and the bootstrap intervals were added in a later commit after a
review.

- Clipping edge: two variants, σ² = 1 and the fitted edge. The second was added after
  the portfolio spectra in 3.2 were seen, so its showing on portfolios in 3.5 was found
  after the fact on the same data. Both are reported in every row.
- Estimation windows of 63, 126, 252, 504 and 1260 days and a 21-day holding period
  for portfolios. The stock test was run with one set of windows, 252, 504 and 1260
  days, and the same holding period. Shorter windows were left out because with about
  370 stocks they only add rows where N exceeds T.
- Stock spectra: one window length, 1309 days counted back from the last day, to match
  LCBP. The scan for the widest band was added after the first table was seen.
- Ledoit-Wolf: the identity target only, one run.
- Bootstrap settings, one of each was run: blocks of 6 holding periods for the
  volatility ratios, blocks of 126 days for the absorption returns, 2000 resamples.
  Other block lengths were not tried, so how much the intervals depend on them is not
  known.
- The reference estimator for the ratios is Ledoit-Wolf because it is defined in every
  row.
- Absorption ratio: the paper's 500-day window, 250-day half-life and n = N/5 only.
- Expected shortfall: β = 0.7 only (the level KPN use in their Fig. 14).
- Simulation sizes were chosen for run time. The divergence table went from 200, 100
  and 50 samples to 2000, 1000 and 500 after the review, and the clipping simulation
  from 100 to 200. Seed 20261123 is the only seed used for the committed results. The
  full reproduce was run twice and gave identical tables.
- The re-run of far-out maximal-loss cells in 3.7 was a one-off check outside
  `reproduce.py` at first and is now a step in it.
- Test tolerances changed after the first run of the tests. The normal approximation
  test was loosened from 0.03 to 0.06 once the shift in 3.7 was understood. The
  planted-factor test went from "error under a quarter of raw" to "under half"
  (measured 0.41). The clipped q0 bound went from 1.1 to 1.25 (measured 1.107). The
  critical-line test first assumed T = 2N and was corrected to T = 2N - 2.

## 6. Not verified

- Marchenko and Pastur (1967) was not read; the density comes from LCBP and PK.
- The journal version of KLPR was not read; exhibit numbers are from the 2010 working
  paper.
- Laloux et al. (2000), the paper PK and PBL credit for clipping, was not read.
- The journal version of LW was not read; the estimator is from the 2001 working
  paper.
- The CKM boundary was compared with their Fig. 2 by eye only.
- The exact law of q0 in 2.3 was not checked against a published statement of it,
  only against simulation.
- The stock prices were not checked against a second source.
