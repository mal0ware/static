# Static

**Most of a correlation matrix is noise. Here is how to see which part.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

With 100 stocks and a year of days, how many of the correlations you measured are real? And what happens to an optimizer that believes all of them?

## What this will contain

- The Marchenko-Pastur density against the eigenvalues of simulated and real correlation matrices
- Eigenvalue clipping, and what it does to out-of-sample portfolio risk
- The absorption ratio through 2008, 2020 and 2022
- Out-of-sample risk as the ratio of assets to observations rises, diverging at the critical point

## The result it is built to show

Simulated points landing on the theory curve as the number of assets approaches the number of observations.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Marchenko and Pastur, "Distribution of eigenvalues for some sets of random matrices", Math. USSR-Sbornik 1(4), 1967
- Laloux, Cizeau, Bouchaud and Potters, "Noise Dressing of Financial Correlation Matrices", Physical Review Letters 83(7), 1999
- Kritzman, Li, Page and Rigobon, "Principal Components as a Measure of Systemic Risk", Journal of Portfolio Management 37(4), 2011
- Kondor, Pafka and Nagy, "Noise sensitivity of portfolio selection under various risk measures", Journal of Banking and Finance 31(5), 2007
- Ciliberti, Kondor and Mézard, "On the feasibility of portfolio optimization under expected shortfall", Quantitative Finance 7(4), 2007

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Simulated matrices, plus free daily prices fetched by script.

## How it is tested

The maths is tested before any result is trusted.

- Eigenvalues of pure-noise matrices fall inside the Marchenko-Pastur bounds
- A planted factor is recovered after cleaning
- The simulated divergence sits where the theory puts it

## Layout

```
src/static/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] The Marchenko-Pastur density against the eigenvalues of simulated and real correlation matrices
- [ ] Eigenvalue clipping, and what it does to out-of-sample portfolio risk
- [ ] The absorption ratio through 2008, 2020 and 2022
- [ ] Out-of-sample risk as the ratio of assets to observations rises, diverging at the critical point
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- The theory assumes independent, identically distributed returns. Real returns have fat tails and changing volatility.
- Cleaning helps risk estimates. It does not tell you expected returns.
- This is research code. It is not investment advice and it does not trade.

## The name

Static is what you hear between stations. An estimated correlation matrix is mostly that.

## Licence

MIT. See `LICENSE`.
