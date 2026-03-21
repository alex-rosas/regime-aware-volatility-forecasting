# Volatility Regimes in Emerging Markets
### When Black-Scholes Assumptions Break Down — MXN/USD

> *The most widely used model in quantitative finance contains assumptions
> that are empirically false for emerging market assets.  This project does
> not say so to be provocative — the data says so, with numbers precise
> enough to be cited.*

![Hero](assets/figures/readme/hero.png)

---

## The Argument

The Black-Scholes-Merton framework models a risky asset price $S_t$ as a
geometric Brownian motion:

$$dS_t = \mu\, S_t\, dt + \sigma\, S_t\, dW_t$$

This implies four testable assumptions: constant volatility, normally
distributed log-returns, no autocorrelation in returns, and continuous
sample paths.  Every one of them fails on the MXN/USD daily series
(January 2000 – March 2026, $n = 6{,}588$ observations), and the failures
are not borderline:

| Test | Statistic | p-value | Null hypothesis |
|------|-----------|---------|-----------------|
| **ARCH-LM** (Engle 1982) | $LM = 1{,}463.3$ | $2.1 \times 10^{-308}$ | No volatility clustering |
| **Jarque-Bera** | $JB = 29{,}801.1$, excess kurtosis $= 10.30$ | $\approx 0$ | Normally distributed returns |
| **Ljung-Box on** $r_t^2$ | $Q(10) = 3{,}663.98$ | $\approx 0$ | No autocorrelation in variance |
| **Ljung-Box on** $r_t$ | $Q(10) = 33.61$ | $0.0002$ | — |

The last two lines together are the GARCH signature: squared returns are
strongly autocorrelated (variance clusters), raw returns are nearly
unpredictable.  These three rejections are the formal justification for
every modeling choice that follows.

---

## Model Architecture

![Pipeline](assets/figures/readme/pipeline_diagram.png)

The pipeline is a deliberate sequence of four modeling layers, each chosen
to address a specific empirically documented failure of the standard model.

### Layer 1 — GARCH(1,1)-*t* and EGARCH(1,1)-*t*

**Why Student-*t* innovations, not Gaussian?**
The Student-*t* excess kurtosis is $6/(\nu - 4)$ for $\nu > 4$.
Inverting at the empirical value $K = 10.30$ gives a moment-matching
estimate $\hat{\nu}_{\text{moments}} \approx 4.6$ before fitting any model.
Using Gaussian innovations is not a simplification here — it is a
misspecification that systematically underestimates tail probabilities.

**Why EGARCH in addition to GARCH-*t*?**
Standard GARCH treats positive and negative shocks symmetrically.  EGARCH
models $\log(\sigma_t^2)$, encoding asymmetry directly via the $\gamma$
parameter:

$$\log(\sigma_t^2) = \omega + \beta\log(\sigma_{t-1}^2)
+ \alpha\bigl(|z_{t-1}| - \mathbb{E}[|z_{t-1}|]\bigr)
+ \gamma\, z_{t-1}$$

For MXN/USD the fitted value is $\hat{\gamma} > 0$ — *positive*
appreciations generate marginally larger volatility than depreciations.
This is the *opposite* of the standard equity leverage effect and is
consistent with the return skewness of $\hat{S} = +0.78$.  Whether
$\gamma$ is positive or negative is an empirical question that deserves
an answer rather than an assumption.

**Why not GJR-GARCH?** GJR-GARCH captures leverage via a dummy variable
on negative shocks: $\sigma_t^2 = \omega + (\alpha + \gamma
\mathbf{1}_{[\epsilon_{t-1}<0]})\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$.
It is more interpretable but requires enforcing $\alpha + \gamma \geq 0$
as an explicit constraint.  EGARCH achieves non-negativity of the
conditional variance by construction through the log formulation.

### Layer 2 — 3-state Hidden Markov Model

**Why a discrete latent state model?**
GARCH captures volatility *persistence* but treats volatility as a single
continuous process evolving around a fixed long-run level.  The MXN/USD
series has lived through structurally distinct episodes — the 2008 Lehman
collapse (annualised vol peak: $45.4\%$), the 2016 Trump election shock
($28.9\%$), the 2020 COVID crash ($41.4\%$) — against a full-sample median
of $9.1\%$.  The right model for that structure represents discrete latent
states.

The HMM is defined by a hidden chain $z_t \in \{0, 1, 2\}$, transition
matrix $A_{ij} = P(z_{t+1} = j \mid z_t = i)$, and Gaussian emissions:

$$P(\mathbf{x}_t \mid z_t = k) = \mathcal{N}(\boldsymbol{\mu}_k,
\boldsymbol{\Sigma}_k), \qquad
\mathbf{x}_t = \begin{bmatrix} r_t \\ r_t^2 \end{bmatrix}$$

Parameters are estimated by Baum-Welch (EM); the MAP state sequence is
recovered by the Viterbi algorithm:
$\hat{\mathbf{Z}} = \arg\max_{\mathbf{Z}} P(\mathbf{Z} \mid \mathbf{r};\,\theta)$.

**Why $K = 3$, not 2 or 4?**  Two states conflate elevated-but-normal
volatility (2022 rate-hike period) with genuine crisis volatility.  Four
states over-segment the data at this sample size.  BIC formally selects
$K = 3$.

**Why not Markov-Switching GARCH?**
RS-GARCH — where GARCH parameters switch with the hidden state — is
statistically more efficient because the transition probabilities and
the GARCH persistence parameters are estimated from the same likelihood.
The trade-off is separability: a joint model cannot be validated
component by component.  The feature-augmented architecture retains GARCH
and HMM as independent, inspectable modules whose outputs can each be
evaluated before combination.  Statistical efficiency is sacrificed for
interpretability and reproducibility — a deliberate choice, not an oversight.

### Layer 3 — Hybrid XGBoost Forecaster

Three hybrid architectures appear in the literature.  All three were evaluated:

| Criterion | Residual Learning | **Feature-Augmented** | LSTM |
|-----------|:-----------------:|:---------------------:|:----:|
| Interpretability (SHAP) | Partial | **Full** | None |
| Sample size requirement | Low | **Low** | High |
| Preserves econometric structure | Yes | **Yes** | No |
| Answers "what does ML add?" | Partial | **Directly** | No |
| HMM regime as named input | No | **Yes** | No |

**Why Feature-Augmented?**
The point forecast is:
$$\hat{\sigma}_{t+1} = f_{\text{XGB}}\!\left(
\hat{\sigma}_t^{\text{GARCH}},\;
\hat{\sigma}_t^{\text{EGARCH}},\;
z_t^{\text{HMM}},\;
r_{t-1}, r_{t-2}, r_{t-3},\;
\text{VIX}_t,\;
\text{T10Y2Y}_t
\right)$$

The ML model learns nonlinear interactions between econometric outputs and
macro signals — the way a VIX spike during a regime transition produces a
different volatility response than the same spike during calm conditions.

**Why not LSTM?**
At $n = 6{,}588$ daily observations, an LSTM with sufficient capacity
requires regularization so aggressive that it likely learns nothing beyond
what GARCH already captures.  The empirical literature shows LSTM hybrids
outperforming GARCH-type models at intraday frequency ($n > 10^6$) or in
very long daily samples.  At this scale the evidence is mixed at best.

**Why not Residual Learning?**
After a well-specified GARCH-*t* model the residual structure is expected
to be small.  More critically: under Architecture 1 the HMM regime $z_t$
would not appear as a feature.  The Feature-Augmented architecture is the
only one where the HMM enters the model as an explicit, named input whose
contribution can be measured by SHAP.

**Empirical finding:**  SHAP analysis reveals that `regime` is the single
most important feature — mean absolute SHAP value approximately five times
larger than `sigma_egarch_ann` or `sigma_garch_ann`.  The discrete regime
label contributes more to the forecast than the continuous volatility
estimates — a finding that validates the architecture choice.

### Layer 4 — Asymmetric Split Conformal Prediction

A point forecast is insufficient for risk management.  The GARCH-*t* model
produces its own confidence intervals, but they are asymptotic (valid only
as $n \to \infty$) and assume correct model specification — both questionable
at excess kurtosis $10.30$.  Split conformal prediction makes no
distributional assumption and its coverage guarantee holds in finite samples.

**The asymmetric innovation.**
Standard split conformal uses the absolute residuals $s_i = |y_i -
\hat{y}_i|$, producing a symmetric interval of the form
$[\hat{y} - \hat{q},\; \hat{y} + \hat{q}]$.  For volatility risk
management the upper bound matters more than the lower.  We implement an
`AsymmetricConformalPredictor` that uses *signed* residuals and splits the
error budget asymmetrically with parameter $\phi$:

$$\alpha_{\text{upper}} = \phi\,\alpha, \qquad
\alpha_{\text{lower}} = (1-\phi)\,\alpha$$

With $\phi = 0.7$ the upper bound absorbs $70\%$ of the error budget —
wider upper bounds, narrower lower bounds, same total marginal coverage.

The key consequence for backtesting: the correct null hypothesis for the
**Kupiec POF test** on the upper bound is not $\alpha$ but
$\alpha_{\text{upper}} = \phi\,\alpha$.  Using the symmetric null when
the predictor is asymmetric artificially inflates the test statistic.

---

## Results

### Out-of-Sample Forecast Accuracy (test set: 2022 – 2026)

| Model | RMSE | QLIKE | DM statistic vs Hybrid | DM p-value |
|-------|------|-------|------------------------|------------|
| **Hybrid XGBoost** | **0.004462** | 1.6480 | — | — |
| GARCH(1,1)-*t* | 0.004974 | 1.6380 | −6.67 | ≈ 0 |
| EGARCH(1,1)-*t* | 0.004859 | 1.6392 | −5.36 | ≈ 0 |

RMSE improvement over GARCH: **10.28%**.  The Diebold-Mariano test rejects
equal predictive accuracy at $p \approx 0$ against both baselines.

The RMSE–QLIKE disagreement is itself informative: the hybrid reduces
average squared error but is marginally less calibrated for tail risk
(QLIKE penalises underestimation more than overestimation).  This gap —
a model trained on RMSE will not automatically minimise QLIKE — is the
direct motivation for the conformal prediction layer.

### Walk-Forward Validation (5-fold expanding window)

| Model | WF RMSE | WF QLIKE |
|-------|---------|----------|
| **Hybrid XGBoost** | **0.004291** | **1.6420** |
| GARCH(1,1)-*t* | 0.004836 | 1.6486 |
| EGARCH(1,1)-*t* | 0.004726 | 1.6465 |

WF RMSE improvement: **11.28%**.  On walk-forward QLIKE the hybrid
*outperforms* GARCH — the picture differs from the single test-split
result and demonstrates that the hybrid's edge is not artefact of one
particular test window.

### Asymmetric Conformal Coverage

| Interval | Nominal | Empirical | Kupiec null ($\alpha_{\text{upper}}$) | H₀ rejected? |
|----------|---------|-----------|--------------------------------------|--------------|
| 80% | $\alpha = 0.20$ | — | $0.14$ | No |
| **90%** | $\alpha = 0.10$ | — | $0.07$ | No |
| 95% | $\alpha = 0.05$ | — | $0.035$ | No |

Applying the correct asymmetric null, the Kupiec test does not reject at
any level — the upper bounds are well-calibrated in aggregate.

### Regime-Conditioned Coverage (90% interval)

| Regime | Days | Emp. upper viol. rate | Gap vs 7% target |
|--------|------|-----------------------|------------------|
| Low vol | 396 | ≪ 7% | negative (conservative) |
| Medium vol | 560 | ≈ 7% | ≈ 0 (well-calibrated) |
| High vol (crisis) | 72 | > 7% | **positive (undercoverage)** |

The constant-width interval is too conservative in calm regimes and
insufficiently wide during crises — the empirical motivation for locally
adaptive conformal prediction as a natural extension.

---

## Engineering Stack

| Component | Tool | Why this tool |
|-----------|------|---------------|
| GARCH / EGARCH | `arch` | Only mature Python library supporting GARCH-*t*, EGARCH-*t*, GJR-GARCH; Student-*t* MLE built-in |
| HMM | `hmmlearn` | scikit-learn compatible; Baum-Welch and Viterbi built-in |
| Hybrid forecaster | `xgboost` | Strongest tabular algorithm at this sample size; native early stopping; exact SHAP support |
| Conformal prediction | `src/models/conformal.py` | Custom `AsymmetricConformalPredictor`; MAPIE removed (no asymmetric support) |
| Experiment tracking | DVC + DagsHub | Full artifact versioning; `metrics.json` git-tracked; `dvc repro` reproduces pipeline end-to-end |
| Visualization | `src/dark_viz.py` | Shared dark-mode palette (`#0E1117`); all figures pre-built as PNGs for zero-latency Streamlit |
| Data validation | `great_expectations` | Automated schema and range checks on raw data |

---

## Quickstart

```bash
# clone and set up
git clone https://github.com/<user>/volatility_regimes
cd volatility_regimes
conda env create -f environment.yml
conda activate volatility_regimes

# reproduce full pipeline (uses DVC)
dvc repro

# launch Streamlit app
streamlit run app/streamlit_app.py
```

DVC artifacts (processed data, models) are tracked on DagsHub.
To pull without re-running:
```bash
dvc pull
```

---

## Repository Structure

```
volatility_regimes/
├── data/
│   ├── raw/                   # Banxico SIE + FRED (DVC-tracked)
│   └── processed/             # GARCH, HMM, hybrid outputs (DVC-tracked)
├── notebooks/
│   ├── 01_data_acquisition    # Banxico / Yahoo / FRED API
│   ├── 02_validation          # Great Expectations data quality suite
│   ├── 03_eda                 # Return distributions, stylized facts
│   ├── 04_assumption_tests    # ARCH-LM, Jarque-Bera, Ljung-Box
│   ├── 05_garch_model         # GARCH-t, EGARCH-t, AIC/BIC comparison
│   ├── 06_hmm_regimes         # 3-state HMM, Baum-Welch, Viterbi
│   ├── 07_hybrid_model        # XGBoost + SHAP + Diebold-Mariano
│   ├── 08_prediction_intervals # Asymmetric conformal prediction (φ=0.7)
│   └── 09_evaluation          # Kupiec POF, VaR backtesting, regime coverage
├── src/
│   ├── models/
│   │   ├── garch.py           # VolatilityModel — unified GARCH/EGARCH wrapper
│   │   ├── hmm.py             # RegimeHMM — 3-state Gaussian HMM
│   │   ├── hybrid.py          # HybridVolatilityModel — XGBoost + SHAP
│   │   └── conformal.py       # ConformalPredictor + AsymmetricConformalPredictor
│   ├── dark_viz.py            # Shared dark-mode matplotlib style + figure factories
│   ├── pipeline.py            # All step_* functions + run_pipeline_end_to_end()
│   └── utils.py               # atomic_write context manager
├── stages/                    # DVC stage entry points (01–09)
├── app/
│   ├── streamlit_app.py       # st.navigation() entry point
│   └── pages/                 # 4-page Streamlit app
├── assets/figures/            # Pre-built dark-mode PNGs (git-tracked)
├── dvc.yaml                   # 9-stage DVC pipeline
├── params.yaml                # Single source of truth for all hyperparameters
└── metrics.json               # Git-tracked experiment metrics
```

---

## Data Sources

All data are freely accessible without commercial subscription.

- **MXN/USD** — [Banxico SIE API](https://www.banxico.org.mx/SieAPIRest) · series `SF43718` (FIX rate, daily from 1991)
- **VIX, T10Y2Y** — [FRED API](https://fred.stlouisfed.org/) · free API key required
- **Sample** — January 2000 – March 2026 · $n = 6{,}588$ daily observations
- **Crisis episodes in sample** — GFC 2008 (peak ann. vol: $45.4\%$), Trump shock 2016 ($28.9\%$), COVID 2020 ($41.4\%$)

---

*Luis Alejandro Rosas Martínez · PhD Applied Mathematics · ENSTA Paris · 2025*
