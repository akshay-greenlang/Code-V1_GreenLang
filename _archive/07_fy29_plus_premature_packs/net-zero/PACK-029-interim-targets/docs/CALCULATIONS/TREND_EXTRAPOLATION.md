# Calculation Guide: Trend Extrapolation & Forecasting

**Pack:** PACK-029 Interim Targets Pack
**Version:** 1.0.0
**Engine:** Trend Extrapolation Engine

---

## Table of Contents

1. [Overview](#overview)
2. [Model Selection Framework](#model-selection-framework)
3. [Linear Regression](#linear-regression)
4. [Exponential Smoothing](#exponential-smoothing)
5. [ARIMA Modeling](#arima-modeling)
6. [Confidence Intervals](#confidence-intervals)
7. [Forecast Accuracy Metrics](#forecast-accuracy-metrics)
8. [Automatic Model Selection](#automatic-model-selection)
9. [Seasonal Adjustment](#seasonal-adjustment)
10. [Forecast Horizon Guidelines](#forecast-horizon-guidelines)
11. [Worked Examples](#worked-examples)

---

## Overview

The Trend Extrapolation Engine projects future emissions based on historical data, enabling organizations to assess whether they are on track to meet their interim targets. It supports three forecasting models, each suited to different data characteristics.

### Purpose

1. **Forward projection**: Estimate future emissions under current trajectory
2. **Target gap analysis**: Quantify the gap between projected and target emissions
3. **Early warning**: Detect divergence from the target pathway before it becomes critical
4. **Scenario planning**: Evaluate best-case and worst-case emission trajectories

### Key Principles

- All arithmetic uses `Decimal` precision (no float in calculation paths)
- Confidence intervals are always provided (80% and 95%)
- Multiple models are evaluated and the best-performing one is selected
- Forecast accuracy is quantified with standard metrics (MAE, RMSE, MAPE)
- Results include SHA-256 provenance hashing

---

## Model Selection Framework

### Available Models

| Model | Best For | Data Requirements | Strengths | Limitations |
|-------|----------|-------------------|-----------|-------------|
| Linear Regression | Steady, monotonic trends | 4+ data points | Simple, interpretable, robust | Cannot capture curvature or acceleration |
| Exponential Smoothing | Data with level shifts | 4+ data points | Adapts to recent changes | No confidence intervals from theory (bootstrapped) |
| ARIMA | Complex patterns, autocorrelation | 8+ data points | Captures complex dynamics | Requires stationarity, harder to interpret |

### Decision Tree

```
Start
  |
  +--> Data points >= 8?
  |      |
  |      Yes --> Run all 3 models --> Select best by AIC/MAPE
  |      |
  |      No --> Data points >= 4?
  |               |
  |               Yes --> Run Linear + Exponential Smoothing --> Select best by MAPE
  |               |
  |               No --> Use Linear Regression only (minimum viable forecast)
```

### Model Output Structure

```python
class ForecastResult(BaseModel):
    model_type: str                    # "linear", "exponential_smoothing", "arima"
    forecast_points: list[ForecastPoint]  # Projected values
    confidence_intervals: ConfidenceIntervals
    accuracy_metrics: AccuracyMetrics
    parameters: dict                   # Model-specific parameters
    selected_as_best: bool             # True if this model was selected
    selection_reason: str              # Why this model was chosen
```

---

## Linear Regression

### Mathematical Foundation

Linear regression fits a straight line through the historical emissions data:

```
E(t) = alpha + beta * t + epsilon
```

Where:
- **E(t)** = Emissions at time t (tCO2e)
- **alpha** = Y-intercept (emissions at t=0)
- **beta** = Slope (annual change in emissions, tCO2e/year)
- **t** = Time index (years from baseline)
- **epsilon** = Error term

### Parameter Estimation (Ordinary Least Squares)

```
beta = [ n * SUM(t_i * E_i) - SUM(t_i) * SUM(E_i) ] / [ n * SUM(t_i^2) - (SUM(t_i))^2 ]

alpha = E_bar - beta * t_bar
```

Where:
- **n** = Number of data points
- **E_bar** = Mean of emissions values
- **t_bar** = Mean of time indices

### Goodness of Fit

```
R^2 = 1 - SS_res / SS_tot

Where:
    SS_res = SUM( (E_i - E_hat_i)^2 )     [residual sum of squares]
    SS_tot = SUM( (E_i - E_bar)^2 )        [total sum of squares]
```

### Forecast Formula

```
E_hat(t_future) = alpha + beta * t_future
```

### Standard Error of Forecast

```
SE_forecast = s * sqrt( 1 + 1/n + (t_future - t_bar)^2 / SUM((t_i - t_bar)^2) )

Where:
    s = sqrt( SS_res / (n - 2) )    [residual standard error]
```

### Implementation

```python
def _fit_linear(self, times: list[Decimal], emissions: list[Decimal]) -> LinearModel:
    """Fit ordinary least squares linear regression."""
    n = Decimal(str(len(times)))
    sum_t = sum(times)
    sum_e = sum(emissions)
    sum_te = sum(t * e for t, e in zip(times, emissions))
    sum_t2 = sum(t * t for t in times)

    denominator = n * sum_t2 - sum_t * sum_t
    if denominator == Decimal("0"):
        raise ValueError("All time values are identical; cannot fit linear model.")

    beta = (n * sum_te - sum_t * sum_e) / denominator
    alpha = (sum_e - beta * sum_t) / n

    # Residual standard error
    predictions = [alpha + beta * t for t in times]
    ss_res = sum((e - p) ** 2 for e, p in zip(emissions, predictions))
    ss_tot = sum((e - sum_e / n) ** 2 for e in emissions)

    r_squared = Decimal("1") - ss_res / ss_tot if ss_tot > 0 else Decimal("0")
    s = (ss_res / (n - Decimal("2"))).sqrt() if n > Decimal("2") else Decimal("0")

    return LinearModel(alpha=alpha, beta=beta, r_squared=r_squared, std_error=s)
```

### Interpretation Guide

| beta Value | Interpretation |
|------------|---------------|
| beta < 0 | Emissions are decreasing (on a reduction trajectory) |
| beta = 0 | Emissions are flat (no progress) |
| beta > 0 | Emissions are increasing (moving away from targets) |

| R^2 Value | Interpretation |
|-----------|---------------|
| R^2 > 0.90 | Strong linear trend; linear model is appropriate |
| 0.70 < R^2 < 0.90 | Moderate fit; consider other models |
| R^2 < 0.70 | Weak linear fit; exponential smoothing or ARIMA likely better |

---

## Exponential Smoothing

### Simple Exponential Smoothing (SES)

SES is used when data has no clear trend or seasonality but emissions levels shift over time:

```
S_t = alpha * E_t + (1 - alpha) * S_(t-1)
```

Where:
- **S_t** = Smoothed value at time t
- **E_t** = Actual emissions at time t
- **alpha** = Smoothing parameter (0 < alpha < 1)
- **S_0** = Initial level (set to first observation)

### Holt's Linear Trend (Double Exponential Smoothing)

PACK-029 uses Holt's method, which captures both level and trend:

```
Level:  L_t = alpha * E_t + (1 - alpha) * (L_(t-1) + T_(t-1))
Trend:  T_t = beta_h * (L_t - L_(t-1)) + (1 - beta_h) * T_(t-1)
```

Where:
- **L_t** = Level component at time t
- **T_t** = Trend component at time t
- **alpha** = Level smoothing parameter (0 < alpha < 1)
- **beta_h** = Trend smoothing parameter (0 < beta_h < 1)

### Forecast Formula

```
E_hat(t + h) = L_t + h * T_t
```

Where **h** is the forecast horizon (number of periods ahead).

### Damped Trend Variant

For longer-horizon forecasts, PACK-029 uses a damped trend to prevent unrealistic extrapolation:

```
Level:  L_t = alpha * E_t + (1 - alpha) * (L_(t-1) + phi * T_(t-1))
Trend:  T_t = beta_h * (L_t - L_(t-1)) + (1 - beta_h) * phi * T_(t-1)

Forecast: E_hat(t + h) = L_t + (phi + phi^2 + ... + phi^h) * T_t
                        = L_t + phi * (1 - phi^h) / (1 - phi) * T_t
```

Where **phi** is the damping parameter (0.8 <= phi <= 1.0). When phi = 1.0, this reduces to standard Holt's method.

### Parameter Optimization

Parameters (alpha, beta_h, phi) are optimized by minimizing the sum of squared errors (SSE):

```
SSE = SUM( (E_t - E_hat_t)^2 )    for all in-sample periods
```

PACK-029 uses grid search with step size 0.05 over the valid parameter ranges:

```python
best_params = None
best_sse = Decimal("Infinity")

for alpha in [Decimal(str(a/100)) for a in range(5, 100, 5)]:
    for beta_h in [Decimal(str(b/100)) for b in range(5, 100, 5)]:
        for phi in [Decimal(str(p/100)) for p in range(80, 101, 5)]:
            sse = _compute_sse(emissions, alpha, beta_h, phi)
            if sse < best_sse:
                best_sse = sse
                best_params = (alpha, beta_h, phi)
```

### Confidence Intervals (Bootstrapped)

Since Holt's method does not produce analytical confidence intervals, PACK-029 uses residual bootstrapping:

1. Compute in-sample residuals: `r_t = E_t - E_hat_t`
2. For each bootstrap iteration (B = 1000):
   a. Sample residuals with replacement
   b. Generate synthetic future path by adding sampled residuals to the forecast
3. Compute percentiles of the bootstrap distribution:
   - 80% CI: 10th and 90th percentiles
   - 95% CI: 2.5th and 97.5th percentiles

---

## ARIMA Modeling

### Model Specification

ARIMA(p, d, q) where:
- **p** = Order of the autoregressive (AR) component
- **d** = Degree of differencing (to achieve stationarity)
- **q** = Order of the moving average (MA) component

### Components

#### Differencing (I component)

If the emissions series is non-stationary (has a trend), differencing is applied:

```
First difference:  Delta_E_t = E_t - E_(t-1)
Second difference: Delta^2_E_t = Delta_E_t - Delta_E_(t-1)
```

Stationarity is tested using the Augmented Dickey-Fuller (ADF) test:

```
H_0: Series has a unit root (non-stationary)
H_1: Series is stationary

If p-value < 0.05: reject H_0, series is stationary (d = 0)
If p-value >= 0.05: difference and re-test (d = 1 or d = 2)
```

#### Autoregressive (AR) Component

```
E'_t = c + phi_1 * E'_(t-1) + phi_2 * E'_(t-2) + ... + phi_p * E'_(t-p) + epsilon_t
```

Where E'_t is the differenced series and phi_1...phi_p are AR coefficients.

#### Moving Average (MA) Component

```
E'_t = c + epsilon_t + theta_1 * epsilon_(t-1) + theta_2 * epsilon_(t-2) + ... + theta_q * epsilon_(t-q)
```

Where theta_1...theta_q are MA coefficients and epsilon_t is white noise.

#### Full ARIMA Model

```
E'_t = c + SUM(phi_i * E'_(t-i), i=1..p) + SUM(theta_j * epsilon_(t-j), j=1..q) + epsilon_t
```

### Order Selection

PACK-029 selects ARIMA order using the Akaike Information Criterion (AIC):

```
AIC = 2k - 2 * ln(L)

Where:
    k = p + q + 1 (number of parameters including constant)
    L = Maximum likelihood of the fitted model
```

The engine searches over:
- p in {0, 1, 2, 3}
- d in {0, 1, 2}
- q in {0, 1, 2, 3}

And selects the (p, d, q) combination with the lowest AIC.

### Parameter Bounds

| Parameter | Range | Default Search |
|-----------|-------|---------------|
| p (AR order) | 0-3 | 0, 1, 2, 3 |
| d (differencing) | 0-2 | Determined by ADF test |
| q (MA order) | 0-3 | 0, 1, 2, 3 |

### Forecast and Confidence Intervals

ARIMA provides analytical confidence intervals:

```
E_hat(t + h) = point forecast at horizon h

CI_alpha(t + h) = E_hat(t + h) +/- z_(alpha/2) * sigma_h

Where:
    sigma_h = sigma * sqrt(1 + SUM(psi_j^2, j=1..h-1))
    psi_j = MA coefficients of the infinite MA representation
    z_0.10 = 1.282   (for 80% CI)
    z_0.025 = 1.960  (for 95% CI)
```

### Implementation Notes

```python
def _fit_arima(self, emissions: list[Decimal], max_p: int = 3, max_q: int = 3) -> ARIMAModel:
    """Fit ARIMA model with automatic order selection."""
    # Convert to float for statsmodels (Decimal not supported)
    y = [float(e) for e in emissions]

    # Determine differencing order
    d = self._determine_differencing(y)

    # Grid search for best (p, q)
    best_aic = float("inf")
    best_order = (0, d, 0)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(y, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except Exception:
                continue  # Skip invalid combinations

    # Fit best model
    best_model = ARIMA(y, order=best_order).fit()

    # Convert back to Decimal
    forecasts = [Decimal(str(f)) for f in best_model.forecast(steps=forecast_horizon)]
    conf_int = best_model.get_forecast(steps=forecast_horizon).conf_int()

    return ARIMAModel(
        order=best_order,
        aic=Decimal(str(best_aic)),
        forecasts=forecasts,
        confidence_intervals=self._convert_conf_int(conf_int),
    )
```

---

## Confidence Intervals

### Overview

PACK-029 provides two confidence interval levels for all forecasts:

| Level | Z-Score | Interpretation |
|-------|---------|---------------|
| 80% CI | 1.282 | Emissions will fall within this range 4 out of 5 times |
| 95% CI | 1.960 | Emissions will fall within this range 19 out of 20 times |

### Linear Regression CI

```
CI_alpha(t + h) = E_hat(t + h) +/- t_(alpha/2, n-2) * SE_forecast(t + h)

Where:
    SE_forecast(t+h) = s * sqrt(1 + 1/n + (t_h - t_bar)^2 / SUM((t_i - t_bar)^2))
    t_(alpha/2, n-2) = Student's t critical value with n-2 degrees of freedom
```

### Exponential Smoothing CI (Bootstrapped)

```
For B = 1000 bootstrap iterations:
    1. Sample residuals with replacement: r*_1, r*_2, ..., r*_h
    2. Compute bootstrap forecast: E*_hat(t + j) = E_hat(t + j) + r*_j
    3. Store all E*_hat values

CI_80 = [percentile_10(E*), percentile_90(E*)]
CI_95 = [percentile_2.5(E*), percentile_97.5(E*)]
```

### ARIMA CI (Analytical)

```
CI_alpha(t + h) = E_hat(t + h) +/- z_(alpha/2) * sigma_h

Where sigma_h is derived from the MA(infinity) representation of the ARIMA model.
```

### CI Width Growth

Confidence intervals widen with forecast horizon. PACK-029 flags forecasts where the 95% CI width exceeds 50% of the point forecast:

```python
ci_width_pct = (upper_95 - lower_95) / point_forecast * Decimal("100")
if ci_width_pct > Decimal("50"):
    result.warnings.append(
        f"Wide confidence interval ({ci_width_pct:.1f}%) at horizon {h}. "
        f"Forecast reliability decreases beyond this point."
    )
```

### Data Structure

```json
{
  "confidence_intervals": {
    "ci_80": [
      {"year": 2025, "lower": 175000, "upper": 195000},
      {"year": 2026, "lower": 168000, "upper": 198000},
      {"year": 2027, "lower": 160000, "upper": 202000}
    ],
    "ci_95": [
      {"year": 2025, "lower": 170000, "upper": 200000},
      {"year": 2026, "lower": 160000, "upper": 206000},
      {"year": 2027, "lower": 148000, "upper": 214000}
    ]
  }
}
```

---

## Forecast Accuracy Metrics

### Mean Absolute Error (MAE)

```
MAE = (1/n) * SUM( |E_i - E_hat_i| )
```

**Interpretation:** Average absolute deviation in tCO2e. Easy to understand but does not penalize large errors disproportionately.

### Root Mean Squared Error (RMSE)

```
RMSE = sqrt( (1/n) * SUM( (E_i - E_hat_i)^2 ) )
```

**Interpretation:** Standard deviation of forecast errors in tCO2e. Penalizes large errors more than MAE.

### Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) * SUM( |E_i - E_hat_i| / E_i )    [when E_i > 0]
```

**Interpretation:** Average percentage error. Scale-independent, enabling comparison across entities of different sizes.

### Akaike Information Criterion (AIC)

```
AIC = 2k - 2 * ln(L)
```

**Interpretation:** Balances model fit against complexity. Lower AIC is better. Used for ARIMA order selection.

### Metric Selection for Model Comparison

| Metric | Used For | Lower Is Better |
|--------|----------|-----------------|
| MAE | Interpretability (same units as emissions) | Yes |
| RMSE | Sensitivity to outliers | Yes |
| MAPE | Cross-entity comparison (scale-free) | Yes |
| AIC | ARIMA order selection | Yes |
| R^2 | Linear regression goodness of fit | No (higher is better) |

### PACK-029 Default: MAPE for Model Selection

PACK-029 uses MAPE as the primary metric for selecting the best model, because it allows fair comparison between linear regression, exponential smoothing, and ARIMA regardless of emission scale.

```python
def _select_best_model(self, models: list[ForecastModel]) -> ForecastModel:
    """Select model with lowest MAPE."""
    valid_models = [m for m in models if m.accuracy_metrics.mape is not None]
    if not valid_models:
        return models[0]  # Fallback to first model
    return min(valid_models, key=lambda m: m.accuracy_metrics.mape)
```

### Cross-Validation

PACK-029 uses expanding window cross-validation for robust accuracy estimation:

```
Given n data points:

Fold 1: Train on [1..4], Test on [5]
Fold 2: Train on [1..5], Test on [6]
Fold 3: Train on [1..6], Test on [7]
...
Fold k: Train on [1..n-1], Test on [n]

CV_MAPE = average MAPE across all folds
```

Minimum training window size: 4 data points.

---

## Automatic Model Selection

### Selection Algorithm

```
Input: Historical emissions time series [E_1, E_2, ..., E_n]

Step 1: Data Validation
    - Require n >= 4 data points
    - Check for zero or negative values
    - Detect and flag outliers (>3 standard deviations from mean)

Step 2: Fit All Eligible Models
    - If n >= 4: Fit Linear Regression
    - If n >= 4: Fit Exponential Smoothing (Holt's damped)
    - If n >= 8: Fit ARIMA (with automatic order selection)

Step 3: Compute Accuracy Metrics
    - Use expanding window cross-validation
    - Compute MAE, RMSE, MAPE for each model

Step 4: Select Best Model
    - Primary criterion: Lowest cross-validated MAPE
    - Tie-breaker: Lowest AIC (if available) or simpler model

Step 5: Generate Forecasts
    - Produce point forecasts for requested horizon
    - Compute 80% and 95% confidence intervals
    - Flag if CI width exceeds 50% of point forecast

Step 6: Validate Output
    - Ensure forecast values are non-negative (emissions cannot be negative)
    - Floor negative forecasts to zero with a warning
    - Compute provenance hash

Output: ForecastResult with selected model, all metrics, and forecasts
```

### Selection Rationale Narratives

PACK-029 generates human-readable selection rationales:

```python
selection_narratives = {
    "linear_best": (
        f"Linear regression selected (MAPE: {mape:.1f}%, R²: {r_sq:.3f}). "
        f"Emissions show a consistent {'downward' if slope < 0 else 'upward'} trend "
        f"of {abs(slope):,.0f} tCO2e/year."
    ),
    "holt_best": (
        f"Exponential smoothing selected (MAPE: {mape:.1f}%). "
        f"Recent data suggests a level shift that linear regression cannot capture. "
        f"Damping factor: {phi:.2f}."
    ),
    "arima_best": (
        f"ARIMA({p},{d},{q}) selected (MAPE: {mape:.1f}%, AIC: {aic:.1f}). "
        f"Emissions show autocorrelated patterns best captured by ARIMA."
    ),
}
```

---

## Seasonal Adjustment

### When to Apply

Quarterly monitoring data may exhibit seasonal patterns (e.g., higher heating emissions in winter). PACK-029 applies seasonal adjustment when:

1. At least 8 quarterly data points are available (2 full years)
2. Seasonal variation exceeds 10% of the mean

### Method: Classical Decomposition

```
E_t = Trend_t + Seasonal_t + Residual_t    (additive model)

Step 1: Compute 4-quarter moving average (Trend_t)
Step 2: Compute seasonal component: Seasonal_t = E_t - Trend_t, averaged by quarter
Step 3: Seasonally adjusted: E_adj_t = E_t - Seasonal_t
```

### Seasonal Indices

```
Quarter    | Typical Index (heating-dominated) | Typical Index (cooling-dominated)
-----------|------------------------------------|-----------------------------------
Q1 (Jan-Mar) | +15% to +25%                    | -5% to +5%
Q2 (Apr-Jun) | -10% to -5%                     | +5% to +15%
Q3 (Jul-Sep) | -15% to -10%                    | +15% to +25%
Q4 (Oct-Dec) | +5% to +15%                     | -5% to +5%
```

### Implementation Note

Seasonal adjustment is applied before model fitting and removed from the forecasts:

```python
# Before fitting
adjusted_emissions = self._seasonally_adjust(quarterly_emissions)

# Fit model on adjusted data
model = self._fit_best_model(adjusted_emissions)

# Forecast on adjusted scale
adjusted_forecast = model.forecast(horizon)

# Add seasonal pattern back
final_forecast = self._add_seasonality(adjusted_forecast, seasonal_indices)
```

---

## Forecast Horizon Guidelines

### Recommended Horizons

| Data History | Maximum Reliable Horizon | Reasoning |
|-------------|--------------------------|-----------|
| 3-4 years | 2 years | Short history limits extrapolation reliability |
| 5-7 years | 3-4 years | Moderate history; watch CI width |
| 8-10 years | 5-6 years | Good history; ARIMA viable |
| 10+ years | 7-10 years | Long history; all models viable |

### Rule of Thumb

```
Recommended horizon <= Data history / 2
```

PACK-029 warns when the requested forecast horizon exceeds this guideline.

### Horizon-Dependent Quality Indicators

| CI Width / Point Forecast | Quality Rating | Action |
|--------------------------|----------------|--------|
| < 15% | HIGH | Forecast is reliable; use with confidence |
| 15% - 30% | MEDIUM | Forecast is reasonable; monitor for deviations |
| 30% - 50% | LOW | Forecast has significant uncertainty; supplement with other analysis |
| > 50% | VERY LOW | Forecast is unreliable; reduce horizon or gather more data |

---

## Worked Examples

### Example 1: Manufacturing Company (Linear Trend)

**Input Data:**

| Year | Emissions (tCO2e) |
|------|-------------------|
| 2019 | 220,000 |
| 2020 | 200,000 |
| 2021 | 210,000 |
| 2022 | 195,000 |
| 2023 | 185,000 |
| 2024 | 178,000 |

**Target:** 150,000 tCO2e by 2027

**Linear Regression Fit:**

```
t = [0, 1, 2, 3, 4, 5]  (years from 2019)
E = [220000, 200000, 210000, 195000, 185000, 178000]

n = 6
SUM(t) = 15,  SUM(E) = 1,188,000
SUM(t*E) = 2,815,000,  SUM(t^2) = 55
t_bar = 2.5,  E_bar = 198,000

beta = (6 * 2,815,000 - 15 * 1,188,000) / (6 * 55 - 15^2)
     = (16,890,000 - 17,820,000) / (330 - 225)
     = -930,000 / 105
     = -8,857 tCO2e/year

alpha = 198,000 - (-8,857) * 2.5 = 198,000 + 22,143 = 220,143

Regression: E(t) = 220,143 - 8,857 * t

R^2 = 0.916 (strong linear fit)

Forecasts:
    2025 (t=6): 220,143 - 8,857 * 6 = 167,000 tCO2e
    2026 (t=7): 220,143 - 8,857 * 7 = 158,143 tCO2e
    2027 (t=8): 220,143 - 8,857 * 8 = 149,286 tCO2e

95% CI for 2027: [131,000, 167,500]
```

**Result:** Linear projection suggests the target of 150,000 tCO2e by 2027 is achievable. Point forecast (149,286) is just below target, with the 95% CI lower bound well below and upper bound above.

**Model Selection:** Linear regression selected (MAPE: 3.2%, R^2: 0.916). Consistent downward trend of ~8,857 tCO2e/year supports linear extrapolation.

---

### Example 2: Services Company (Exponential Smoothing)

**Input Data:**

| Year | Emissions (tCO2e) |
|------|-------------------|
| 2019 | 45,000 |
| 2020 | 30,000 |
| 2021 | 35,000 |
| 2022 | 38,000 |
| 2023 | 33,000 |
| 2024 | 28,000 |

**Note:** The COVID dip (2020) and recovery (2021-2022) create a non-linear pattern.

**Linear Regression:**
- beta = -2,629 tCO2e/year, R^2 = 0.603
- MAPE = 8.7%

**Exponential Smoothing (Holt's Damped):**
- Optimal parameters: alpha = 0.65, beta_h = 0.20, phi = 0.90
- MAPE = 5.1%

**Model Selection:** Exponential smoothing selected (MAPE: 5.1% vs 8.7%). The COVID disruption makes exponential smoothing more appropriate because it gives higher weight to recent observations, better capturing the post-COVID recovery and subsequent decline.

```
Level (L_2024) = 28,800 tCO2e
Trend (T_2024) = -3,200 tCO2e/year

Forecasts (damped, phi = 0.90):
    2025: 28,800 + 0.90 * (-3,200) = 25,920 tCO2e
    2026: 28,800 + (0.90 + 0.81) * (-3,200) = 23,328 tCO2e
    2027: 28,800 + (0.90 + 0.81 + 0.729) * (-3,200) = 20,995 tCO2e

80% CI for 2027: [17,500, 24,500] (bootstrapped)
95% CI for 2027: [15,200, 26,800] (bootstrapped)
```

**Interpretation:** Damped trend prevents over-optimistic extrapolation. The declining trend is real but decelerating, suggesting emissions will approach a floor.

---

### Example 3: Energy Company (ARIMA with Structural Break)

**Input Data:**

| Year | Emissions (tCO2e) |
|------|-------------------|
| 2015 | 500,000 |
| 2016 | 510,000 |
| 2017 | 505,000 |
| 2018 | 520,000 |
| 2019 | 515,000 |
| 2020 | 490,000 |
| 2021 | 460,000 |
| 2022 | 430,000 |
| 2023 | 400,000 |
| 2024 | 370,000 |

**Note:** Emissions were flat (2015-2019) then accelerated downward (2020-2024) after a renewable energy transition program began.

**Linear Regression:**
- beta = -16,364 tCO2e/year, R^2 = 0.876
- MAPE = 6.1%

**Exponential Smoothing:**
- MAPE = 4.3%

**ARIMA (automatic selection):**
- Selected order: ARIMA(1, 1, 0)
- AIC: 215.3
- MAPE = 3.2%
- AR(1) coefficient: phi_1 = 0.62 (positive autocorrelation in changes)

**Model Selection:** ARIMA(1,1,0) selected (MAPE: 3.2%, AIC: 215.3). The autocorrelated decline pattern is best captured by ARIMA, which recognizes that large year-over-year drops tend to be followed by similar drops.

```
ARIMA(1,1,0) Forecasts:
    2025: 342,000 tCO2e
    2026: 318,000 tCO2e
    2027: 298,000 tCO2e
    2028: 282,000 tCO2e
    2029: 269,000 tCO2e
    2030: 259,000 tCO2e

95% CI for 2030: [195,000, 323,000]

Target (2030): 250,000 tCO2e

Gap analysis:
    Point forecast: 259,000 (9,000 above target)
    Probability of meeting target: ~42% (target falls within 80% CI)
```

**Interpretation:** The ARIMA model projects emissions slightly above the 2030 target. However, the target falls well within the confidence interval, suggesting that meeting it is plausible but not certain under current trajectory. Additional corrective actions of approximately 9,000 tCO2e may be needed.

### Example Comparison Summary

| Metric | Example 1 (Linear) | Example 2 (Holt's) | Example 3 (ARIMA) |
|--------|--------------------|--------------------|-------------------|
| Data points | 6 | 6 | 10 |
| Selected model | Linear | Holt's Damped | ARIMA(1,1,0) |
| MAPE | 3.2% | 5.1% | 3.2% |
| Key feature | Steady decline | COVID disruption | Structural break |
| Target feasible? | Yes (point below) | Likely (strong decline) | Marginal (9K gap) |

---

## Appendix: Parameter Reference

### Linear Regression Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| alpha | Y-intercept | Any real number |
| beta | Slope (tCO2e/year) | Any real number |
| R^2 | Coefficient of determination | 0 to 1 |
| s | Residual standard error | >= 0 |

### Exponential Smoothing Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| alpha | Level smoothing | (0, 1) | Optimized |
| beta_h | Trend smoothing | (0, 1) | Optimized |
| phi | Trend damping | [0.8, 1.0] | Optimized |
| L_0 | Initial level | Any positive | First observation |
| T_0 | Initial trend | Any real | Second minus first |

### ARIMA Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| p | AR order | 0-3 | Auto-selected |
| d | Differencing order | 0-2 | ADF test |
| q | MA order | 0-3 | Auto-selected |
| phi_i | AR coefficients | Stationarity region | MLE |
| theta_j | MA coefficients | Invertibility region | MLE |

---

**End of Trend Extrapolation Calculation Guide**
