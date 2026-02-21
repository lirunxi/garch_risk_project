# Out-of-Sample GARCH Portfolio Risk Engine

A statistically rigorous Python framework for modeling and validating portfolio market risk using **out-of-sample GARCH(1,1) volatility forecasts**, dynamic covariance matrices, and formal **VaR/ES backtesting**.

The project implements a full institutional-style risk pipeline:  
data ingestion → diagnostics → volatility forecasting → portfolio risk → validation → regime analysis → dashboards.

---

## Key Features

- **Out-of-sample GARCH(1,1) volatility forecasting** (expanding window)
- **Time-varying covariance matrices** (GARCH vols + rolling correlations)
- Portfolio **99% VaR & Expected Shortfall** (Parametric & Historical)
- Formal **risk-model backtesting**
  - Kupiec Unconditional Coverage  
  - Christoffersen Independence  
  - Christoffersen Conditional Coverage
- **Volatility-regime performance analysis**
- Risk **visualization dashboards**
- Modular, production-style Python architecture

---

## Methodology Overview

For each asset:

1. Fit GARCH(1,1) using data up to time *t-1*  
2. Forecast 1-step-ahead volatility σᵢ,t  
3. Construct covariance matrix  
   Σₜ = Dₜ Rₜ Dₜ  
   where Dₜ = diag(σᵢ,t)  
4. Compute portfolio volatility  

```
σₚ,t = sqrt(wᵀ Σₜ w)
```

5. Compute VaR / ES  
6. Backtest forecasts vs realized returns  
7. Evaluate across volatility regimes  

---

## Statistical Diagnostics

The framework includes rigorous model validation:

- Augmented Dickey–Fuller stationarity tests  
- Volatility clustering analysis  
- ACF/PACF of returns & squared returns  
- GARCH residual diagnostics  
- Ljung–Box tests  
- VaR coverage & independence tests  

---

## Project Structure

```
risk_model/
    data_loader.py
    preprocessing.py
    eda.py
    garch_models.py
    portfolio.py
    var_es.py
    backtesting.py
    regimes.py

run_full_analysis.py
plot_dashboards.py

output/
    (generated results)
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/<your-username>/oos-garch-risk-engine.git
cd oos-garch-risk-engine
```

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run full risk pipeline:

```bash
python run_full_analysis.py
```

Outputs saved to:

```
output/
```

Generate dashboards:

```bash
python plot_dashboards.py
```

---

## Example Outputs

- Portfolio returns with VaR bands & exceedances  
- GARCH portfolio volatility  
- Parametric vs Historical VaR comparison  
- Regime-based exception rates  

---

## Risk Modeling Components

- Univariate GARCH(1,1)  
- Rolling correlation  
- Dynamic covariance matrices  
- Portfolio volatility  
- Parametric VaR / ES  
- Historical VaR / ES  
- Kupiec & Christoffersen tests  
- Regime analysis  

---

## Backtesting Framework

The engine evaluates:

- Exception frequency vs expected  
- Exception clustering  
- Conditional coverage  
- Regime sensitivity  

This mirrors institutional market-risk validation workflows.

---

## Assets Used

- SPY  
- QQQ  
- TLT  
- GLD  
- AAPL  

Data source: Yahoo Finance (`yfinance`)

---

## Tech Stack

- Python  
- NumPy  
- pandas  
- ARCH  
- statsmodels  
- matplotlib  
- yfinance  

---
