# ================================================
# 1. Load Data & Construct Portfolio NAV (Revised)
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
nav_path = r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\portfolio_results\portfolio_nav_history.csv"
df = pd.read_csv(nav_path, parse_dates=['Date'], index_col='Date')

# Drop the total NAV column if it exists
if 'NAV' in df.columns:
    df.drop(columns=['NAV'], inplace=True)

# === Allocation & Mapping ===
allocation = {
    'Apple': 0.02, 'Lockheed_martin': 0.02, 'Nvidia': 0.02, 'Procter_gamble': 0.02,
    'Johnson_johnson': 0.02, 'Toyota': 0.02, 'Nestle': 0.02, 'X_steel': 0.02,
    '10y_treasury': 0.05, 'Lqd_etf': 0.04, '10y_tips': 0.05, '1y_eur_zcb': 0.05,
    'High_yield_corp_debt': 0.04, '5y_green_bond': 0.03, '30y_revenue_bond': 0.04,
    'Sp500_futures_1m': 0.04, 'Vix_futures': 0.03, 'Crude_oil_futures': 0.03,
    'Gold_futures': 0.04, 'Soybean_futures': 0.03, 'Costco_itm_call': 0.03,
    'Xom_itm_put': 0.03, 'Eurusd_atm_call': 0.03, 'Usdjpy_atm_put': 0.03,
    'Gbpusd_6m_forward': 0.04, 'Usdinr_3m_forward': 0.04, 'Ford_cds': 0.03,
    'Dax_variance_swap': 0.03, 'Nikkei_asian_put': 0.03, 'Spx_knockout_call': 0.03,
    'Cash': 0.05
}

# Map allocation keys to NAV columns
nav_columns = [f"{k}_NAV" for k in allocation if f"{k}_NAV" in df.columns]

# Only keep instrument NAV columns
df_nav = df[nav_columns]

# Construct portfolio NAV time series
portfolio_nav = df_nav.sum(axis=1)

# Plot log-scale NAV
plt.figure(figsize=(10, 5))
plt.plot(portfolio_nav, label='Portfolio NAV (Initial Allocation)')
plt.yscale('log')
plt.title('Portfolio NAV (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Portfolio NAV (log scale)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()

# ================================================
# 2. ADF Test for Each Asset (Log Returns)
# ================================================
from statsmodels.tsa.stattools import adfuller

# Calculate log returns from NAVs for all further analysis
log_returns = np.log(df_nav / df_nav.shift(1)).dropna()
asset_log_returns = log_returns.copy()
adf_results = {}

for asset in asset_log_returns.columns:
    series = asset_log_returns[asset].dropna()
    result = adfuller(series)
    adf_results[asset] = {'ADF Statistic': result[0], 'p-value': result[1]}

adf_df = pd.DataFrame(adf_results).T
adf_df['Stationary (p < 0.05)'] = adf_df['p-value'] < 0.05

# Keep only stationary assets
stationary_assets = adf_df[adf_df['Stationary (p < 0.05)']].index.tolist()
asset_log_returns = asset_log_returns[stationary_assets]

# ================================================
# 3. Fit ARMA(p,q) per Asset & Residual Diagnostics
# ================================================

import warnings
import statsmodels.api as sm
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# Store ARMA model results
arma_models = {}
residuals_dict = {}

for asset in stationary_assets:
    print(f"\n=== {asset} ===")
    
    series = asset_log_returns[asset].dropna()
    
    # Auto ARMA (no integration)
    model = auto_arima(series,
                       start_p=0, start_q=0,
                       max_p=2, max_q=2,
                       seasonal=False,
                       d=0,  # no integration
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True,
                       information_criterion='aicc')
    
    arma_models[asset] = model
    residuals = pd.Series(model.resid(), index=series.index)
    residuals_dict[asset] = residuals

    
    # Ljung-Box test for residuals (up to lag 14)
    lb_test = sm.stats.acorr_ljungbox(residuals, lags=[14], return_df=True)
    lb_stat = lb_test.iloc[0, 0]
    lb_pval = lb_test.iloc[0, 1]
    
    print(f"Ljung-Box Q(14) = {lb_stat:.2f}, p-value = {lb_pval:.4f}")
    
    if lb_pval > 0.05:
        print("✅ Residuals appear to be white noise.")
    else:
        print("⚠️ Residuals show autocorrelation → ARMA model may be insufficient.")



# ================================================
# 4. Check Squared Residuals for ARCH Effects
# ================================================

garch_candidates = []

for asset in residuals_dict:
    residuals = residuals_dict[asset].dropna()
    squared_residuals = residuals ** 2

    lb_sq_test = sm.stats.acorr_ljungbox(squared_residuals, lags=[14], return_df=True)
    lb_sq_stat = lb_sq_test.iloc[0, 0]
    lb_sq_pval = lb_sq_test.iloc[0, 1]

    print(f"\n=== {asset} ===")
    print(f"Ljung-Box on squared residuals Q(14) = {lb_sq_stat:.2f}, p-value = {lb_sq_pval:.4f}")

    if lb_sq_pval < 0.05:
        print("⚠️ Conditional heteroscedasticity detected → GARCH(1,1) recommended.")
        garch_candidates.append(asset)
    else:
        print("✅ No significant ARCH effect in residuals.")

# Create DataFrame from squared residual Ljung-Box test results
arch_test_results = []

for asset in residuals_dict:
    residuals = residuals_dict[asset].dropna()
    squared_residuals = residuals ** 2

    lb_sq_test = sm.stats.acorr_ljungbox(squared_residuals, lags=[14], return_df=True)
    lb_sq_stat = lb_sq_test.iloc[0, 0]
    lb_sq_pval = lb_sq_test.iloc[0, 1]

    arch_test_results.append({
        'Asset': asset,
        'Ljung-Box Q(14)': round(lb_sq_stat, 2),
        'p-value': round(lb_sq_pval, 4),
        'ARCH Effect Detected': lb_sq_pval < 0.05
    })

# Store as DataFrame
arch_test_df = pd.DataFrame(arch_test_results).set_index('Asset')

# ================================================
# 5. Fit GARCH(1,1) to Assets with ARCH Effects
# ================================================

# GARCH(1,1) is applied only when ARCH effects are statistically significant, ensuring parsimony and model validity.

from arch import arch_model

garch_models = {}
garch_vols = {}
garch_summary = []

for asset in garch_candidates:
    print(f"\n=== GARCH(1,1) Fit for {asset} ===")
    series = asset_log_returns[asset].dropna()

    # Fit GARCH(1,1) with constant mean
    model = arch_model(series, vol='Garch', p=1, q=1, mean='Zero', dist='t')
    result = model.fit(disp='off')

    garch_models[asset] = result
    garch_vols[asset] = result.conditional_volatility

    garch_summary.append({
        'Asset': asset,
        'AIC': round(result.aic, 4),
        'BIC': round(result.bic, 4),
        'Log-Likelihood': round(result.loglikelihood, 2)
    })

# Display GARCH Summary Table
garch_summary_df = pd.DataFrame(garch_summary).sort_values('AIC')
from tabulate import tabulate
print(tabulate(garch_summary_df, headers='keys', tablefmt='pretty'))


# ================================================
# 6. Copula-based Portfolio Sigma Estimation
# ================================================
from scipy.stats import kendalltau, t as t_dist

# Step 0: Align weights to assets correctly
initial_nav = df.iloc[0][asset_log_returns.columns]
weights = initial_nav / initial_nav.sum()
weights = weights.values  # shape: (N,)

# Step 1: Prepare cond_sigmas using GARCH or fallback to 250-day rolling std
rets = asset_log_returns.copy()  # log returns matrix for selected assets
cond_sigmas = pd.DataFrame(index=rets.index)
vol_cap = 0.3  # Cap for asset volatility (adjust as needed)

for asset in rets.columns:
    if asset in garch_vols:
        cond_sigmas[asset] = garch_vols[asset]  # GARCH-based
    else:
        # Fallback: use a longer rolling window (250 days)
        cond_sigmas[asset] = rets[asset].rolling(window=250).std()
    # Cap extreme volatilities
    num_capped = (cond_sigmas[asset] > vol_cap).sum()
    cond_sigmas[asset] = cond_sigmas[asset].clip(upper=vol_cap)
    if num_capped > 0:
        print(f"[INFO] {asset}: {num_capped} values capped at {vol_cap}")

# Step 2: Compute t-copula-based correlation matrix
def compute_t_copula_corrs(rets_window):
    tau_matrix = rets_window.corr(method='kendall').values
    return np.sin(np.pi * tau_matrix / 2)

# Step 3: Portfolio sigma estimator using t-Copula
def copula_portfolio_sigma_t(cond_sigmas, rets, weights, update_freq=5):
    asset_names = cond_sigmas.columns
    all_dates = cond_sigmas.index

    rho_dict = {}
    for idx in range(0, len(all_dates), update_freq):
        date_t = all_dates[idx]
        try:
            end_idx = rets.index.get_loc(date_t)
        except KeyError:
            continue
        start_idx = max(0, end_idx - 60)  # shorter window
        rets_window = rets.iloc[start_idx:end_idx][asset_names].dropna()
        if rets_window.shape[0] < 30:
            continue
        rho = compute_t_copula_corrs(rets_window)
        rho_dict[date_t] = rho

    # Forward fill
    rho_dates = sorted(rho_dict.keys())
    rho_filled = {}
    for i in range(len(rho_dates) - 1):
        start, end = rho_dates[i], rho_dates[i + 1]
        fill_dates = all_dates[(all_dates >= start) & (all_dates < end)]
        for d in fill_dates:
            rho_filled[d] = rho_dict[start]
    for d in all_dates[all_dates >= rho_dates[-1]]:
        rho_filled[d] = rho_dict[rho_dates[-1]]

    # Estimate portfolio sigma
    sigmas = []
    full_weights = pd.Series(weights, index=cond_sigmas.columns)

    for t in all_dates:
        if t not in rho_filled:
            sigmas.append(np.nan)
            continue

        valid_assets = cond_sigmas.loc[t].dropna().index
        rho_assets = cond_sigmas.columns
        valid_assets = valid_assets.intersection(rho_assets)

        if len(valid_assets) < 3:
            sigmas.append(np.nan)
            continue

        sigma_t = cond_sigmas.loc[t, valid_assets].values
        w_t = full_weights[valid_assets].values
        rho_valid = pd.DataFrame(rho_filled[t], index=rho_assets, columns=rho_assets)
        rho_valid = rho_valid.loc[valid_assets, valid_assets].values

        w_t = w_t / np.sum(w_t)
        if np.isnan(sigma_t).any():
            sigmas.append(np.nan)
            continue

        cov = np.diag(sigma_t) @ rho_valid @ np.diag(sigma_t)
        port_sigma = np.sqrt(w_t @ cov @ w_t)
        sigmas.append(port_sigma)

    return pd.Series(sigmas, index=all_dates)

# Run portfolio sigma estimation
print("Estimating portfolio sigma using t-Copula...")
copula_sigma_t = copula_portfolio_sigma_t(cond_sigmas, rets, weights, update_freq=5)

# Plot result
plt.figure(figsize=(12, 5))
plt.plot(copula_sigma_t.dropna(), label="t Copula")
plt.title("Portfolio Conditional Sigma via t Copula (GARCH-based, 250-day window fallback)")
plt.ylabel("Portfolio Sigma")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"t Copula results: {copula_sigma_t.dropna().shape[0]} values, from {copula_sigma_t.dropna().index.min().date()} to {copula_sigma_t.dropna().index.max().date()}")



# ================================================
# 7. Student-t DoF 최적화 기반 VaR 계산 및 Backtest
# ================================================
from scipy.stats import t as student_t
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define the notional amount (example value)
notional = 10000000  # Adjust this value as needed

# Define the confidence level (e.g., for 99% VaR)
confidence_level = 0.99  # Adjust this value as needed

# Step 1: Prepare and verify data alignment
pnl_actual = np.log(portfolio_nav / portfolio_nav.shift(1)).dropna() * notional
common_idx = copula_sigma_t.index.intersection(pnl_actual.index)
copula_sigma_aligned = copula_sigma_t.loc[common_idx]
pnl_actual_aligned = pnl_actual.loc[common_idx]

# Verify alignment
if len(common_idx) == 0:
    raise ValueError("No common dates between portfolio returns and conditional sigmas")
print(f"Data alignment verification: {len(common_idx)} common dates found")

# Step 2: DoF candidates and z-values calculation
# Optionally, allow for a hardcoded conservative DoF
use_hardcoded_dof = False  # Set to True to use a fixed DoF
hardcoded_dof = 5  # More reasonable value for degrees of freedom

dof_range = range(3,50)
backtest_summary = []

for dof in dof_range:
    z_t = student_t.ppf(confidence_level, df=dof)
    VaR_t = z_t * copula_sigma_aligned  # No need to multiply by notional again
    exceptions = (-pnl_actual_aligned > VaR_t * notional)  # Apply notional here for comparison
    total_days = exceptions.count()
    observed = int(exceptions.sum())
    expected = int((1 - confidence_level) * total_days)
    ratio = observed / expected if expected > 0 else float('inf')
    backtest_summary.append({
        'DoF': dof,
        'Observed': observed,
        'Expected': expected,
        'Ratio': round(ratio, 2),
        'Distance': abs(ratio - 1.0),
        'Mean VaR': int(VaR_t.mean() * notional)  # Apply notional for display
    })

bt_df = pd.DataFrame(backtest_summary).sort_values('Distance')
if use_hardcoded_dof:
    optimal_dof = hardcoded_dof
    print(f"\n[INFO] Using hardcoded DoF: {hardcoded_dof}")
else:
    optimal_dof = bt_df.iloc[0]['DoF']
z_star = student_t.ppf(confidence_level, df=optimal_dof)

print("\n=== Student-t DoF 최적화 결과 (99% VaR 기준) ===")
print(tabulate(bt_df.head(10), headers='keys', tablefmt='pretty', showindex=False))
print(f"\n✅ 최적화된 자유도: {optimal_dof:.1f}, z = {z_star:.4f}")

# Step 4: Calculate VaR and perform backtesting with optimal DoF
VaR_optimal = z_star * copula_sigma_aligned  # VaR as percentage
VaR_optimal.name = f"VaR_t(df={int(optimal_dof)})"
exceptions_final = (-pnl_actual_aligned > VaR_optimal * notional)  # Compare with actual PnL

# Plot histogram of actual losses with VaR threshold
plt.figure(figsize=(10, 6))
plt.hist(-pnl_actual_aligned, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Actual Losses')
plt.axvline(VaR_optimal.mean() * notional, color='red', linestyle='--', linewidth=2, label='Mean VaR')
plt.axvline(VaR_optimal.max() * notional, color='orange', linestyle='--', linewidth=2, label='Max VaR')
plt.title('Histogram of Actual Losses with VaR Thresholds')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

summary = {
    'Total Days': len(exceptions_final),
    'Observed Exceptions': int(exceptions_final.sum()),
    'Expected Exceptions': int((1 - confidence_level) * len(exceptions_final)),
    'Mean VaR': int(VaR_optimal.mean() * notional),  # Apply notional for display
    'Max VaR': int(VaR_optimal.max() * notional),    # Apply notional for display
    'Min VaR': int(VaR_optimal.min() * notional)     # Apply notional for display
}

print("\n=== 최적 Student-t 기반 VaR Backtest 결과 ===")
print(tabulate([summary], headers='keys', tablefmt='pretty'))

# Step 5: Plot
plt.figure(figsize=(12, 5))
plt.plot(pnl_actual_aligned, label='Actual PnL')
plt.plot(VaR_optimal * notional, label=f'VaR (t, df={int(optimal_dof)})', color='red')
plt.title('99% 1-Day VaR vs Actual PnL (Optimized t-Distribution)')
plt.xlabel('Date')
plt.ylabel('PnL / VaR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()