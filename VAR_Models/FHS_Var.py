import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy.stats import binom
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Set seed
np.random.seed(42)

# Load data directly using simplified approach
returns_url = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_returns_history.csv"
nav_url = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_nav_history.csv"

returns_df = pd.read_csv(returns_url, parse_dates=['Date'])
nav_df = pd.read_csv(nav_url, parse_dates=['Date'])
df = pd.merge(returns_df, nav_df, on='Date', how='inner').set_index('Date')

# Parameters
confidence_level = 0.99
notional = 10_000_000
window = 250
bootstrap_n = 10000

# Compute log returns
df['LogReturn'] = np.log(df['NAV'] / df['NAV'].shift(1))
df.dropna(inplace=True)
returns = df['LogReturn'] * 100  # in percentage units for GARCH

# Fit GARCH(1,1)
model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
garch_fit = model.fit(disp='off')
resids = garch_fit.std_resid.dropna()
vols = garch_fit.conditional_volatility / 100  # back to return units

# Filtered Historical Simulation (Rolling)
fhs_var_series = []
fhs_es_series = []
for t in range(window, len(returns)):
    sigma_t = vols.iloc[t]
    boot_resids = np.random.choice(resids, size=1000, replace=True)
    simulated_returns = sigma_t * boot_resids
    var = -np.quantile(simulated_returns, 1 - confidence_level)
    es = -np.mean(simulated_returns[simulated_returns < -var])
    fhs_var_series.append(var * notional)
    fhs_es_series.append(es * notional)

# Store results
fhs_dates = df.index[window:]
df['RollingFHS_VaR'] = np.nan
df['RollingFHS_ES'] = np.nan
df.loc[fhs_dates, 'RollingFHS_VaR'] = fhs_var_series
df.loc[fhs_dates, 'RollingFHS_ES'] = fhs_es_series

# Compute PnL
df['PnL'] = df['NAV'].diff()
df['PnL_Scaled'] = df['PnL'] / df['NAV'].shift(1) * notional
df['Exception'] = df['PnL_Scaled'] < -df['RollingFHS_VaR']

# === Plot 1: Rolling VaR vs PnL
plt.figure(figsize=(12, 6))
plt.plot(df.index, -df['RollingFHS_VaR'], label='1-Day 99% FHS VaR ($10M)', color='crimson')
plt.plot(df.index, df['PnL_Scaled'], label='Actual Daily PnL ($10M)', alpha=0.6, color='steelblue')
plt.scatter(df[df['Exception']].index, df[df['Exception']]['PnL_Scaled'],
            color='black', marker='x', label='Exceptions (Loss > VaR)')
plt.title('Rolling 1-Day 99% FHS VaR vs. Actual PnL (Scaled to $10M)')
plt.xlabel('Date')
plt.ylabel('USD (scaled to $10M)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Plot 2: Histogram of Most Recent Simulated PnLs
sigma_last = vols.iloc[-1]
sim_resids = np.random.choice(resids, size=1000)
sim_pnls = sigma_last * sim_resids * notional
hist_sim_var = -np.quantile(sim_pnls, 1 - confidence_level)

plt.figure(figsize=(10, 6))
plt.hist(sim_pnls, bins=50, color='skyblue', edgecolor='black')
plt.axvline(-hist_sim_var, color='red', linestyle='--', linewidth=2,
            label=f'VaR @ 99% ($10M): USD {hist_sim_var:,.0f}')
plt.title('FHS: Portfolio PnL Distribution (Latest Day, Scaled to $10M)')
plt.xlabel('Profit and Loss (USD, $10M)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: Bootstrap FHS VaR Distribution
bootstrap_estimates = []
for _ in range(bootstrap_n):
    boot = np.random.choice(resids, size=250, replace=True)
    sigma = np.std(boot)
    sim_returns = sigma * np.random.choice(boot, size=1000)
    var_est = -np.quantile(sim_returns, 1 - confidence_level) * notional
    bootstrap_estimates.append(var_est)

lower_ci = np.percentile(bootstrap_estimates, 2.5)
upper_ci = np.percentile(bootstrap_estimates, 97.5)

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_estimates, bins=50, color='salmon', edgecolor='black')
plt.axvline(hist_sim_var, color='blue', linestyle='--', linewidth=2, label=f'Latest VaR: USD {hist_sim_var:,.0f}')
plt.axvline(lower_ci, color='green', linestyle='--', linewidth=2, label=f'2.5% CI: USD {lower_ci:,.0f}')
plt.axvline(upper_ci, color='green', linestyle='--', linewidth=2, label=f'97.5% CI: USD {upper_ci:,.0f}')
plt.title('Bootstrap Distribution of FHS VaR Estimates (Scaled to $10M)')
plt.xlabel('VaR Estimate (USD)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Plot 4: Binomial Test
total_days = df['RollingFHS_VaR'].count()
exceptions = df['Exception'].sum()
expected = int((1 - confidence_level) * total_days)
x = np.arange(0, 2 * expected)
binom_probs = binom.pmf(x, total_days, 1 - confidence_level)

plt.figure(figsize=(10, 5))
plt.plot(x, binom_probs, label='Expected Binomial Distribution', color='blue')
plt.axvline(expected, color='green', linestyle='--', label=f'Expected: {expected}')
plt.axvline(exceptions, color='red', linestyle='-', label=f'Actual: {exceptions}')
plt.title(f'Binomial Test of 99% 1-Day VaR Violations (FHS, $10M)\n(Total Days: {total_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Ljung-Box Test on Residuals ===
lb_test = acorr_ljungbox(resids, lags=[10], return_df=True)
print("\nLjung-Box Test on Standardized Residuals (lag 10)")
print(lb_test)

# === ADF Stationarity Test on Last 250 Days ===
adf_result = adfuller(resids[-250:])
print("\nADF Test on Last 250 GARCH Residuals (Bootstrap Window)")
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("=> Residuals are stationary (good for FHS bootstrapping).")
else:
    print("=> Residuals are NOT stationary (FHS may be unreliable).")

# === ACF Plot for Residuals ===
sm.graphics.tsa.plot_acf(resids, lags=20)
plt.title("ACF of Standardized GARCH Residuals")
plt.tight_layout()
plt.show()

# === Sensitivity Analysis: Vary Bootstrap Window Sizes ===
sensitivity_results = {}
for w in [100, 250, 500]:
    boot = np.random.choice(resids[-w:], size=1000, replace=True)
    sigma = np.std(boot)
    sim_returns = sigma * np.random.choice(boot, size=1000)
    var_est = -np.quantile(sim_returns, 1 - confidence_level) * notional
    sensitivity_results[w] = var_est

print("\nBootstrap VaR Sensitivity to Window Size:")
for w, var in sensitivity_results.items():
    print(f"Window {w} days: 1-day 99% VaR = USD {var:,.0f}")
