# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# =========================
# Load Datasets
# =========================
returns_url = r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\Portfolio\portfolio_results\portfolio_returns_history.csv"
nav_url     = r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\Portfolio\portfolio_results\portfolio_nav_history.csv"

returns_df = pd.read_csv(returns_url)
nav_df     = pd.read_csv(nav_url)

# Convert Dates
returns_df['Date'] = pd.to_datetime(returns_df['Date'])
nav_df['Date']     = pd.to_datetime(nav_df['Date'])

# Merge on Date
df = pd.merge(returns_df, nav_df, on='Date', how='inner')
df.set_index('Date', inplace=True)

# =========================
# Parameters
# =========================
confidence_level = 0.99
notional = 10_000_000

# =========================
# LOG Returns
# =========================
df['LogReturn'] = np.log(df['NAV'] / df['NAV'].shift(1))
df.dropna(subset=['LogReturn'], inplace=True)

# Last 250 observations
returns_last250 = df['LogReturn'].iloc[-250:]
current_nav = df['NAV'].iloc[-1]

# =========================
# Historical Simulation VaR (scaled)
# =========================
simulated_navs = current_nav * np.exp(returns_last250)
portfolio_pnl = simulated_navs - current_nav
scaled_pnl = portfolio_pnl / current_nav * notional

hist_sim_var = -np.quantile(scaled_pnl, 1 - confidence_level)
print(f"\nHistorical Simulation VaR (99% confidence, $10M notional): USD {hist_sim_var:,.2f}")

# =========================
# Plot Histogram of Scaled PnLs
# =========================
plt.figure(figsize=(10, 6))
plt.hist(scaled_pnl, bins=50, color='skyblue', edgecolor='black')
plt.axvline(-hist_sim_var, color='red', linestyle='--', linewidth=2,
            label=f'VaR @ 99% ($10M): USD {hist_sim_var:,.0f}')
plt.title('Historical Simulation: Portfolio PnL Distribution (Scaled to $10M)')
plt.xlabel('Profit and Loss (USD, $10M)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Bootstrap for Scaled VaR CI
# =========================
bootstrap_estimates = []
n_sim = 10000

for _ in range(n_sim):
    sample = np.random.choice(returns_last250, size=250, replace=True)
    sim_nav = current_nav * np.exp(sample)
    pnl = sim_nav - current_nav
    scaled_pnl_sample = pnl / current_nav * notional
    var_estimate = -np.quantile(scaled_pnl_sample, 1 - confidence_level)
    bootstrap_estimates.append(var_estimate)

lower_ci = np.percentile(bootstrap_estimates, 2.5)
upper_ci = np.percentile(bootstrap_estimates, 97.5)

print(f"Bootstrap 95% CI for VaR (scaled to $10M): USD {lower_ci:,.2f} to USD {upper_ci:,.2f}")

# Plot Bootstrap Distribution (scaled)
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_estimates, bins=50, color='lightcoral', edgecolor='black')
plt.axvline(hist_sim_var, color='blue', linestyle='--', linewidth=2, label=f'Original VaR: USD {hist_sim_var:,.0f}')
plt.axvline(lower_ci, color='green', linestyle='--', linewidth=2, label=f'Lower 2.5% CI: USD {lower_ci:,.0f}')
plt.axvline(upper_ci, color='green', linestyle='--', linewidth=2, label=f'Upper 97.5% CI: USD {upper_ci:,.0f}')
plt.title('Bootstrap Distribution of Historical VaR Estimates (Scaled to $10M)')
plt.xlabel('VaR Estimate (USD, $10M)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Rolling VaR + Violations (scaled)
# =========================
window = 250
df['PnL'] = df['NAV'].diff()
df['RollingVaR_Scaled'] = (df['LogReturn']
    .rolling(window=window)
    .apply(lambda x: -np.quantile(x, 1 - confidence_level), raw=True)
) * notional
df['PnL_Scaled'] = df['PnL'] / df['NAV'].shift(1) * notional
df['Exception'] = df['PnL_Scaled'] < -df['RollingVaR_Scaled']

# Plot Rolling VaR vs PnL with exceptions
plt.figure(figsize=(12, 6))
plt.plot(df.index, -df['RollingVaR_Scaled'], label='1-Day 99% Historical VaR ($10M)', color='crimson')
plt.plot(df.index, df['PnL_Scaled'], label='Actual Daily PnL ($10M)', alpha=0.6)
plt.scatter(df[df['Exception']].index, df[df['Exception']]['PnL_Scaled'],
            color='black', marker='x', label='Exceptions (Loss > VaR)')
plt.title('Rolling 1-Day 99% Historical VaR vs. Actual PnL (Scaled to $10M)')
plt.xlabel('Date')
plt.ylabel('USD (scaled to $10M)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Backtest Summary
# =========================
total_days = df['RollingVaR_Scaled'].count()
exceptions = df['Exception'].sum()
expected_exceptions = int((1 - confidence_level) * total_days)

print("\nBacktest Summary (Scaled to $10M):")
print(f"Total Days in VaR Test: {total_days}")
print(f"Observed Exceptions: {exceptions}")
print(f"Expected Exceptions (1% of {total_days}): {expected_exceptions}")

# =========================
# Binomial VaR Exception Test
# =========================
x = np.arange(0, 2 * expected_exceptions)
binom_probs = stats.binom.pmf(x, total_days, 1 - confidence_level)

plt.figure(figsize=(10, 5))
plt.plot(x, binom_probs, label='Expected Binomial Distribution', color='blue')
plt.axvline(expected_exceptions, color='green', linestyle='--', label=f'Expected: {expected_exceptions}')
plt.axvline(exceptions, color='red', linestyle='-', label=f'Actual: {exceptions}')
plt.title(f'Binomial Test of 99% 1-Day VaR Violations (Scaled to $10M)\n(Total Days: {total_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

