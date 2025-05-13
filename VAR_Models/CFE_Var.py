import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, binom
import requests
from io import StringIO

returns_url = "https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/portfolio_results/portfolio_returns_history.csv"
nav_url = "https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/portfolio_results/portfolio_nav_history.csv"

response_returns = requests.get(returns_url)
response_nav = requests.get(nav_url)

if response_returns.status_code == 200 and response_nav.status_code == 200:
    returns_df = pd.read_csv(StringIO(response_returns.text))
    nav_df = pd.read_csv(StringIO(response_nav.text))
    
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    nav_df['Date'] = pd.to_datetime(nav_df['Date'])
    df = pd.merge(returns_df, nav_df, on='Date', how='inner').set_index('Date')

# Parameters
confidence_level = 0.99
notional = 10_000_000
window = 250
alpha = 1 - confidence_level

# Compute log returns
df['LogReturn'] = np.log(df['NAV'] / df['NAV'].shift(1))
df.dropna(inplace=True)

cf_var_list = []
cf_dates = []

for i in range(window, len(df)):
    sample = df['LogReturn'].iloc[i - window:i]
    mu = np.mean(sample)
    sigma = np.std(sample)
    gamma1 = np.clip(skew(sample), -2, 2)  # capped skewness
    gamma2 = np.clip(kurtosis(sample, fisher=True), -7, 7)  # capped excess kurtosis

    z = norm.ppf(alpha)
    z_cf = z + (1/6)*(z**2 - 1)*gamma1 + (1/24)*(z**3 - 3*z)*gamma2 - (1/36)*(2*z**3 - 5*z)*gamma1**2

    # Fallback if z_cf explodes
    if np.isnan(z_cf) or abs(z_cf) > 10:
        z_cf = np.quantile(sample, alpha)
        var_cf = -z_cf * notional
    else:
        var_cf = -(mu + sigma * z_cf) * notional

    cf_var_list.append(var_cf)
    cf_dates.append(df.index[i])

df['CF_VaR'] = np.nan
df.loc[cf_dates, 'CF_VaR'] = cf_var_list
df['PnL'] = df['NAV'].diff()
df['PnL_Scaled'] = df['PnL'] / df['NAV'].shift(1) * notional
df['Exception_CF'] = df['PnL_Scaled'] < -df['CF_VaR']

# === Plot: Cornish-Fisher VaR vs. PnL
plt.figure(figsize=(12, 6))
plt.plot(df.index, -df['CF_VaR'], label='1-Day 99% CF VaR ($10M)', color='crimson')
plt.plot(df.index, df['PnL_Scaled'], label='Actual Daily PnL ($10M)', color='steelblue', alpha=0.6)
plt.scatter(df[df['Exception_CF']].index, df[df['Exception_CF']]['PnL_Scaled'],
            color='black', marker='x', label='Exceptions (Loss > VaR)')
plt.title('Cornish-Fisher VaR vs Actual PnL (Scaled to $10M)')
plt.xlabel('Date')
plt.ylabel('USD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Binomial Test
total_days = df['CF_VaR'].count()
exceptions = df['Exception_CF'].sum()
expected = int((1 - confidence_level) * total_days)
x = np.arange(0, 2 * expected)
binom_probs = binom.pmf(x, total_days, 1 - confidence_level)

plt.figure(figsize=(10, 5))
plt.plot(x, binom_probs, label='Expected Binomial Distribution', color='blue')
plt.axvline(expected, color='green', linestyle='--', label=f'Expected: {expected}')
plt.axvline(exceptions, color='red', linestyle='-', label=f'Actual: {exceptions}')
plt.title(f'Binomial Test of 99% 1-Day VaR Violations (CF, $10M)\n(Total Days: {total_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === Plot: Histogram of Most Recent Cornish-Fisher PnL Distribution ===
sample = df['LogReturn'].iloc[-window:]
mu = np.mean(sample)
sigma = np.std(sample)
gamma1 = np.clip(skew(sample), -2, 2)
gamma2 = np.clip(kurtosis(sample, fisher=True), -7, 7)

z = norm.ppf(alpha)
z_cf = z + (1/6)*(z**2 - 1)*gamma1 + (1/24)*(z**3 - 3*z)*gamma2 - (1/36)*(2*z**3 - 5*z)*gamma1**2
if np.isnan(z_cf) or abs(z_cf) > 10:
    simulated_returns = sample
    var_hist = -np.quantile(sample, 1 - confidence_level)
else:
    simulated_returns = mu + sigma * np.random.normal(size=1000)
    var_hist = -(mu + sigma * z_cf)

sim_pnls = simulated_returns * notional

plt.figure(figsize=(10, 6))
plt.hist(sim_pnls, bins=50, color='skyblue', edgecolor='black')
plt.axvline(-var_hist * notional, color='red', linestyle='--', linewidth=2,
            label=f'VaR @ 99% ($10M): USD {-var_hist * notional:,.0f}')
plt.title('Cornish-Fisher: Simulated PnL Distribution (Latest Day, Scaled to $10M)')
plt.xlabel('Profit and Loss (USD)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot: Bootstrap VaR Distribution ===
bootstrap_estimates = []
for _ in range(1000):
    boot = np.random.choice(sample, size=window, replace=True)
    mu_b = np.mean(boot)
    sigma_b = np.std(boot)
    gamma1_b = np.clip(skew(boot), -2, 2)
    gamma2_b = np.clip(kurtosis(boot, fisher=True), -7, 7)
    z_b = norm.ppf(alpha)
    z_cf_b = z_b + (1/6)*(z_b**2 - 1)*gamma1_b + (1/24)*(z_b**3 - 3*z_b)*gamma2_b - (1/36)*(2*z_b**3 - 5*z_b)*gamma1_b**2

    if np.isnan(z_cf_b) or abs(z_cf_b) > 10:
        var_est = -np.quantile(boot, 1 - confidence_level) * notional
    else:
        var_est = -(mu_b + sigma_b * z_cf_b) * notional  # 👈 CF VaR as negative value

    bootstrap_estimates.append(var_est)

# Confidence intervals
lower_ci = np.percentile(bootstrap_estimates, 2.5)
upper_ci = np.percentile(bootstrap_estimates, 97.5)

# Latest CF VaR (already negative)
latest_var = -(mu + sigma * z_cf) * notional

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_estimates, bins=50, color='salmon', edgecolor='black')
plt.axvline(latest_var, color='blue', linestyle='--', linewidth=2, label=f'Latest VaR: USD {latest_var:,.0f}')
plt.axvline(lower_ci, color='green', linestyle='--', linewidth=2, label=f'2.5% CI: USD {lower_ci:,.0f}')
plt.axvline(upper_ci, color='green', linestyle='--', linewidth=2, label=f'97.5% CI: USD {upper_ci:,.0f}')

plt.title('Bootstrap Distribution of Cornish-Fisher VaR Estimates (Scaled to $10M)')
plt.xlabel('VaR Estimate (USD Loss)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
