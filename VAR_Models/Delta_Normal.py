import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, t

# Load data directly using simplified approach
PORTFOLIO_NAV_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_nav_history.csv"
PORTFOLIO_RETURNS_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_returns_history.csv"
CONFIDENCE_LEVEL = 0.99
Z_SCORE = norm.ppf(1 - CONFIDENCE_LEVEL)
NOTIONAL = 10000000
DEGREES_FREEDOM = 5

# Load portfolio data
df_NAV = pd.read_csv(PORTFOLIO_NAV_URL, parse_dates=['Date'])
portfolio_NAV = df_NAV[['Date', 'NAV']].copy()
df_NAV = df_NAV.drop(columns=['NAV'])
df_NAV = df_NAV.set_index('Date')

df_portfolio_returns = pd.read_csv(PORTFOLIO_RETURNS_URL, parse_dates=['Date'])

if isinstance(df_portfolio_returns, pd.Series):
    df_portfolio_returns = df_portfolio_returns.to_frame(name="Portfolio")

def compute_var_single_asset(returns, window):
    returns_copy = returns.copy()
    
    if 'Date' in returns_copy.columns:
        returns_copy = returns_copy.set_index('Date')
    
    returns_numeric = returns_copy.select_dtypes(include=[np.number])
    rolling_mean = returns_numeric.rolling(window=window).mean()
    rolling_std = returns_numeric.rolling(window=window).std()
    var = -(rolling_mean + Z_SCORE * rolling_std) * NOTIONAL
    return var

df_portfolio_returns_indexed = df_portfolio_returns.copy()
if 'Date' in df_portfolio_returns_indexed.columns:
    df_portfolio_returns_indexed = df_portfolio_returns_indexed.set_index('Date')

VaR_single_asset_100d = compute_var_single_asset(df_portfolio_returns_indexed, 100)
VaR_single_asset_250d = compute_var_single_asset(df_portfolio_returns_indexed, 250)

if isinstance(df_portfolio_returns, pd.DataFrame):
    std_scalar = df_portfolio_returns.select_dtypes(include=[np.number]).std().iloc[0]
    mean_full = df_portfolio_returns.select_dtypes(include=[np.number]).mean().iloc[0]
else:
    std_scalar = df_portfolio_returns.std()
    mean_full = df_portfolio_returns.mean()

VaR_single_asset_full = -(mean_full + Z_SCORE * std_scalar) * NOTIONAL

df_nav_returns = np.log(df_NAV / df_NAV.shift(1)).dropna()
portfolio_nav = df_NAV.sum(axis=1)
portfolio_returns_from_NAV = np.log(portfolio_nav / portfolio_nav.shift(1)).dropna()
initial_nav = portfolio_nav.iloc[0]

df_portfolio_returns_check = df_portfolio_returns.copy()
if 'Date' in df_portfolio_returns_check.columns:
    df_portfolio_returns_check = df_portfolio_returns_check.set_index('Date')
common_dates = portfolio_returns_from_NAV.index.intersection(df_portfolio_returns_check.index)
correlation = portfolio_returns_from_NAV.loc[common_dates].corr(df_portfolio_returns_check.loc[common_dates, 'Return'])
print(f"Correlation between calculated and reported returns: {correlation:.4f}")

def compute_var_covariance_matrix(nav_returns, window, nav_df):
    rolling_var = []
    rolling_dates = []

    for i in range(window, len(nav_returns)):
        current_weights = nav_df.iloc[i-1] / nav_df.iloc[i-1].sum()
        returns_window = nav_returns.iloc[i-window:i]
        mean_returns = returns_window.mean()
        cov_matrix = returns_window.cov()

        portfolio_mean = np.dot(current_weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))

        var = -(portfolio_mean + Z_SCORE * portfolio_std) * NOTIONAL
        rolling_var.append(var)
        rolling_dates.append(nav_returns.index[i])

    return pd.Series(data=rolling_var, index=rolling_dates)

VaR_cov_100d = compute_var_covariance_matrix(df_nav_returns, 100, df_NAV)
VaR_cov_250d = compute_var_covariance_matrix(df_nav_returns, 250, df_NAV)

latest_weights = df_NAV.iloc[-1] / df_NAV.iloc[-1].sum()
mean_returns_full = df_nav_returns.mean()
cov_matrix_full = df_nav_returns.cov()
portfolio_mean = np.dot(latest_weights, mean_returns_full)
portfolio_std = np.sqrt(np.dot(latest_weights.T, np.dot(cov_matrix_full, latest_weights)))
VaR_cov_full = -(portfolio_mean + Z_SCORE * portfolio_std) * NOTIONAL

def compute_var_t_distribution(returns, window, nav_df):
    rolling_var = []
    rolling_dates = []
    
    for i in range(window, len(returns)):
        current_weights = nav_df.iloc[i-1] / nav_df.iloc[i-1].sum()
        returns_window = returns.iloc[i-window:i]
        mean_returns = returns_window.mean()
        cov_matrix = returns_window.cov()
        
        portfolio_mean = np.dot(current_weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
        
        t_quantile = t.ppf(1 - CONFIDENCE_LEVEL, DEGREES_FREEDOM)
        
        scaling_factor = np.sqrt((DEGREES_FREEDOM - 2) / DEGREES_FREEDOM)
        var_t = -(portfolio_mean + t_quantile * portfolio_std * scaling_factor) * NOTIONAL
        
        rolling_var.append(var_t)
        rolling_dates.append(returns.index[i])
    
    return pd.Series(data=rolling_var, index=rolling_dates)

VaR_t_dist_250d = compute_var_t_distribution(df_nav_returns, 250, df_NAV)

print("1-day 99% VaR (Single Asset) - Full Period in USD:", VaR_single_asset_full)

plt.figure(figsize=(14, 6))
var_column = VaR_single_asset_250d.columns[0]

plt.plot(VaR_single_asset_250d.index, VaR_single_asset_250d[var_column], 
         label='250-day Rolling VaR', linestyle='--', linewidth=2)
plt.plot(VaR_single_asset_100d.index, VaR_single_asset_100d[var_column], 
         label='100-day Rolling VaR', linestyle='--', linewidth=2)
plt.axhline(y=VaR_single_asset_full, color='gray', linestyle='-', linewidth=2, label='Full-History VaR')
plt.title('1-Day 99% Delta-Normal VaR (in $) Considering Portfolio as Single Asset')
plt.ylabel('VaR ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

n_days_1 = len(df_portfolio_returns.iloc[250:])
expected_exceptions_1 = int((1 - CONFIDENCE_LEVEL) * n_days_1)
actual_exceptions_1 = (df_portfolio_returns['Return'] * NOTIONAL < -VaR_single_asset_full).sum()

print("\n--- Single Asset VaR Backtesting ---")
print("Expected Exceptions:", expected_exceptions_1)
print("Actual Exceptions:", actual_exceptions_1)
print("Final Full VaR Value:", VaR_single_asset_full)

x = np.arange(0, 2 * expected_exceptions_1 + 20)
pmf = binom.pmf(x, n_days_1, 1 - CONFIDENCE_LEVEL)

plt.figure(figsize=(10, 5))
plt.plot(x, pmf, label='Expected Binomial Distribution')
plt.axvline(expected_exceptions_1, color='green', linestyle='--', label=f'Expected: {expected_exceptions_1}')
plt.axvline(actual_exceptions_1, color='red', linestyle='-', label=f'Actual: {actual_exceptions_1}')
plt.title(f'Method 1: Binomial Test of 99% 1-Day VaR Violations\n(Total Days: {n_days_1})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n----- Multi-Asset VaR Analysis -----")
print(f"1-day 99% VaR (Dynamic Weights) - Full Period in USD: ${VaR_cov_full:,.2f}")

plt.figure(figsize=(14, 6))
plt.plot(VaR_cov_250d.index, VaR_cov_250d, label='Normal 250-day VaR', linestyle='--', linewidth=2)
plt.plot(VaR_t_dist_250d.index, VaR_t_dist_250d, label='t-dist 250-day VaR', linestyle=':', linewidth=2)
plt.axhline(y=VaR_cov_full, color='gray', linestyle='-', linewidth=2, label='Full-History VaR')
plt.title('1-Day 99% Delta-Normal VaR (in $) Considering Individual Securities')
plt.ylabel('VaR ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

n_days_2 = len(df_portfolio_returns.iloc[250:])
expected_exceptions_2 = int((1 - CONFIDENCE_LEVEL) * n_days_2)
actual_exceptions_2 = (df_portfolio_returns['Return'] * NOTIONAL < -VaR_cov_full).sum()

print("\n--- Multi-Asset VaR Backtesting ---")
print("Expected Exceptions:", expected_exceptions_2)
print("Actual Exceptions:", actual_exceptions_2)
print("Final Full VaR Value:", VaR_cov_full)

x = np.arange(0, 2 * expected_exceptions_2 + 20)
pmf = binom.pmf(x, n_days_2, 1 - CONFIDENCE_LEVEL)

plt.figure(figsize=(10, 5))
plt.plot(x, pmf, label='Expected Binomial Distribution')
plt.axvline(expected_exceptions_2, color='green', linestyle='--', label=f'Expected: {expected_exceptions_2}')
plt.axvline(actual_exceptions_2, color='red', linestyle='-', label=f'Actual: {actual_exceptions_2}')
plt.title(f'Method 2: Binomial Test of 99% 1-Day VaR Violations\n(Total Days: {n_days_2})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()