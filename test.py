# =========================
# Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# =========================
# Load Return Data
# =========================
returns_url = r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\Portfolio\portfolio_results\portfolio_returns_history.csv"

# Load returns data
returns_df = pd.read_csv(returns_url)
returns_df['Date'] = pd.to_datetime(returns_df['Date'])
returns_df.set_index('Date', inplace=True)

# =========================
# Load NAV Data
# =========================
nav_url = r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\Portfolio\portfolio_results\portfolio_nav_history.csv"
nav_df = pd.read_csv(nav_url)
nav_df['Date'] = pd.to_datetime(nav_df['Date'])
nav_df.set_index('Date', inplace=True)

# Ensure the NAV column is in the correct format
returns_df['NAV'] = nav_df['NAV']

# Print summary of the portfolio
first_date = returns_df.index.min().strftime('%Y-%m-%d')
last_date = returns_df.index.max().strftime('%Y-%m-%d')
start_nav = returns_df['NAV'].iloc[0]
end_nav = returns_df['NAV'].iloc[-1]
total_return = (end_nav / start_nav - 1) * 100

print(f"Portfolio Summary:")
print(f"Period: {first_date} to {last_date}")
print(f"Initial Value: EUR {start_nav:,.2f}")
print(f"Final Value: EUR {end_nav:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {(((end_nav / start_nav) ** (252 / len(returns_df))) - 1) * 100:.2f}%\n")

# =========================
# Normal Distribution VaR
# =========================
# We'll use the last 250 days of returns data
returns_last250 = returns_df['Return'].iloc[-250:].values
current_nav = returns_df['NAV'].iloc[-1]

# Set confidence level
confidence_level = 0.99

# Calculate mean and standard deviation of returns
mean_return = np.mean(returns_last250)
std_return = np.std(returns_last250)

# Calculate VaR using normal distribution
var_normal_pct = -stats.norm.ppf(1 - confidence_level, loc=mean_return, scale=std_return)
var_normal_eur = var_normal_pct * current_nav

# Also calculate historical VaR for comparison
var_hist_pct = -np.percentile(returns_last250, (1 - confidence_level) * 100)
var_hist_eur = var_hist_pct * current_nav

print(f"VaR at 99% Confidence Level:")
print(f"Normal Distribution VaR: EUR {var_normal_eur:,.2f} ({var_normal_pct*100:.2f}%)")
print(f"Historical Simulation VaR: EUR {var_hist_eur:,.2f} ({var_hist_pct*100:.2f}%)\n")

# Plot Histogram with Fitted Distribution
plt.figure(figsize=(12, 8))

# Plot histogram of returns
plt.hist(returns_last250, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# Generate x values for distribution plot
x = np.linspace(min(returns_last250) - 0.01, max(returns_last250) + 0.01, 1000)

# Plot normal distribution
y_normal = stats.norm.pdf(x, mean_return, std_return)
plt.plot(x, y_normal, 'r-', linewidth=2, label='Normal Distribution')

# Add VaR lines
plt.axvline(-var_normal_pct, color='red', linestyle='--', 
           label=f'Normal VaR: {var_normal_pct*100:.2f}%')
plt.axvline(-var_hist_pct, color='black', linestyle=':', 
           label=f'Historical VaR: {var_hist_pct*100:.2f}%')

plt.title('Return Distribution with Normal Distribution')
plt.xlabel('Returns')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Plot Portfolio P&L
# =========================
# Simulate today's possible P&Ls based on historical returns
portfolio_pnl = current_nav * returns_last250

plt.figure(figsize=(12, 6))
plt.hist(portfolio_pnl, bins=50, color='skyblue', edgecolor='black')
plt.axvline(-var_normal_eur, color='red', linestyle='--', linewidth=2, 
           label=f'Normal VaR: EUR {var_normal_eur:,.2f}')
plt.axvline(-var_hist_eur, color='black', linestyle=':', linewidth=2, 
           label=f'Historical VaR: EUR {var_hist_eur:,.2f}')
plt.title('Portfolio P&L Distribution (99% Confidence)')
plt.xlabel('Profit and Loss (EUR)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Bootstrap for VaR Confidence Interval
# =========================
bootstrap_estimates = []
n_sim = 10000

for _ in range(n_sim):
    # Sample returns with replacement
    sampled_returns = np.random.choice(returns_last250, size=500, replace=True)
    
    # Calculate parameters from the sample
    sample_mean = np.mean(sampled_returns)
    sample_std = np.std(sampled_returns)
    
    # Calculate VaR using normal distribution
    var_estimate = -stats.norm.ppf(1 - confidence_level, loc=sample_mean, scale=sample_std)
    bootstrap_estimates.append(var_estimate * current_nav)

# Calculate confidence intervals
lower_ci = np.percentile(bootstrap_estimates, 2.5)
upper_ci = np.percentile(bootstrap_estimates, 97.5)

print(f"VaR 99% Confidence Interval (bootstrap 95%):")
print(f"EUR {lower_ci:,.2f} to EUR {upper_ci:,.2f}")

# Plot Bootstrap Distribution
plt.figure(figsize=(12, 6))
plt.hist(bootstrap_estimates, bins=50, color='skyblue', edgecolor='black')
plt.axvline(var_normal_eur, color='red', linestyle='--', linewidth=2, 
           label=f'Normal VaR: EUR {var_normal_eur:,.2f}')
plt.axvline(lower_ci, color='green', linestyle='--', linewidth=2, 
           label=f'Lower 2.5% CI: EUR {lower_ci:,.2f}')
plt.axvline(upper_ci, color='green', linestyle='--', linewidth=2, 
           label=f'Upper 97.5% CI: EUR {upper_ci:,.2f}')

plt.title('Bootstrap Distribution of VaR Estimates')
plt.xlabel('VaR Estimate (EUR)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Rolling VaR + Violations
# =========================
# Calculate rolling VaR with a window of 360 days
window = 250
returns_df['PnL'] = returns_df['NAV'].diff()
returns_df['ReturnPct'] = returns_df['Return']  # Just for clarity, same as Return column

# Calculate rolling VaR as a percentage first, then translate to EUR values
rolling_var_pct = []
rolling_var_eur = []
exceptions = []

for i in range(window, len(returns_df)):
    # Historical window of returns
    hist_returns = returns_df['ReturnPct'].iloc[i-window:i].values
    
    # Calculate mean and standard deviation for this window
    window_mean = np.mean(hist_returns)
    window_std = np.std(hist_returns)
    
    # Calculate VaR using normal distribution
    var_pct = -stats.norm.ppf(1 - confidence_level, loc=window_mean, scale=window_std)
    rolling_var_pct.append(var_pct)
    
    # Previous day's NAV
    prev_nav = returns_df['NAV'].iloc[i-1]
    
    # Calculate VaR in EUR
    rolling_var_eur.append(var_pct * prev_nav)
    
    # Actual return for the next day
    actual_return = returns_df['ReturnPct'].iloc[i]
    
    # Check for VaR exceptions (actual loss exceeding VaR)
    exceptions.append(actual_return < -var_pct)

# Add rolling VaR to the DataFrame
rolling_start_idx = window
rolling_df = returns_df.iloc[rolling_start_idx:].copy()
rolling_df['RollingVaR_Pct'] = rolling_var_pct
rolling_df['RollingVaR_EUR'] = rolling_var_eur
rolling_df['Exception'] = exceptions

# Plot Rolling VaR (as a percentage) vs Actual Returns
plt.figure(figsize=(12, 6))
plt.plot(rolling_df.index, rolling_df['ReturnPct'] * 100, 
         label='Actual Daily Return (%)', alpha=0.6, color='gray')
plt.plot(rolling_df.index, rolling_df['RollingVaR_Pct'] * 100, 
         label='Normal 99% VaR (%)', color='red')

# Plot exceptions
plt.scatter(rolling_df[rolling_df['Exception']].index, 
           rolling_df[rolling_df['Exception']]['ReturnPct'] * 100, 
           color='black', marker='x', s=50, label='VaR Violations')

plt.title('Rolling 1-Day 99% VaR vs. Actual Returns (%)')
plt.xlabel('Date')
plt.ylabel('Percentage (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Backtest Summary
total_days = len(rolling_df)
observed_exceptions = rolling_df['Exception'].sum()
expected_exceptions = total_days * (1 - confidence_level)
exception_rate = observed_exceptions / total_days

print("\nBacktest Summary:")
print(f"Total Days in VaR Test: {total_days}")
print(f"Expected Exceptions ({(1-confidence_level)*100:.1f}% of {total_days}): {expected_exceptions:.1f}")
print(f"Observed Exceptions: {observed_exceptions}")
print(f"Exception Rate: {exception_rate:.2%}\n")

# =========================
# Kupiec Test
# =========================
def kupiec_test(exceptions, total, alpha):
    """
    Perform Kupiec's Proportion of Failures (POF) test
    """
    observed_rate = exceptions / total
    expected_rate = alpha
    
    # Calculate the test statistic
    if exceptions == 0:
        # Avoid log(0) case
        lr_test_statistic = -2 * (total * np.log(1 - expected_rate))
    else:
        lr_test_statistic = -2 * (
            total * np.log(1 - expected_rate) + exceptions * np.log(expected_rate) -
            exceptions * np.log(observed_rate) - (total - exceptions) * np.log(1 - observed_rate)
        )
    
    # The test statistic follows a chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_test_statistic, 1)
    
    return {
        'test_statistic': lr_test_statistic,
        'p_value': p_value,
        'is_valid': p_value > 0.05
    }

# Perform Kupiec test
alpha = 1 - confidence_level
kupiec_result = kupiec_test(observed_exceptions, total_days, alpha)

print("Kupiec Test Results:")
print(f"Test Statistic: {kupiec_result['test_statistic']:.4f}")
print(f"P-value: {kupiec_result['p_value']:.4f}")
print(f"Model Validation: {'VALID' if kupiec_result['is_valid'] else 'INVALID'}")

# =========================
# Binomial Distribution of Exceptions
# =========================
# Plot binomial distribution with exception count
x_values = np.arange(0, max(int(expected_exceptions * 3), observed_exceptions + 10))
pmf_values = stats.binom.pmf(x_values, total_days, alpha)

plt.figure(figsize=(12, 6))
plt.bar(x_values, pmf_values, alpha=0.7, label='Binomial PMF')
plt.axvline(observed_exceptions, color='red', linestyle='--', linewidth=2, 
           label=f'Observed Exceptions: {observed_exceptions}')
plt.axvline(expected_exceptions, color='green', linestyle='-', linewidth=2, 
           label=f'Expected Exceptions: {expected_exceptions:.1f}')

# Add text about p-value and test result
plt.text(0.05, 0.95, 
         f"P-value: {kupiec_result['p_value']:.4f}\nModel validation: {'VALID' if kupiec_result['is_valid'] else 'INVALID'}",
        transform=plt.gca().transAxes, fontsize=12, 
        bbox=dict(facecolor='white', alpha=0.8))

plt.title('Binomial Distribution of VaR Exceptions')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
