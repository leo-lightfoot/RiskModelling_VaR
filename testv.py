#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:19:53 2025

@author: adityavijaykumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import binom
import requests
from io import StringIO

# Constants
portfolio_value = 10_000_000
z_99 = norm.ppf(0.99)

# Load pre-calculated portfolio returns
url2 = r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\Portfolio\portfolio_results\portfolio_returns_history.csv'
df_returns = pd.read_csv(url2)
df_returns['Date'] = pd.to_datetime(df_returns['Date'])
df_returns = df_returns.set_index('Date')

# Function to compute VaR using historical returns directly
def delta_normal_var(returns_window, z_score=z_99):
    # Simply calculate standard deviation from historical returns
    std_dev = returns_window['Return'].std()
    # Returns are already in decimal format (e.g., 0.001 = 0.1%)
    return z_score * std_dev * portfolio_value

# Preallocate results DataFrame
var_df = pd.DataFrame(index=df_returns.index)
var_df['VaR_250d'] = np.nan
var_df['VaR_100d'] = np.nan

# Rolling 250d VaR
for i in range(500, len(df_returns)):
    window = df_returns.iloc[i-250:i]
    var_df.iloc[i, var_df.columns.get_loc('VaR_250d')] = delta_normal_var(window)

# Rolling 100d VaR
for i in range(250, len(df_returns)):
    window = df_returns.iloc[i-100:i]
    var_df.iloc[i, var_df.columns.get_loc('VaR_100d')] = delta_normal_var(window)

# Full-history VaR
full_var_value = delta_normal_var(df_returns)
var_df['VaR_full'] = full_var_value

# Drop early rows with NaNs
var_df.dropna(inplace=True)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(var_df.index, var_df['VaR_250d'], label='250-day Rolling VaR', linewidth=2)
plt.plot(var_df.index, var_df['VaR_100d'], label='100-day Rolling VaR', linestyle='--', linewidth=2)
plt.plot(var_df.index, var_df['VaR_full'], label='Full-History VaR', linestyle='-', linewidth=2)

plt.title('1-Day 99% Delta-Normal VaR')
plt.ylabel('VaR (in currency units)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Backtesting - Compare actual returns to VaR
n_days = len(var_df)
confidence_level = 0.99
expected_exceptions = int((1 - confidence_level) * n_days)

# Check how many times actual losses exceeded VaR
portfolio_returns = df_returns.loc[var_df.index]
actual_exceptions = (portfolio_returns['Return'] * portfolio_value < -var_df['VaR_full']).sum()

# Binomial distribution of exceptions
x = np.arange(0, 100)  # Focus on first ~100 exceptions only
pmf = binom.pmf(x, n_days, 1 - confidence_level)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, pmf, label='Expected Binomial Distribution')
plt.axvline(expected_exceptions, color='green', linestyle='--', label=f'Expected: {expected_exceptions}')
plt.axvline(actual_exceptions, color='red', linestyle='-', label=f'Actual: {actual_exceptions}')

plt.title(f'Binomial Test of 99% 1-Day VaR Violations\n(Total Days: {n_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()