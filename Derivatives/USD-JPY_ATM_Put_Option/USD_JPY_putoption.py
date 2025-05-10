import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
from scipy.stats import norm

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# Constants
NOTIONAL = 100
MATURITY_DAYS = 90
ANNUAL_BASIS = 365.0

def garman_kohlhagen_put(S, K, T, r_d, r_f, sigma):
    """Price a European FX Put option using Garman-Kohlhagen model (price in USD)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
    return put_price

# Load data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Parse date
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean relevant numeric data
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

cols = ['fx_usdjpy_rate', 'USD-JPY_IVOL', 'fed_funds_rate', 'Basic_Loan_Rate_JPY']
data = clean_numeric_data(data, cols)

# Convert interest rates and vol to decimals
data['fed_funds_rate'] = data['fed_funds_rate'] / 100
data['Basic_Loan_Rate_JPY'] = data['Basic_Loan_Rate_JPY'] / 100
data['USD-JPY_IVOL'] = data['USD-JPY_IVOL'] / 100

# Filter date range
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize pricing columns
data['option_price'] = np.nan
data['days_to_expiry'] = 0
data['roll_date'] = False

# Rolling logic
start_date = data.index[0]
current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)

for i in range(len(data)):
    current_date = data.index[i]
    
    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
    
    days_to_expiry = (current_expiry - current_date).days
    T = days_to_expiry / ANNUAL_BASIS
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry
    
    S = data.loc[current_date, 'fx_usdjpy_rate']
    sigma = data.loc[current_date, 'USD-JPY_IVOL']
    r_d = data.loc[current_date, 'fed_funds_rate']
    r_f = data.loc[current_date, 'Basic_Loan_Rate_JPY']
    K = S  # ATM Put → Strike = Spot
    
    if days_to_expiry == 0:
        # Set to intrinsic value at expiry
        price = max(K - S, 0)
    elif not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r_d) and not np.isnan(r_f):
        price = garman_kohlhagen_put(S, K, T, r_d, r_f, sigma)
    else:
        price = np.nan
    data.loc[current_date, 'option_price'] = price

# Calculate returns
data['daily_return'] = data['option_price'].pct_change()
data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))

# Adjust returns on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        prev = data['option_price'].iloc[i - 1]
        curr = data['option_price'].iloc[i]
        if prev > 0:
            roll_yield = (curr - prev) / prev
            data.loc[data.index[i], 'daily_return'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='NAV_3M_USDJPY_Put')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_return'].iloc[i]):
        nav.iloc[i] = nav.iloc[i - 1] * (1 + data['daily_return'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i - 1]

# Output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_USDJPY_Put': nav.values,
    'Log_Return': data['log_return'].values,
    'Option_Price_USD': data['option_price'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values,
    'USDJPY_Spot': data['fx_usdjpy_rate'].values,
    'Vol_3M': data['USD-JPY_IVOL'].values * 100,
    'USD_Rate': data['fed_funds_rate'].values * 100,
    'JPY_Rate': data['Basic_Loan_Rate_JPY'].values * 100
})

# Save CSV
output_df.to_csv('3m_usdjpy_put_option_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='3M USD/JPY ATM Put Option (USD)', linewidth=2)

plt.title('NAV of Rolling 3M USD/JPY ATM Put Option (in USD)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('3m_usdjpy_put_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check '3m_usdjpy_put_option_data.csv' and NAV plot.")
