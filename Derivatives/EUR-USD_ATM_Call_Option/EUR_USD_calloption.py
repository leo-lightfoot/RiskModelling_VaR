import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

from scipy.stats import norm

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# Constants
NOTIONAL = 100
MATURITY_DAYS = 30
ANNUAL_BASIS = 365.0

def garman_kohlhagen_call(S, K, T, r_d, r_f, sigma):
    """Price a European FX Call option using Garman-Kohlhagen model."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
    return call_price

# Load data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Parse date
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean numeric data
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

cols_needed = ['fx_eurusd_rate', 'EUR-USD_IVOL', 'fed_funds_rate', 'Euro_STR']
data = clean_numeric_data(data, cols_needed)

# Convert interest rates and vol to decimals
data['fed_funds_rate'] = data['fed_funds_rate'] / 100
data['Euro_STR'] = data['Euro_STR'] / 100
data['EUR-USD_IVOL'] = data['EUR-USD_IVOL'] / 100

# Filter to 2005–2024
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize pricing
data['option_price'] = np.nan
data['days_to_expiry'] = 0
data['roll_date'] = False

# Rolling mechanism
start_date = data.index[0]
current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)

for i in range(len(data)):
    current_date = data.index[i]
    
    # Roll if we reach expiry
    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
    
    # Time to expiry
    days_to_expiry = (current_expiry - current_date).days
    T = days_to_expiry / ANNUAL_BASIS
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry
    
    # Get inputs
    S = data.loc[current_date, 'fx_eurusd_rate']
    sigma = data.loc[current_date, 'EUR-USD_IVOL']
    r_d = data.loc[current_date, 'fed_funds_rate']
    r_f = data.loc[current_date, 'Euro_STR']
    K = S  # ATM option
    
    # Price option if data is available
    if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r_d) and not np.isnan(r_f):
        price = garman_kohlhagen_call(S, K, T, r_d, r_f, sigma)
        data.loc[current_date, 'option_price'] = price

# Compute returns
data['daily_return'] = data['option_price'].pct_change()
data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))

# Adjust return on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        previous = data['option_price'].iloc[i-1]
        current = data['option_price'].iloc[i]
        if previous > 0:
            roll_yield = (current - previous) / previous
            data.loc[data.index[i], 'daily_return'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='NAV_1M_EURUSD_Call')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_return'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_return'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Output dataframe
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_EURUSD_Call': nav.values,
    'Log_Return': data['log_return'].values,
    'Option_Price': data['option_price'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values,
    'EURUSD_Spot': data['fx_eurusd_rate'].values,
    'Vol_1M': data['EUR-USD_IVOL'].values * 100,
    'USD_Rate': data['fed_funds_rate'].values * 100,
    'EUR_Rate': data['Euro_STR'].values * 100
})

# Save output
output_df.to_csv('1m_eurusd_call_option_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='1M EUR/USD ATM Call Option', linewidth=2)
plt.title('NAV of Rolling 1M EUR/USD Call Option (ATM)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('1m_eurusd_call_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV plot.")
