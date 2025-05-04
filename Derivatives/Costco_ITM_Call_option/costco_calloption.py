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
MATURITY_DAYS = 90
ANNUAL_BASIS = 365.0
TRANSACTION_COST = 0.005  # 0.5% transaction cost on rolling

def black_scholes_call(S, K, T, r, sigma, q):
    """Price European call using Black-Scholes-Merton."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Load and clean data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean numeric columns
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

cols = ['costco_stock_price', 'COST_IVOL_3MMA', 'fed_funds_rate', 'COST_DIV_YIELD']
data = clean_numeric_data(data, cols)

# Convert interest rates and volatility to decimal
data['fed_funds_rate'] = data['fed_funds_rate'] / 100
data['COST_DIV_YIELD'] = data['COST_DIV_YIELD'] / 100
data['COST_IVOL_3MMA'] = data['COST_IVOL_3MMA'] / 100

# Filter date range
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize columns
data['option_price'] = np.nan
data['days_to_expiry'] = 0
data['roll_date'] = False
data['strike_price'] = np.nan  # Add column to track strike price

# Rolling logic
start_date = data.index[0]
current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
current_strike = 0.9 * data.loc[start_date, 'costco_stock_price']  # Initial strike at 90% of spot

for i in range(len(data)):
    current_date = data.index[i]
    
    # Set strike price for the current date (fixed until next roll)
    data.loc[current_date, 'strike_price'] = current_strike
    
    # Check if we need to roll the option
    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
        # Update strike price only on roll dates to 90% of current spot
        current_strike = 0.9 * data.loc[current_date, 'costco_stock_price']
    
    days_to_expiry = (current_expiry - current_date).days
    T = days_to_expiry / ANNUAL_BASIS
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry
    
    S = data.loc[current_date, 'costco_stock_price']
    sigma = data.loc[current_date, 'COST_IVOL_3MMA']
    r = data.loc[current_date, 'fed_funds_rate']
    q = data.loc[current_date, 'COST_DIV_YIELD']
    K = data.loc[current_date, 'strike_price']  # Use the fixed strike from the column

    if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r) and not np.isnan(q):
        price = black_scholes_call(S, K, T, r, sigma, q)
        data.loc[current_date, 'option_price'] = price

# Calculate returns
data['daily_return'] = data['option_price'].pct_change()
data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))

# Adjust returns on roll dates to account for transaction costs
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        prev = data['option_price'].iloc[i - 1]  # Value of old option
        curr = data['option_price'].iloc[i]      # Value of new option
        if prev > 0:
            # Apply transaction cost when rolling the position
            roll_yield = (curr - prev) / prev - TRANSACTION_COST
            data.loc[data.index[i], 'daily_return'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='NAV_Costco_3M_ITM_Call')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_return'].iloc[i]):
        nav.iloc[i] = nav.iloc[i - 1] * (1 + data['daily_return'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i - 1]

# Output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_Costco_ITM_Call': nav.values,
    'Log_Return': data['log_return'].values,
    'Option_Price': data['option_price'].values,
    'Strike_Price': data['strike_price'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values,
    'Costco_Spot': data['costco_stock_price'].values,
    'Implied_Volatility': data['COST_IVOL_3MMA'].values * 100,
    'Fed_Funds_Rate': data['fed_funds_rate'].values * 100,
    'Dividend_Yield': data['COST_DIV_YIELD'].values * 100
})

# Save CSV
output_df.to_csv('costco_3m_itm_call_option_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='3M ITM Call on Costco', linewidth=2)
plt.title('NAV of Rolling 3M ITM Call on Costco (Strike = 90% of Spot at Roll)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('costco_3m_itm_call_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check 'costco_3m_itm_call_option_data.csv' and NAV plot.")
print(f"Transaction cost of {TRANSACTION_COST*100}% applied at each roll date.")
