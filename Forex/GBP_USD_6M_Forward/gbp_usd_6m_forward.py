import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Forward contract parameters
NOTIONAL_GBP = 1.0  # Notional amount in GBP
DAYS_IN_YEAR = 365
DAYS_FORWARD = 182  # Approximately 6 months

# Function to get next expiry date
def get_next_expiry(date):
    # For a 6-month forward, expiry is 182 days from entry
    return date + timedelta(days=DAYS_FORWARD)

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Set date as index
data.set_index('date', inplace=True)

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, columns):
    for col in columns:
        # Replace "Data Unavailable" with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the GBP/USD spot and interest rate data
data = clean_numeric_data(data, ['fx_gbpusd_rate', 'fed_funds_rate', 'factor_GBP_sonia'])

# Rename columns for clarity
data.rename(columns={
    'fx_gbpusd_rate': 'spot',
    'fed_funds_rate': 'usd_rate',
    'factor_GBP_sonia': 'gbp_rate'
}, inplace=True)

# Convert rates from percentage to decimal
data['usd_rate'] = data['usd_rate'] / 100
data['gbp_rate'] = data['gbp_rate'] / 100

# Drop rows with NaN values in essential columns
data = data.dropna(subset=['spot', 'usd_rate', 'gbp_rate'])

# Initialize columns for forward analysis
data['forward_rate'] = np.nan
data['days_to_expiry'] = np.nan
data['roll_date'] = False
data['contract_pnl'] = np.nan  # P&L of each individual contract
data['active_contract'] = False  # Flag for the currently active contract

# Create NAV series starting at 100
nav_series = pd.Series(index=data.index, data=np.nan)
nav_series.iloc[0] = 100

entry_dates = []
forward_rates = []
expiry_dates = []

# Process each date sequentially
current_contract = None
for i in range(len(data)):
    current_date = data.index[i]
    
    # If we don't have an active contract or the current contract has expired, enter a new one
    if current_contract is None or current_date >= current_contract['expiry_date']:
        # Mark roll date if we're rolling over (not the first contract)
        if current_contract is not None:
            data.loc[current_date, 'roll_date'] = True
        
        # Calculate the forward rate using interest rate parity
        spot = data.loc[current_date, 'spot']
        r_gbp = data.loc[current_date, 'gbp_rate']
        r_usd = data.loc[current_date, 'usd_rate']
        T = DAYS_FORWARD / DAYS_IN_YEAR
        
        # Forward rate formula: Spot * (1 + r_foreign) / (1 + r_base)
        forward_rate = spot * (1 + r_usd * T) / (1 + r_gbp * T)
        
        # Store contract details
        current_contract = {
            'entry_date': current_date,
            'expiry_date': current_date + timedelta(days=DAYS_FORWARD),
            'forward_rate': forward_rate,
            'notional_gbp': NOTIONAL_GBP
        }
        
        # Store dates and rates for debugging/analysis
        entry_dates.append(current_date)
        forward_rates.append(forward_rate)
        expiry_dates.append(current_contract['expiry_date'])
        
        # Mark this row as an active contract entry point
        data.loc[current_date, 'active_contract'] = True
    
    # Update data for the current date
    data.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
    data.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
    
    # Calculate unrealized P&L (mark-to-market)
    # For a GBP/USD forward, buying GBP and selling USD:
    # P&L = Notional * (Spot - Forward Rate)
    spot = data.loc[current_date, 'spot']
    forward_rate = current_contract['forward_rate']
    pnl = NOTIONAL_GBP * (spot - forward_rate)
    data.loc[current_date, 'contract_pnl'] = pnl

# Calculate daily returns based on P&L changes
data['daily_return'] = data['contract_pnl'].diff() / NOTIONAL_GBP
data.loc[data.index[0], 'daily_return'] = 0  # First day has no return

# Fill NaN values in daily returns with zeros (for roll dates)
data['daily_return'].fillna(0, inplace=True)

# Compute cumulative NAV
nav_series = pd.Series(index=data.index)
nav_series.iloc[0] = 100  # Start with 100
for i in range(1, len(data)):
    prev_nav = nav_series.iloc[i-1]
    daily_return = data['daily_return'].iloc[i]
    nav_series.iloc[i] = prev_nav * (1 + daily_return)

# Compute log returns of NAV
log_returns = np.log(nav_series / nav_series.shift(1))

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': data.index,
    'NAV_GBPUSD_6M_Forward': nav_series.values,
    'Log_Return_GBPUSD_6M_Forward': log_returns.values,
    'Forward_Rate': data['forward_rate'].values,
    'Spot_Rate': data['spot'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values,
    'Contract_PnL': data['contract_pnl'].values
})

# Save to CSV
output_df.to_csv('gbpusd_6m_forward_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(data.index, nav_series, label='GBP/USD 6M Forward', linewidth=2)
plt.title('NAV of GBP/USD 6-Month Forward Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('gbpusd_6m_forward_nav.png', bbox_inches='tight', dpi=300)
plt.close()

# Print some summary statistics
num_contracts = len(entry_dates)
print(f"Analysis complete. Number of contracts: {num_contracts}, Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.")
