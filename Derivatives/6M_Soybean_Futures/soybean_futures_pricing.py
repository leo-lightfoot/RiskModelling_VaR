import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from statsmodels.tsa.stattools import acf
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Futures pricing parameters
BASE_STORAGE_COST = 0.04     # 4% base annual storage cost
SEASONAL_AMPLITUDE = 0.02    # 2% seasonal fluctuation
CONVENIENCE_YIELD = 0.02     # 2% annual convenience yield
TIME_TO_MATURITY = 0.5       # 6 months = 0.5 years
PHASE_SHIFT = 10             # Peak storage cost in October (month 10)

# Function to get next expiry date
def get_next_expiry(date):
    # Soybean futures typically expire in March, May, July, August, September, November
    expiry_months = [3, 5, 7, 8, 9, 11]
    current_year = date.year
    current_month = date.month
    
    # Find next expiry month
    next_month = None
    for month in expiry_months:
        if month > current_month:
            next_month = month
            break
    
    if next_month is None:
        next_month = expiry_months[0]
        current_year += 1
    
    # Set expiry to 15th of the month
    return datetime(current_year, next_month, 15)

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean numeric columns
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

data = clean_numeric_data(data, ['soybean_spot_usd', 'fed_funds_rate'])
data['fed_funds_rate'] = data['fed_funds_rate'] / 100

# Add month column for seasonality
data['month'] = data.index.month

# Calculate seasonal storage cost
def seasonal_storage_cost(month):
    return BASE_STORAGE_COST + SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (month - PHASE_SHIFT) / 12)

data['seasonal_storage_cost'] = data['month'].apply(seasonal_storage_cost)

# Initialize variables for roll-over logic
data['days_to_expiry'] = 0
data['roll_date'] = False
data['contract_price'] = np.nan

# Calculate futures prices with roll-over logic
current_expiry = get_next_expiry(data.index[0])
for i in range(len(data)):
    current_date = data.index[i]
    
    # Check if we need to roll over
    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = get_next_expiry(current_date)
    
    # Calculate days to expiry
    days_to_expiry = (current_expiry - current_date).days
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry
    
    # Calculate time to maturity in years
    time_to_maturity = days_to_expiry / 365.0
    
    # Calculate futures price
    data.loc[current_date, 'contract_price'] = data.loc[current_date, 'soybean_spot_usd'] * np.exp(
        (data.loc[current_date, 'fed_funds_rate'] + 
         data.loc[current_date, 'seasonal_storage_cost'] - 
         CONVENIENCE_YIELD) * time_to_maturity
    )

# Calculate returns with roll-over adjustment
data['daily_returns'] = data['contract_price'].pct_change()
data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))

# Adjust returns on roll dates to account for roll yield
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        # Calculate roll yield (difference between old and new contract)
        roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
        data.loc[data.index[i], 'daily_returns'] = roll_yield

# Compute NAV with roll-over adjustments
nav = pd.Series(index=data.index, name='soybean_futures_6m')
nav.iloc[0] = 100
for i in range(1, len(nav)):
    if not np.isnan(data['daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Export output
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_Soybean_Futures_6M': nav.values,
    'Log_Return_Soybean_Futures_6M': data['log_returns'].values,
    'Contract_Price': data['contract_price'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values
})
output_df.to_csv('soybean_futures_6m_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='Soybean Futures 6M', linewidth=2)
# Mark roll dates
roll_dates = nav.index[data['roll_date']]
#plt.scatter(roll_dates, nav[roll_dates], color='red', label='Roll Dates', zorder=5)
plt.title('NAV of Soybean 6-Month Futures (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('soybean_futures_6m_nav.png', bbox_inches='tight', dpi=300)
plt.close()

# Simple statistical check
print("\nBasic Statistical Analysis:")
print(f"Mean daily return: {data['daily_returns'].mean():.4f}")
print(f"Volatility (annualized): {data['daily_returns'].std() * np.sqrt(252):.4f}")
print(f"Autocorrelation (lag 1): {acf(data['daily_returns'].dropna(), nlags=1)[1]:.4f}")
print(f"Number of roll-overs: {data['roll_date'].sum()}")

print("\nAnalysis complete. Check CSV and plot for results.")
