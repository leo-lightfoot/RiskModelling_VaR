import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Futures pricing parameters
STORAGE_COST = 0.02  # 2% annual storage cost
CONVENIENCE_YIELD = 0.01  # 1% annual convenience yield
TIME_TO_MATURITY = 0.25  # 3 months = 0.25 years

# Function to get next expiry date
def get_next_expiry(date):
    # Gold futures typically expire in February, April, June, August, October, December
    expiry_months = [2, 4, 6, 8, 10, 12]
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
    
    # Set expiry to last business day of the month
    # For simplicity, we'll use the 28th
    return datetime(current_year, next_month, 28)

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

# Clean the gold spot and Fed funds rate data
data = clean_numeric_data(data, ['gold_spot_price', 'fed_funds_rate'])

# Convert Fed funds rate from percentage to decimal
data['fed_funds_rate'] = data['fed_funds_rate'] / 100

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
    data.loc[current_date, 'contract_price'] = data.loc[current_date, 'gold_spot_price'] * np.exp(
        (data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST - CONVENIENCE_YIELD) * time_to_maturity
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
nav = pd.Series(index=data.index, name='gold_futures_3m')
nav.iloc[0] = 100
for i in range(1, len(nav)):
    if not np.isnan(data['daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Export output
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_Gold_Futures_3M': nav.values,
    'Log_Return_Gold_Futures_3M': data['log_returns'].values,
    'Contract_Price': data['contract_price'].values,
    'Days_to_Expiry': data['days_to_expiry'].values,
    'Roll_Date': data['roll_date'].values
})
output_df.to_csv('gold_futures_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='Gold Futures 3M', linewidth=2)
# Mark roll dates
roll_dates = nav.index[data['roll_date']]
plt.title('NAV of Gold Futures Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('gold_futures_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.") 