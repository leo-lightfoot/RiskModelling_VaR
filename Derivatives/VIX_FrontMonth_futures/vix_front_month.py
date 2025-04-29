import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean the VIX data
data['vix_index_level'] = pd.to_numeric(data['vix_index_level'], errors='coerce')

# Calculate 30-day moving average (target mean for mean reversion)
vix_mean = data['vix_index_level'].rolling(window=30).mean()

# Set parameters
theta = 0.2           # Speed of mean reversion (adjustable)
sigma = data['vix_index_level'].pct_change().std() * data['vix_index_level'].mean()
contract_days = 21    # Approx 1 month (trading days)
initial_price = data['vix_index_level'].iloc[0]

# Initialize simulation
vix_futures_price = []
current_price = initial_price

for i in range(len(data)):
    if i < 30:
        # Not enough history for moving average
        vix_futures_price.append(np.nan)
        continue
    
    # Start of a new future contract?
    if (i - 30) % contract_days == 0:
        current_price = data['vix_index_level'].iloc[i]  # Assume it realizes to spot
        days_left = contract_days

    # Get long-term mean from moving average
    long_term_mean = vix_mean.iloc[i]

    # Mean-reverting update
    shock = sigma * np.random.normal()
    current_price += theta * (long_term_mean - current_price) + shock
    vix_futures_price.append(current_price)

# Create futures price series
data['vix_futures_1m'] = pd.Series(vix_futures_price, index=data.index)

# Calculate daily returns (percentage change)
daily_returns = data['vix_futures_1m'].pct_change()

# Calculate log returns
log_returns = np.log(data['vix_futures_1m'] / data['vix_futures_1m'].shift(1))

# Calculate NAV starting at 100
nav = pd.Series(index=data.index, name='vix_futures_1m')
nav.iloc[0] = 100
for i in range(1, len(nav)):
    if not np.isnan(daily_returns.iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + daily_returns.iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_VIX_Futures_1M': nav.values,
    'Log_Return_VIX_Futures_1M': log_returns.values
})

# Save to CSV
output_df.to_csv('vix_futures_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='VIX Futures 1M', linewidth=2)
plt.title('NAV of VIX Futures Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('vix_futures_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print("Analysis complete. Check the CSV file and visualizations.")
