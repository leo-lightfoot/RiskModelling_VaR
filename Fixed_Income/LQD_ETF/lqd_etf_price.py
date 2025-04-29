import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Set date as index
data.set_index('date', inplace=True)

# Identify the LQD ETF column
lqd_column = 'lqd_corporate_bond_etf'
print(f"Analyzing LQD ETF data: {lqd_column}")

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, column):
    # Replace "Data Unavailable" with NaN
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Clean the LQD ETF data
data = clean_numeric_data(data, lqd_column)

# Calculate daily returns (percentage change)
daily_returns = data[lqd_column].pct_change()

# Calculate log returns
log_returns = np.log(data[lqd_column] / data[lqd_column].shift(1))

# Calculate NAV for LQD ETF starting with 100
nav = pd.DataFrame(index=data.index, columns=[lqd_column])
nav.iloc[0][lqd_column] = 100
for i in range(1, len(nav)):
    if not np.isnan(daily_returns.iloc[i]):
        nav.iloc[i][lqd_column] = nav.iloc[i-1][lqd_column] * (1 + daily_returns.iloc[i])
    else:
        nav.iloc[i][lqd_column] = nav.iloc[i-1][lqd_column]

# Build a single DataFrame for CSV output
output_df = pd.DataFrame({'Date': nav.index})
output_df['NAV_LQD_ETF'] = nav[lqd_column].values
output_df['Log_Return_LQD_ETF'] = log_returns.values

# Save to CSV
output_df.to_csv('lqd_etf_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav[lqd_column], label='LQD Corporate Bond ETF')
plt.title('NAV of LQD ETF Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('lqd_etf_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print("Analysis complete. NAV and log return data saved to CSV.")
