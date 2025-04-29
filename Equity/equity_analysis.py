import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\Data\data_restructured.csv')

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Set date as index
data.set_index('date', inplace=True)

# Identify the equity columns
equity_columns = [col for col in data.columns if col.startswith('eq_')]
print(f"Equity columns: {equity_columns}")

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, columns):
    for col in columns:
        # Replace "Data Unavailable" with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the equity data
data = clean_numeric_data(data, equity_columns)

# Calculate daily returns (percentage change)
daily_returns = data[equity_columns].pct_change()

# Calculate log returns
log_returns = np.log(data[equity_columns] / data[equity_columns].shift(1))

# Calculate NAV for each equity starting with 100
nav = pd.DataFrame(index=data.index, columns=equity_columns)
for col in equity_columns:
    nav.iloc[0][col] = 100
    for i in range(1, len(nav)):
        if not np.isnan(daily_returns.iloc[i][col]):
            nav.iloc[i][col] = nav.iloc[i-1][col] * (1 + daily_returns.iloc[i][col])
        else:
            nav.iloc[i][col] = nav.iloc[i-1][col]

# Build a single DataFrame for CSV output
output_df = pd.DataFrame({'Date': nav.index})
for col in equity_columns:
    equity_name = col.replace('eq_', '')
    output_df[f'NAV_{equity_name}'] = nav[col].values
    output_df[f'Log_Return_{equity_name}'] = log_returns[col].values

# Save to CSV
output_df.to_csv('equity_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
for col in equity_columns:
    plt.plot(nav.index, nav[col], label=col.replace('eq_', ''))
plt.title('NAV of Equities Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('equity_nav_log.png', bbox_inches='tight', dpi=300)
plt.close()

print("Analysis complete. Check the CSV file and NAV visualization.") 