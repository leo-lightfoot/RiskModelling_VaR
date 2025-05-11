import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Monte Carlo Parameters
NOTIONAL = 100
DAYS_IN_OPTION = 63  # Approx. 3 months assuming all days are trading days
MC_PATHS = 10000
np.random.seed(42)

# Load the data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Ensure numeric conversion
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

data = clean_numeric_data(data, ['Nikkei_spot', 'NKY_30D_ivol', 'NKY_Div_yield', 'Basic_Loan_Rate_JPY'])

# Convert percentage columns to decimals
data['NKY_30D_ivol'] /= 100
data['NKY_Div_yield'] /= 100
data['Basic_Loan_Rate_JPY'] /= 100

# Initialize containers
data['asian_put_price'] = np.nan
data['roll_date'] = False
data['days_to_maturity'] = 0

# Pricing function using Monte Carlo simulation
def price_asian_put_mc(S0, K, r, q, sigma, T, steps=63, paths=1000):
    dt = T / steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Simulate paths
    Z = np.random.randn(paths, steps)
    price_paths = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))

    # Arithmetic average of each path
    average_price = np.mean(price_paths, axis=1)

    # Asian put payoff
    payoff = np.maximum(K - average_price, 0)

    # Discounted expected payoff
    return np.exp(-r * T) * np.mean(payoff)

# Simulate rolling 3-month Asian puts
# Simulate non-overlapping rolling 3-month Asian puts (roll only at expiry)
current_start = data.index[0]

while current_start + timedelta(days=DAYS_IN_OPTION - 1) <= data.index[-1]:
    current_end = current_start + timedelta(days=DAYS_IN_OPTION - 1)
    date_range = data.loc[current_start:current_end].index

    for i, date in enumerate(date_range):
        if i == 0:
            data.loc[date, 'roll_date'] = True
        data.loc[date, 'days_to_maturity'] = DAYS_IN_OPTION - i

        # Ensure valid data
        if pd.notna(data.loc[date, 'Nikkei_spot']) and pd.notna(data.loc[date, 'NKY_30D_ivol']) \
           and pd.notna(data.loc[date, 'Basic_Loan_Rate_JPY']) and pd.notna(data.loc[date, 'NKY_Div_yield']):

            S0 = data.loc[date, 'Nikkei_spot']
            sigma = data.loc[date, 'NKY_30D_ivol']
            r = data.loc[date, 'Basic_Loan_Rate_JPY']
            q = data.loc[date, 'NKY_Div_yield']
            T = (DAYS_IN_OPTION - i) / 252.0  # Assuming 252 trading days

            K = S0  # ATM strike

            option_price = price_asian_put_mc(S0, K, r, q, sigma, T, steps=DAYS_IN_OPTION - i, paths=MC_PATHS)
            data.loc[date, 'asian_put_price'] = option_price

    # Move start date to the day after current option expires
    current_start = current_end + timedelta(days=1)

# Calculate returns
data['daily_returns'] = data['asian_put_price'].pct_change()
data['log_returns'] = np.log(data['asian_put_price'] / data['asian_put_price'].shift(1))

# Adjust returns on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        prev_price = data['asian_put_price'].iloc[i-1]
        new_price = data['asian_put_price'].iloc[i]
        if not np.isnan(prev_price) and not np.isnan(new_price):
            roll_yield = (new_price - prev_price) / prev_price
            data.loc[data.index[i], 'daily_returns'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='Asian_Put_NAV')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_Asian_Put': nav.values,
    'Log_Return_Asian_Put': data['log_returns'].values,
    'Asian_Put_Price': data['asian_put_price'].values,
    'Days_to_Maturity': data['days_to_maturity'].values,
    'Roll_Date': data['roll_date'].values,
    'Nikkei_Spot': data['Nikkei_spot'].values,
    'Volatility_%': data['NKY_30D_ivol'].values * 100,
    'Dividend_Yield_%': data['NKY_Div_yield'].values * 100,
    'Risk_Free_Rate_%': data['Basic_Loan_Rate_JPY'].values * 100
})

# Save to CSV
output_df.to_csv('asian_put_nikkei_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='3-Month Rolling Asian Put (Nikkei)', linewidth=2)
plt.title('NAV of Rolling 3-Month Asian Put Option on Nikkei', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('asian_put_nikkei_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check 'asian_put_nikkei_data.csv' and NAV plot for details.")
