import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Bond parameters
NOTIONAL = 100  # EUR
MATURITY = 1  # 1 year
FREQUENCY = 1  # Annual payment (zero coupon)

def calculate_zcb_price(yield_rate, years_to_maturity):
    """
    Calculate zero coupon bond price using continuous compounding
    """
    return NOTIONAL * np.exp(-yield_rate * years_to_maturity)

# Load the data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Convert date to datetime format with correct format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Set date as index
data.set_index('date', inplace=True)

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, columns):
    for col in columns:
        # Replace "Data Unavailable" with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the yield and FX data
data = clean_numeric_data(data, ['1_year_euro_yield_curve', 'fx_eurusd_rate'])

# Convert yield from percentage to decimal
data['1_year_euro_yield_curve'] = data['1_year_euro_yield_curve'] / 100

# Initialize variables for bond pricing
data['zcb_price_eur'] = np.nan
data['days_to_maturity'] = 0
data['roll_date'] = False

# Calculate bond prices with roll-over logic
current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
for i in range(len(data)):
    current_date = data.index[i]
    
    # Check if we need to roll over
    if current_date >= current_maturity:
        data.loc[current_date, 'roll_date'] = True
        current_maturity = current_date + pd.DateOffset(years=MATURITY)
    
    # Calculate days to maturity
    days_to_maturity = (current_maturity - current_date).days
    data.loc[current_date, 'days_to_maturity'] = days_to_maturity
    
    # Calculate time to maturity in years
    time_to_maturity = days_to_maturity / 365.0
    
    # Calculate bond price in EUR
    if not np.isnan(data.loc[current_date, '1_year_euro_yield_curve']):
        data.loc[current_date, 'zcb_price_eur'] = calculate_zcb_price(
            data.loc[current_date, '1_year_euro_yield_curve'],
            time_to_maturity
        )

# Calculate returns in EUR
data['daily_returns_eur'] = data['zcb_price_eur'].pct_change()
data['log_returns_eur'] = np.log(data['zcb_price_eur'] / data['zcb_price_eur'].shift(1))

# Adjust returns on roll dates to account for roll yield
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        # Calculate roll yield (difference between old and new bond)
        roll_yield = (data['zcb_price_eur'].iloc[i] - data['zcb_price_eur'].iloc[i-1]) / data['zcb_price_eur'].iloc[i-1]
        data.loc[data.index[i], 'daily_returns_eur'] = roll_yield

# Compute NAV in EUR
nav_eur = pd.Series(index=data.index, name='1_year_eur_zcb')
nav_eur.iloc[0] = NOTIONAL
for i in range(1, len(nav_eur)):
    if not np.isnan(data['daily_returns_eur'].iloc[i]):
        nav_eur.iloc[i] = nav_eur.iloc[i-1] * (1 + data['daily_returns_eur'].iloc[i])
    else:
        nav_eur.iloc[i] = nav_eur.iloc[i-1]

# Convert NAV to USD
nav_usd = nav_eur * data['fx_eurusd_rate']

# Calculate USD returns
data['daily_returns_usd'] = nav_usd.pct_change()
data['log_returns_usd'] = np.log(nav_usd / nav_usd.shift(1))

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav_eur.index,
    'NAV_EUR': nav_eur.values,
    'NAV_USD': nav_usd.values,
    'Log_Return_USD': data['log_returns_usd'].values,
    'ZCB_Price_EUR': data['zcb_price_eur'].values,
    'Days_to_Maturity': data['days_to_maturity'].values,
    'Roll_Date': data['roll_date'].values,
    'Yield': data['1_year_euro_yield_curve'].values * 100,  # Convert back to percentage
    'EUR_USD_Rate': data['fx_eurusd_rate'].values
})

# Save to CSV
output_df.to_csv('1_year_eur_zcb_data.csv', index=False)

# Plot NAV in USD over time
plt.figure(figsize=(14, 8))
plt.plot(nav_usd.index, nav_usd, label='1-Year EUR ZCB (USD)', linewidth=2)
plt.title('NAV of 1-Year EUR Zero Coupon Bond in USD Over Time (Starting at 100 EUR)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (USD, log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('1_year_eur_zcb_nav_usd.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.")
