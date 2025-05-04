import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# TIPS pricing parameters
NOTIONAL = 100
COUPON_RATE = 0.0125  # 1.25% annual
FREQUENCY = 2  # Semiannual payments
MATURITY = 10  # 10 years

def calculate_tips_price(real_yield, coupon_rate, years_to_maturity, frequency=2, inflation_factor=1.0):
    """
    Calculate TIPS price using real yield and inflation adjustment
    """
    if years_to_maturity <= 0:
        return NOTIONAL * inflation_factor
    
    # Adjusted principal for inflation
    adj_principal = NOTIONAL * inflation_factor
    
    # Calculate payments
    periods = int(years_to_maturity * frequency)
    period_yield = real_yield / frequency
    period_coupon = coupon_rate / frequency
    coupon_payment = adj_principal * period_coupon
    
    # Present value of coupon payments
    if abs(period_yield) < 1e-10:  # Handle zero yield case
        coupon_pv = coupon_payment * periods
    else:
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
    
    # Present value of principal
    principal_pv = adj_principal / (1 + period_yield)**periods
    
    return coupon_pv + principal_pv

# Load the data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], dayfirst=True)

# Set date as index
data.set_index('date', inplace=True)

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, columns):
    for col in columns:
        # Replace "Data Unavailable" with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Clean the required data
data = clean_numeric_data(data, ['Real_10Y_yield', 'CPI'])

# Convert yield from percentage to decimal
data['Real_10Y_yield'] = data['Real_10Y_yield'] / 100

# Convert annual inflation percentage to monthly rate
data['monthly_inflation'] = data['CPI'] / 12 / 100
data['monthly_inflation'] = data['monthly_inflation'].fillna(method='ffill')

# Initialize variables for TIPS pricing
data['tips_price'] = np.nan
data['days_to_maturity'] = 0
data['roll_date'] = False
data['inflation_factor'] = np.nan

# Track rollover dates and inflation base
current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
start_date = data.index[0]

# Calculate TIPS prices with roll-over logic
for i in range(len(data)):
    current_date = data.index[i]
    
    # Check if we need to roll over
    if current_date >= current_maturity:
        data.loc[current_date, 'roll_date'] = True
        start_date = current_date  # Reset base date for inflation calculation
        current_maturity = current_date + pd.DateOffset(years=MATURITY)
    
    # Calculate days to maturity
    days_to_maturity = (current_maturity - current_date).days
    data.loc[current_date, 'days_to_maturity'] = days_to_maturity
    time_to_maturity = days_to_maturity / 365.0
    
    # Calculate inflation factor from issue date to current date
    months_elapsed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
    relevant_dates = data.iloc[:i+1]
    relevant_dates = relevant_dates[(relevant_dates.index >= start_date) & (relevant_dates.index <= current_date)]
    
    if len(relevant_dates) > 0:
        avg_monthly_inflation = relevant_dates['monthly_inflation'].mean()
        inflation_factor = (1 + avg_monthly_inflation) ** months_elapsed
    else:
        inflation_factor = 1.0
    
    data.loc[current_date, 'inflation_factor'] = inflation_factor
    
    # Calculate TIPS price
    if not pd.isna(data.loc[current_date, 'Real_10Y_yield']):
        data.loc[current_date, 'tips_price'] = calculate_tips_price(
            data.loc[current_date, 'Real_10Y_yield'],
            COUPON_RATE,
            time_to_maturity,
            FREQUENCY,
            inflation_factor
        )

# Calculate returns
data['daily_returns'] = data['tips_price'].pct_change()
data['log_returns'] = np.log(data['tips_price'] / data['tips_price'].shift(1))

# Adjust returns on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        # Keep same value on roll dates (no artificial jumps)
        data.loc[data.index[i], 'daily_returns'] = 0

# Compute NAV with roll-over adjustments
nav = pd.Series(index=data.index, name='10Y_TIPS')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not pd.isna(data['daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_10Y_TIPS': nav.values,
    'Log_Return_10Y_TIPS': data['log_returns'].values,
    'TIPS_Price': data['tips_price'].values,
    'Days_to_Maturity': data['days_to_maturity'].values,
    'Roll_Date': data['roll_date'].values,
    'Yield': data['Real_10Y_yield'].values * 100,  # Convert back to percentage
    'Inflation_Factor': data['inflation_factor'].values,
    'CPI': data['CPI'].values * 100  # Convert back to percentage
})

# Save to CSV
output_df.to_csv('10_year_tips_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='10-Year TIPS', linewidth=2)
# Mark roll dates
roll_dates = nav.index[data['roll_date']]
plt.scatter(roll_dates, nav[roll_dates], color='red', label='Roll Dates', zorder=5)
plt.title('NAV of 10-Year TIPS Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('10_year_tips_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.")
