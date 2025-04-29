import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Bond parameters
NOTIONAL = 100  # USD
MATURITY = 5  # Typical high-yield bond fund has ~5 year effective duration
COUPON_RATE = 0.065  # 6.5% typical for high yield corporate bonds
FREQUENCY = 2  # Semiannual coupon payments

def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2):
    """
    Calculate bond price using the yield to maturity with coupon payments
    """
    if years_to_maturity <= 0 or np.isnan(yield_rate):
        return np.nan
        
    coupon_payment = (coupon_rate / frequency) * NOTIONAL
    periods = years_to_maturity * frequency
    period_yield = yield_rate / frequency
    
    # Calculate present value of coupon payments
    coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield if period_yield > 0 else coupon_payment * periods
    
    # Calculate present value of face value
    face_value_pv = NOTIONAL / (1 + period_yield)**periods
    
    return coupon_pv + face_value_pv

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

# Clean the yield data
data = clean_numeric_data(data, ['factor_corporate_yield', '10y_treasury_yield', 'high_yield_credit spread'])

# Convert yield from percentage to decimal
data['factor_corporate_yield'] = data['factor_corporate_yield'] / 100
data['10y_treasury_yield'] = data['10y_treasury_yield'] / 100
data['high_yield_credit spread'] = data['high_yield_credit spread'] / 100

# Calculate high yield rate (treasury + high yield spread)
# This is closer to how high yield bonds are priced in reality
data['high_yield_rate'] = data['10y_treasury_yield'] + data['high_yield_credit spread']

# Fill missing values with the corporate yield as fallback
data['high_yield_rate'] = data['high_yield_rate'].fillna(data['factor_corporate_yield'])

# Initialize variables for bond pricing
data['bond_price'] = np.nan
data['days_to_maturity'] = 0
data['roll_date'] = False

# Calculate bond prices with monthly roll-over logic (more realistic for a fund like VWEHX)
current_maturity = data.index[0] + pd.DateOffset(months=1)  # Monthly roll-over
for i in range(len(data)):
    current_date = data.index[i]
    
    # Check if we need to roll over (monthly)
    if current_date >= current_maturity:
        data.loc[current_date, 'roll_date'] = True
        current_maturity = current_date + pd.DateOffset(months=1)
    
    # Calculate days to maturity
    days_to_maturity = (current_maturity - current_date).days + (MATURITY * 365)  # Include the 5-year duration
    data.loc[current_date, 'days_to_maturity'] = days_to_maturity
    
    # Calculate time to maturity in years
    time_to_maturity = days_to_maturity / 365.0
    
    # Calculate bond price with coupon payments
    if not np.isnan(data.loc[current_date, 'high_yield_rate']):
        data.loc[current_date, 'bond_price'] = calculate_bond_price(
            data.loc[current_date, 'high_yield_rate'],
            COUPON_RATE,
            time_to_maturity,
            FREQUENCY
        )

# Calculate returns
data['daily_returns'] = data['bond_price'].pct_change()
data['log_returns'] = np.log(data['bond_price'] / data['bond_price'].shift(1))

# Add coupon accrual component to returns (significant for high yield funds)
daily_coupon_return = COUPON_RATE / 252  # Daily equivalent of annual coupon
data['daily_returns'] = data['daily_returns'] + daily_coupon_return

# Adjust returns on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        # Calculate roll yield (difference between old and new bond)
        roll_yield = (data['bond_price'].iloc[i] - data['bond_price'].iloc[i-1]) / data['bond_price'].iloc[i-1]
        data.loc[data.index[i], 'daily_returns'] = roll_yield + daily_coupon_return

# Compute NAV
nav = pd.Series(index=data.index, name='high_yield_corp_debt')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV': nav.values,
    'Log_Return': data['log_returns'].values,
    'Bond_Price': data['bond_price'].values,
    'Days_to_Maturity': data['days_to_maturity'].values,
    'Roll_Date': data['roll_date'].values,
    'High_Yield_Rate': data['high_yield_rate'].values * 100,  # Convert back to percentage
    'Treasury_Yield': data['10y_treasury_yield'].values * 100,
    'Credit_Spread': data['high_yield_credit spread'].values * 100
})

# Save to CSV
output_df.to_csv('high_yield_corp_debt_data.csv', index=False)

# Plot NAV over time
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='High Yield Corporate Debt', linewidth=2)
plt.title('NAV of High Yield Corporate Debt Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('high_yield_corp_debt_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.") 