import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Bond pricing parameters
NOTIONAL = 100
COUPON_RATE = 0.025  # 2.5% annual
FREQUENCY = 1        # Annual payments
MATURITY = 5         # 5 years
GREENIUM = -0.002    # -20 bps spread

def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=1):
    """
    Calculate bond price using yield to maturity
    """
    coupon_payment = (coupon_rate / frequency) * NOTIONAL
    periods = int(np.round(years_to_maturity * frequency))
    period_yield = yield_rate / frequency

    # Present value of coupon payments
    if period_yield == 0:
        coupon_pv = coupon_payment * periods
    else:
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield

    # Present value of face value
    face_value_pv = NOTIONAL / (1 + period_yield)**periods

    return coupon_pv + face_value_pv

# Load the dataset
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Parse date
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean and convert 5Y yield
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

data = clean_numeric_data(data, ['5y_treasury_yield'])
data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100  # convert to decimal

# Initialize storage
data['bond_price_green'] = np.nan
data['days_to_maturity_green'] = 0
data['roll_date_green'] = False

# Start date and maturity tracking
start_date = datetime(2005, 1, 1)
end_date = datetime(2024, 12, 31)
current_maturity = start_date + pd.DateOffset(years=MATURITY)

# Align to available data
data = data[(data.index >= start_date) & (data.index <= end_date)]

for i in range(len(data)):
    current_date = data.index[i]

    # Roll over bond on or after maturity
    if current_date >= current_maturity:
        data.loc[current_date, 'roll_date_green'] = True
        current_maturity = current_date + pd.DateOffset(years=MATURITY)

    # Calculate time to maturity
    days_to_maturity = (current_maturity - current_date).days
    time_to_maturity = days_to_maturity / 365.0
    data.loc[current_date, 'days_to_maturity_green'] = days_to_maturity

    # Calculate effective yield (Treasury + greenium)
    base_yield = data.loc[current_date, '5y_treasury_yield']
    if not np.isnan(base_yield):
        effective_yield = base_yield + GREENIUM
        data.loc[current_date, 'bond_price_green'] = calculate_bond_price(
            yield_rate=effective_yield,
            coupon_rate=COUPON_RATE,
            years_to_maturity=time_to_maturity,
            frequency=FREQUENCY
        )

# Calculate returns
data['daily_returns_green'] = data['bond_price_green'].pct_change()
data['log_returns_green'] = np.log(data['bond_price_green'] / data['bond_price_green'].shift(1))

# Adjust return on roll dates
for i in range(1, len(data)):
    if data['roll_date_green'].iloc[i]:
        roll_yield = (data['bond_price_green'].iloc[i] - data['bond_price_green'].iloc[i-1]) / data['bond_price_green'].iloc[i-1]
        data.loc[data.index[i], 'daily_returns_green'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='NAV_Green_Bond')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['daily_returns_green'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns_green'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_Green_Bond': nav.values,
    'Log_Return_Green_Bond': data['log_returns_green'].values,
    'Bond_Price': data['bond_price_green'].values,
    'Days_to_Maturity': data['days_to_maturity_green'].values,
    'Roll_Date': data['roll_date_green'].values,
    'Yield_Treasury_5Y': data['5y_treasury_yield'].values * 100,
    'Effective_Yield': (data['5y_treasury_yield'] + GREENIUM) * 100
})

# Save output
output_df.to_csv('5y_green_bond_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='5-Year Green Bond NAV', linewidth=2)
roll_dates = nav.index[data['roll_date_green']]
plt.scatter(roll_dates, nav[roll_dates], color='green', label='Roll Dates', zorder=5)
plt.title('NAV of Rolling 5-Year Green Bond (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('5y_green_bond_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date_green'].sum()}")
print("Check the CSV file and NAV visualization.")
