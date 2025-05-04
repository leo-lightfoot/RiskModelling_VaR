import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

# Bond pricing parameters
NOTIONAL = 100
COUPON_RATE = 0.04  # 4% annual
FREQUENCY = 2  # Semiannual payments
ISSUE_DATE = pd.Timestamp("2005-01-01")
FINAL_MATURITY = pd.Timestamp("2034-01-01")
SPREAD = 0.008  # 80 bps

def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2):
    """
    Calculate bond price using the yield to maturity
    """
    coupon_payment = (coupon_rate / frequency) * NOTIONAL
    periods = years_to_maturity * frequency
    period_yield = yield_rate / frequency

    if period_yield == 0:
        return coupon_payment * periods + NOTIONAL

    coupon_pv = coupon_payment * (1 - (1 + period_yield) ** -periods) / period_yield
    face_value_pv = NOTIONAL / (1 + period_yield) ** periods

    return coupon_pv + face_value_pv

# Load the data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean the 30-year yield data
def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

data = clean_numeric_data(data, ['30Y_treasury_yield'])

# Convert yield to decimal
data['30Y_treasury_yield'] = data['30Y_treasury_yield'] / 100

# Initialize bond pricing columns
data['rev_bond_price'] = np.nan
data['days_to_maturity'] = 0
data['roll_date'] = False

# Initialize maturity tracker
current_maturity = ISSUE_DATE + pd.DateOffset(years=30)
for i in range(len(data)):
    current_date = data.index[i]
    
    # Only evaluate between issue and final maturity
    if current_date < ISSUE_DATE or current_date > FINAL_MATURITY:
        continue

    # Roll bond if matured (simulate reinvestment in a new 30Y bond)
    if current_date >= current_maturity:
        data.loc[current_date, 'roll_date'] = True
        current_maturity = current_date + pd.DateOffset(years=30)
    
    # Days to maturity
    days_to_maturity = (current_maturity - current_date).days
    data.loc[current_date, 'days_to_maturity'] = days_to_maturity

    time_to_maturity = days_to_maturity / 365.0

    treasury_yield = data.loc[current_date, '30Y_treasury_yield']
    if not np.isnan(treasury_yield):
        effective_yield = treasury_yield + SPREAD
        price = calculate_bond_price(
            effective_yield,
            COUPON_RATE,
            time_to_maturity,
            FREQUENCY
        )
        data.loc[current_date, 'rev_bond_price'] = price

# Returns
data['rev_daily_returns'] = data['rev_bond_price'].pct_change()
data['rev_log_returns'] = np.log(data['rev_bond_price'] / data['rev_bond_price'].shift(1))

# Adjust for roll yield
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        roll_yield = (data['rev_bond_price'].iloc[i] - data['rev_bond_price'].iloc[i-1]) / data['rev_bond_price'].iloc[i-1]
        data.loc[data.index[i], 'rev_daily_returns'] = roll_yield

# Compute NAV
nav = pd.Series(index=data.index, name='Revenue_Bond_NAV')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    if not np.isnan(data['rev_daily_returns'].iloc[i]):
        nav.iloc[i] = nav.iloc[i-1] * (1 + data['rev_daily_returns'].iloc[i])
    else:
        nav.iloc[i] = nav.iloc[i-1]

# Build output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_30Y_Revenue_Bond': nav.values,
    'Log_Return_Rev_Bond': data['rev_log_returns'].values,
    'Bond_Price': data['rev_bond_price'].values,
    'Days_to_Maturity': data['days_to_maturity'].values,
    'Roll_Date': data['roll_date'].values,
    'Effective_Yield (%)': (data['30Y_treasury_yield'] + SPREAD) * 100
})

# Save output
output_df.to_csv('30_year_revenue_bond_data.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='30-Year AA+ Revenue Bond', linewidth=2)
roll_dates = nav.index[data['roll_date']]
plt.scatter(roll_dates, nav[roll_dates], color='red', label='Roll Dates', zorder=5)
plt.title('NAV of 30-Year AA+ Revenue Bond Over Time (Starting at 100)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', ls='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale('log')
date_form = DateFormatter("%Y-%m")
plt.gca().xaxis.set_major_formatter(date_form)
plt.savefig('30_year_revenue_bond_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check the CSV file and NAV visualization.")
