import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Constants
NOTIONAL = 100
MATURITY_YEARS = 5
ROLL_FREQUENCY_DAYS = 365  # roll annually
RECOVERY_RATE = 0.4
PAYMENTS_PER_YEAR = 4
DAY_COUNT = 365
TRANSACTION_COST = 0.002  # 0.2%

# Load Data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean numeric columns
cols = ['5_Y_ford_credit_spread', '5y_treasury_yield']
data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

# Convert from %/bps to decimals
data['5_Y_ford_credit_spread'] /= 10000  # bps to decimal
data['5y_treasury_yield'] /= 100  # % to decimal

# Filter date range
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize columns
data['roll_date'] = False
data['cds_value'] = np.nan
data['daily_return'] = np.nan
data['log_return'] = np.nan

# Rolling CDS strategy
start_date = data.index[0]
current_roll_date = start_date
next_roll_date = current_roll_date + pd.Timedelta(days=ROLL_FREQUENCY_DAYS)

# Function to compute CDS MTM value
def cds_value(spread, hazard_rate, r, maturity):
    dt = 1 / PAYMENTS_PER_YEAR
    n = int(maturity * PAYMENTS_PER_YEAR)
    discount_factors = np.exp(-r * dt * np.arange(1, n + 1))
    survival_probs = np.exp(-hazard_rate * dt * np.arange(1, n + 1))
    
    premium_leg = spread * np.sum(discount_factors * survival_probs) * NOTIONAL * dt
    protection_leg = (1 - RECOVERY_RATE) * np.sum(discount_factors * np.diff(np.insert(survival_probs, 0, 1))) * NOTIONAL
    
    return premium_leg - protection_leg

# Initialize NAV series
nav = pd.Series(index=data.index, name='NAV_5Y_CDS_Ford')
nav.iloc[0] = NOTIONAL

# Loop through data
for i in range(1, len(data)):
    current_date = data.index[i]
    previous_date = data.index[i - 1]

    if current_date >= next_roll_date:
        data.loc[current_date, 'roll_date'] = True
        current_roll_date = current_date
        next_roll_date = current_roll_date + pd.Timedelta(days=ROLL_FREQUENCY_DAYS)

    spread = data.loc[current_date, '5_Y_ford_credit_spread']
    r = data.loc[current_date, '5y_treasury_yield']
    
    if not np.isnan(spread) and not np.isnan(r):
        hazard_rate = spread / (1 - RECOVERY_RATE)
        value = cds_value(spread, hazard_rate, r, MATURITY_YEARS)
        data.loc[current_date, 'cds_value'] = value

        prev_value = data.loc[previous_date, 'cds_value']
        if not np.isnan(prev_value) and prev_value > 0:
            gross_return = (value - prev_value) / prev_value
            if data.loc[current_date, 'roll_date']:
                gross_return -= TRANSACTION_COST
            data.loc[current_date, 'daily_return'] = gross_return
            data.loc[current_date, 'log_return'] = np.log(value / prev_value)

# NAV Computation
for i in range(1, len(nav)):
    prev = nav.iloc[i - 1]
    ret = data['daily_return'].iloc[i]
    nav.iloc[i] = prev * (1 + ret) if not np.isnan(ret) else prev

# Output DataFrame
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_5Y_CDS_Ford': nav.values,
    'Log_Return': data['log_return'].values,
    'CDS_Value': data['cds_value'].values,
    'Roll_Date': data['roll_date'].values,
    'Ford_CDS_Spread_bps': data['5_Y_ford_credit_spread'].values * 10000,
    '5Y_Treasury_Yield_%': data['5y_treasury_yield'].values * 100
})

# Save to CSV
output_df.to_csv('ford_5y_cds_rolling.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='Rolling 5Y Ford CDS Seller', linewidth=2)
plt.title('NAV of Rolling 5Y CDS on Ford (Seller Position)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV (log scale)', fontsize=14)
plt.yscale('log')
plt.grid(True, which='both', ls='--')
plt.legend(fontsize=12)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('ford_5y_cds_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Number of roll-overs: {data['roll_date'].sum()}")
print("Check 'ford_5y_cds_rolling.csv' and NAV plot.")
print(f"Transaction cost of {TRANSACTION_COST*100}% applied at each roll date.")
