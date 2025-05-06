import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import timedelta

# Constants
NOTIONAL = 100
MATURITY_DAYS = 30
ANNUAL_BASIS = 365.0
TRANSACTION_COST = 0.002  # 0.2% transaction cost on roll

# Load data
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean and convert
cols = ['DAX_Call_ivol_30D', 'DAX_Put_ivol_30D']
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data[['DAX_Call_ivol_30D', 'DAX_Put_ivol_30D']] /= 100

# Filter date range
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize
data['fixed_variance'] = np.nan
data['fixed_vol'] = np.nan
data['roll_date'] = False
data['days_to_expiry'] = 0

# Rolling logic
start_date = data.index[0]
current_expiry = start_date + timedelta(days=MATURITY_DAYS)

for i in range(len(data)):
    current_date = data.index[i]
    
    # Roll check
    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = current_date + timedelta(days=MATURITY_DAYS)
    
    # Time to expiry
    days_to_expiry = (current_expiry - current_date).days
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry
    
    # Compute fixed leg (variance strike) as average squared vol
    call_ivol = data.loc[current_date, 'DAX_Call_ivol_30D']
    put_ivol = data.loc[current_date, 'DAX_Put_ivol_30D']
    
    if not np.isnan(call_ivol) and not np.isnan(put_ivol):
        avg_ivol = 0.5 * (call_ivol + put_ivol)
        fixed_var = avg_ivol ** 2
        data.loc[current_date, 'fixed_vol'] = avg_ivol
        data.loc[current_date, 'fixed_variance'] = fixed_var

# NAV Calculation — assume you are paid daily implied variance (fixed) as return proxy
data['daily_return'] = 0.0
data['log_return'] = 0.0

# Return logic: use fixed variance to generate synthetic return
# (in real variance swap you get realized - fixed, here we price fixed leg only)
data['daily_return'] = data['fixed_variance'] / ANNUAL_BASIS
data['log_return'] = np.log(1 + data['daily_return'])

# Apply transaction cost on roll
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        data.loc[data.index[i], 'daily_return'] -= TRANSACTION_COST

# NAV Computation
nav = pd.Series(index=data.index, name='NAV_DAX30D_VarianceSwap')
nav.iloc[0] = NOTIONAL
for i in range(1, len(nav)):
    nav.iloc[i] = nav.iloc[i - 1] * (1 + data['daily_return'].iloc[i])

# Output
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV': nav.values,
    'Fixed_Variance': data['fixed_variance'].values,
    'Fixed_IVOL': data['fixed_vol'].values,
    'Roll_Date': data['roll_date'].values,
    'Call_IVOL': data['DAX_Call_ivol_30D'].values * 100,
    'Put_IVOL': data['DAX_Put_ivol_30D'].values * 100
})
output_df.to_csv('dax_variance_swap_fixed_leg.csv', index=False)

# Plot NAV
plt.figure(figsize=(14, 7))
plt.plot(nav.index, nav, label='30D DAX Variance Swap (Fixed Leg)', linewidth=2)
plt.title('NAV of 30D DAX Variance Swap (Fixed Leg Only)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('NAV', fontsize=14)
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig('dax_variance_swap_nav.png', bbox_inches='tight', dpi=300)
plt.close()

print(f"Analysis complete. Rolls: {data['roll_date'].sum()} times.")
print("Check CSV and NAV plot.")
