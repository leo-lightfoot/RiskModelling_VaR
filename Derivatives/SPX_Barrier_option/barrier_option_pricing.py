import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime
from scipy.stats import norm

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')

# Constants
NOTIONAL = 100
MATURITY_DAYS = 30
ANNUAL_BASIS = 365.0
TRANSACTION_COST = 0.003  # 0.3%
BARRIER_MULTIPLIER = 1.1  # 110% knock-out level
MIN_OPTION_VALUE = 1e-6   # Minimum option value to prevent numerical issues

# Black-Scholes call pricing function
def black_scholes_call(S, K, T, r, sigma, q):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo_barrier_call(S, K, T, r, sigma, q, B, n_paths=10000, n_steps=21):
    dt = T / n_steps
    disc = np.exp(-r * T)
    S_paths = np.full((n_paths, n_steps + 1), S)
    barrier_breached = np.zeros(n_paths, dtype=bool)
    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        barrier_breached |= (S_paths[:, t] >= B)
    payoffs = np.where(barrier_breached, 0.0, np.maximum(S_paths[:, -1] - K, 0))
    return disc * np.mean(payoffs)

# === LOAD DATA ===
# Placeholder for CSV path — update this with actual file path
data = pd.read_csv(r"C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Clean and convert data
cols = ['sp500_index', 'SPX_Div_yield', 'vix_index_level', 'fed_funds_rate']
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data['SPX_Div_yield'] /= 100
data['fed_funds_rate'] /= 100
data['vix_index_level'] /= 100  # VIX in %

# Filter date range
data = data[(data.index >= '2005-01-01') & (data.index <= '2024-12-31')]

# Initialize new columns
data['option_price'] = np.nan
data['strike_price'] = np.nan
data['barrier_level'] = np.nan
data['days_to_expiry'] = 0
data['roll_date'] = False
data['knocked_out'] = False

# Rolling logic
start_date = data.index[0]
current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
current_strike = data.loc[start_date, 'sp500_index']
current_barrier = BARRIER_MULTIPLIER * current_strike

for i in range(len(data)):
    current_date = data.index[i]
    spot = data.loc[current_date, 'sp500_index']
    
    data.loc[current_date, 'strike_price'] = current_strike
    data.loc[current_date, 'barrier_level'] = current_barrier

    if current_date >= current_expiry:
        data.loc[current_date, 'roll_date'] = True
        current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
        current_strike = data.loc[current_date, 'sp500_index']
        current_barrier = BARRIER_MULTIPLIER * current_strike
        # Reset the knocked out flag when rolling to a new option
        data.loc[current_date, 'knocked_out'] = False

    days_to_expiry = (current_expiry - current_date).days
    T = days_to_expiry / ANNUAL_BASIS
    data.loc[current_date, 'days_to_expiry'] = days_to_expiry

    sigma = data.loc[current_date, 'vix_index_level']
    r = data.loc[current_date, 'fed_funds_rate']
    q = data.loc[current_date, 'SPX_Div_yield']
    K = data.loc[current_date, 'strike_price']
    B = data.loc[current_date, 'barrier_level']

    if spot >= B:
        data.loc[current_date, 'option_price'] = 0.0
        data.loc[current_date, 'knocked_out'] = True
    elif not np.isnan(spot) and not np.isnan(sigma) and not np.isnan(r) and not np.isnan(q):
        price = monte_carlo_barrier_call(spot, K, T, r, sigma, q, B)
        data.loc[current_date, 'option_price'] = price

    # Add a step to propagate knocked_out status for the same option until roll date
    if i > 0 and not data.loc[current_date, 'roll_date'] and data.loc[data.index[i-1], 'knocked_out']:
        data.loc[current_date, 'knocked_out'] = True
        data.loc[current_date, 'option_price'] = 0.0

# Calculate returns
data['daily_return'] = data['option_price'].pct_change()
data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))

# Adjust return on roll dates
for i in range(1, len(data)):
    if data['roll_date'].iloc[i]:
        prev = data['option_price'].iloc[i - 1]
        curr = data['option_price'].iloc[i]
        if prev > 0:
            data.loc[data.index[i], 'daily_return'] = (curr - prev) / prev - TRANSACTION_COST

# Compute NAV
nav = pd.Series(index=data.index, name='NAV_SPX_KnockOut_Call')
nav.iloc[0] = NOTIONAL
MIN_NAV = 1.0  # Set a minimum NAV to avoid extremely small values in log scale

for i in range(1, len(nav)):
    if data['roll_date'].iloc[i]:
        # Reset NAV to NOTIONAL on every roll date to avoid astronomic growth
        nav.iloc[i] = NOTIONAL
    else:
        ret = data['daily_return'].iloc[i]
        new_val = nav.iloc[i - 1] * (1 + ret) if not np.isnan(ret) else nav.iloc[i - 1]
        nav.iloc[i] = max(new_val, MIN_NAV)  # Apply minimum value to avoid extremely small numbers

# Normalize NAV to start at NOTIONAL
# nav = nav / nav.iloc[0] * NOTIONAL

# Export
output_df = pd.DataFrame({
    'Date': nav.index,
    'NAV_SPX_KnockOut': nav.values,
    'Option_Price': data['option_price'].values,
    'Log_Return': data['log_return'].values,
    'Strike_Price': data['strike_price'].values,
    'Barrier_Level': data['barrier_level'].values,
    'Spot': data['sp500_index'].values,
    'Roll_Date': data['roll_date'].values,
    'Knocked_Out': data['knocked_out'].values,
    'Fed_Funds_Rate': data['fed_funds_rate'].values * 100,
    'Dividend_Yield': data['SPX_Div_yield'].values * 100,
    'Implied_Volatility': data['vix_index_level'].values * 100
})

output_df.to_csv("spx_knockout_call_option.csv", index=False)

# Plot
plt.figure(figsize=(14, 8))
plt.plot(nav.index, nav, label='30D ATM KO Call on S&P 500', linewidth=2)
plt.title(f"NAV of 30D Rolling KO Call on S&P 500 (Barrier at {int(BARRIER_MULTIPLIER*100)}% of Spot)", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("NAV (log scale)", fontsize=14)
plt.yscale("log")
plt.ylim(MIN_NAV, nav.max() * 1.1)  # Set reasonable y-axis limits
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--')
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.savefig("spx_knockout_nav.png", bbox_inches="tight", dpi=300)
plt.close()

print(f"Analysis complete. Rollovers: {data['roll_date'].sum()}, Knock-Outs: {data['knocked_out'].sum()}")
