import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load return data
returns = pd.read_csv(r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_returns_history.csv", index_col=0, parse_dates=True)
returns = returns['Return']

# Crisis periods - using the same ones as in historical_stress_test.py
crisis_periods = {
    '2008 Financial Crisis': ('2008-09-01', '2009-03-31'),
    '2012 Euro Crisis': ('2011-07-01', '2012-07-31'),
    '2018 Tariff War': ('2018-01-26', '2018-12-24'),
    'COVID-19 Pandemic': ('2020-02-15', '2020-04-30'),
    'Russia-Ukraine War': ('2022-02-24', '2022-06-30')
}

confidence_level = 0.99
uniform_period_length = 30

# Store results
var_comparison = []

# Plot individual crisis plots
for crisis_name, (start_date, end_date) in crisis_periods.items():
    crisis_data = returns[start_date:end_date]

    if len(crisis_data) < uniform_period_length:
        var_comparison.append((crisis_name, 'Insufficient Data', 'Insufficient Data'))
        continue

    # Full-period 1-day Historical VaR
    full_var = np.percentile(crisis_data, (1 - confidence_level) * 100)

    # Rolling 30-day VaR
    rolling_var = crisis_data.rolling(window=uniform_period_length).apply(
        lambda x: np.percentile(x, (1 - confidence_level) * 100), raw=True
    )

    # Identify the window with the worst (lowest) 1-day VaR
    max_rolling_var = rolling_var.min()
    worst_var_date = rolling_var.idxmin()
    worst_var_window_start = worst_var_date - pd.Timedelta(days=uniform_period_length - 1)
    worst_var_window_end = worst_var_date

    var_comparison.append((crisis_name, -full_var, -max_rolling_var))

    # Trim the date range to include 30 days before and after the crisis period for clarity
    start_plot_date = pd.to_datetime(start_date) - pd.Timedelta(days=30)
    end_plot_date = pd.to_datetime(end_date) + pd.Timedelta(days=30)

    # Plot returns with highlights
    plt.figure(figsize=(12, 5))
    plt.plot(returns.index, returns.cumsum(), label='Cumulative Return', color='gray')

    # Highlight full crisis period
    plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='orange', alpha=0.2, label=f'{crisis_name} Crisis Period')

    # Highlight worst 30-day window
    plt.axvspan(worst_var_window_start, worst_var_window_end, color='blue', alpha=0.3, label=f'{crisis_name} Max 30-Day Stress Window')

    # Mark the worst VaR day
    plt.axvline(worst_var_date, color='red', linestyle='--', label=f'{crisis_name} Worst VaR Day')
    plt.scatter([worst_var_date], [returns.loc[worst_var_date:].cumsum().iloc[0]], color='red', zorder=5)

    # Set the plot range
    plt.xlim(start_plot_date, end_plot_date)

    plt.title(f"{crisis_name}: Crisis Period, Max-Stress Window, and Worst VaR Day")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Combined plot of all crises over the full data period
plt.figure(figsize=(14, 8))
plt.plot(returns.index, returns.cumsum(), label='Cumulative Return', color='gray')

# Loop through each crisis and plot over the full data horizon
for crisis_name, (start_date, end_date) in crisis_periods.items():
    crisis_data = returns[start_date:end_date]
    
    if len(crisis_data) < uniform_period_length:
        continue

    # Full-period 1-day Historical VaR
    full_var = np.percentile(crisis_data, (1 - confidence_level) * 100)

    # Rolling 30-day VaR
    rolling_var = crisis_data.rolling(window=uniform_period_length).apply(
        lambda x: np.percentile(x, (1 - confidence_level) * 100), raw=True
    )

    # Identify the window with the worst (lowest) 1-day VaR
    max_rolling_var = rolling_var.min()
    worst_var_date = rolling_var.idxmin()
    worst_var_window_start = worst_var_date - pd.Timedelta(days=uniform_period_length - 1)
    worst_var_window_end = worst_var_date

    # Highlight full crisis period
    plt.axvspan(pd.to_datetime(start_date), pd.to_datetime(end_date), color='orange', alpha=0.2, label=f'{crisis_name} Crisis Period')

    # Highlight worst 30-day window
    plt.axvspan(worst_var_window_start, worst_var_window_end, color='blue', alpha=0.3, label=f'{crisis_name} Max 30-Day Stress Window')

    # Mark the worst VaR day
    plt.axvline(worst_var_date, color='red', linestyle='--', label=f'{crisis_name} Worst VaR Day')
    plt.scatter([worst_var_date], [returns.loc[worst_var_date:].cumsum().iloc[0]], color='red', zorder=5)

# Add title and labels for the combined plot
plt.title("Cumulative Returns with Crisis Periods, Stress Windows, and Worst VaR Days (Combined)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(loc='best', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()

# Show the combined plot
plt.show()

# Print the comparison table of VaR results
print("99% 1-Day Historical VaR Comparison by Crisis Period:")
print(f"{'Crisis':30} {'Full-Period VaR':>20} {'Max 30-Day VaR':>20}")
print("-" * 75)
for crisis, full, max_30 in var_comparison:
    if isinstance(full, str):
        print(f"{crisis:30} {full:>20} {max_30:>20}")
    else:
        print(f"{crisis:30} {full:>19.2%} {max_30:>19.2%}")
