import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from scipy.stats import kendalltau, rankdata, t as student_t, binom
from copulas.bivariate.gumbel import Gumbel
from copulas.bivariate.clayton import Clayton
from tabulate import tabulate
import openturns as ot
from itertools import combinations

# Data source URL
url = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_nav_history.csv"

# Key parameters
notional = 10_000_000
confidence_level = 0.99
vol_cap = 0.3

# Portfolio allocation across assets
allocation = {
    'Apple': 0.02, 'Lockheed_martin': 0.02, 'Nvidia': 0.02, 'Procter_gamble': 0.02,
    'Johnson_johnson': 0.02, 'Toyota': 0.02, 'Nestle': 0.02, 'X_steel': 0.02,
    '10y_treasury': 0.05, 'Lqd_etf': 0.04, '10y_tips': 0.05, '1y_eur_zcb': 0.05,
    'High_yield_corp_debt': 0.04, '5y_green_bond': 0.03, '30y_revenue_bond': 0.04,
    'Sp500_futures_1m': 0.04, 'Vix_futures': 0.03, 'Crude_oil_futures': 0.03,
    'Gold_futures': 0.04, 'Soybean_futures': 0.03, 'Costco_itm_call': 0.03,
    'Xom_itm_put': 0.03, 'Eurusd_atm_call': 0.03, 'Usdjpy_atm_put': 0.03,
    'Gbpusd_6m_forward': 0.04, 'Usdinr_3m_forward': 0.04, 'Ford_cds': 0.03,
    'Dax_variance_swap': 0.03, 'Nikkei_asian_put': 0.03, 'Spx_knockout_call': 0.03,
    'Cash': 0.05
}

warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
if 'NAV' in df.columns:
    df.drop(columns=['NAV'], inplace=True)

# Extract NAV columns for assets in our allocation
nav_columns = [f"{k}_NAV" for k in allocation if f"{k}_NAV" in df.columns]
df_nav = df[nav_columns]
portfolio_nav = df_nav.sum(axis=1)

# Plot portfolio NAV over time
plt.figure(figsize=(10, 5))
plt.plot(portfolio_nav, label='Portfolio NAV (Initial Allocation)')
plt.yscale('log')
plt.title('Portfolio NAV (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Portfolio NAV (log scale)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate log returns
log_returns = np.log(df_nav / df_nav.shift(1)).dropna()
asset_log_returns = log_returns.copy()
adf_results = {}

# Test stationarity of each asset's returns
for asset in asset_log_returns.columns:
    series = asset_log_returns[asset].dropna()
    result = adfuller(series)
    adf_results[asset] = {'ADF Statistic': result[0], 'p-value': result[1]}

adf_df = pd.DataFrame(adf_results).T
adf_df['Stationary (p < 0.05)'] = adf_df['p-value'] < 0.05
stationary_assets = adf_df[adf_df['Stationary (p < 0.05)']].index.tolist()
asset_log_returns = asset_log_returns[stationary_assets]

# Fit ARMA models to stationary asset returns
arma_models = {}
residuals_dict = {}

for asset in stationary_assets:
    print(f"\n=== {asset} ===")
    
    series = asset_log_returns[asset].dropna()
    
    # Auto-select best ARMA model
    model = auto_arima(series,
                       start_p=0, start_q=0,
                       max_p=2, max_q=2,
                       seasonal=False,
                       d=0,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True,
                       information_criterion='aicc')
    
    arma_models[asset] = model
    residuals = pd.Series(model.resid(), index=series.index)
    residuals_dict[asset] = residuals
    
    # Test for autocorrelation in residuals
    lb_test = sm.stats.acorr_ljungbox(residuals, lags=[14], return_df=True)
    lb_stat = lb_test.iloc[0, 0]
    lb_pval = lb_test.iloc[0, 1]
    
    print(f"Ljung-Box Q(14) = {lb_stat:.2f}, p-value = {lb_pval:.4f}")
    
    if lb_pval > 0.05:
        print("Residuals appear to be white noise.")
    else:
        print("Residuals show autocorrelation -> ARMA model may be insufficient.")

# Identify assets needing GARCH modeling
garch_candidates = []
arch_test_results = []

for asset in residuals_dict:
    residuals = residuals_dict[asset].dropna()
    squared_residuals = residuals ** 2

    # Test for ARCH effects in residuals
    lb_sq_test = sm.stats.acorr_ljungbox(squared_residuals, lags=[14], return_df=True)
    lb_sq_stat = lb_sq_test.iloc[0, 0]
    lb_sq_pval = lb_sq_test.iloc[0, 1]

    print(f"\n=== {asset} ===")
    print(f"Ljung-Box on squared residuals Q(14) = {lb_sq_stat:.2f}, p-value = {lb_sq_pval:.4f}")

    if lb_sq_pval < 0.05:
        print("Conditional heteroscedasticity detected -> GARCH(1,1) recommended.")
        garch_candidates.append(asset)
    else:
        print("No significant ARCH effect in residuals.")

    arch_test_results.append({
        'Asset': asset,
        'Ljung-Box Q(14)': round(lb_sq_stat, 2),
        'p-value': round(lb_sq_pval, 4),
        'ARCH Effect Detected': lb_sq_pval < 0.05
    })

arch_test_df = pd.DataFrame(arch_test_results).set_index('Asset')

# Fit GARCH models to candidate assets
garch_models = {}
garch_vols = {}
garch_summary = []

for asset in garch_candidates:
    print(f"\n=== GARCH(1,1) Fit for {asset} ===")
    series = asset_log_returns[asset].dropna()

    # Fit GARCH(1,1) model with t distribution
    model = arch_model(series, vol='Garch', p=1, q=1, mean='Zero', dist='t')
    result = model.fit(disp='off')

    garch_models[asset] = result
    garch_vols[asset] = result.conditional_volatility

    garch_summary.append({
        'Asset': asset,
        'AIC': round(result.aic, 4),
        'BIC': round(result.bic, 4),
        'Log-Likelihood': round(result.loglikelihood, 2)
    })

garch_summary_df = pd.DataFrame(garch_summary).sort_values('AIC')
print(tabulate(garch_summary_df, headers='keys', tablefmt='pretty'))

# Function to compute correlation matrices based on copula method
def compute_copula_corrs_fast(rets_window, method="t"):
    tau_matrix = rets_window.corr(method='kendall').values

    if method in ["t", "gaussian"]:
        return np.sin(np.pi * tau_matrix / 2)
    elif method == "clayton":
        with np.errstate(divide='ignore', invalid='ignore'):
            theta = 2 * tau_matrix / (1 - tau_matrix)
            theta = np.nan_to_num(theta, nan=0.0)
        return np.clip(theta / (theta + 2), -1, 1)
    elif method == "gumbel":
        return fit_gumbel_corr_matrix(rets_window)  
    else:
        raise ValueError(f"Unsupported copula method: {method}")

# Helper function for Gumbel copula fitting
def fit_gumbel_corr_matrix(rets_window):
    assets = rets_window.columns
    rho_matrix = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            x = rets_window.iloc[:, i].values
            y = rets_window.iloc[:, j].values
            u = rankdata(x) / (len(x) + 1)
            v = rankdata(y) / (len(y) + 1)

            try:
                cop = Gumbel()
                cop.fit(np.column_stack([u, v]))
                tau = cop.kendall_tau
                rho = np.sin(np.pi * tau / 2)
            except Exception:
                rho = 0.0

            rho_matrix.iloc[i, j] = rho
            rho_matrix.iloc[j, i] = rho

    return rho_matrix.values

# Calculate portfolio volatility using copula-based correlation matrices
def copula_sigma_fast(cond_sigmas, rets, weights, method="t", update_freq=20):
    asset_names = cond_sigmas.columns
    all_dates = cond_sigmas.index

    # Calculate correlation matrices at regular intervals
    rho_dict = {}
    for idx in range(0, len(all_dates), update_freq):
        date_t = all_dates[idx]
        try:
            end_idx = rets.index.get_loc(date_t)
        except KeyError:
            continue
        start_idx = max(0, end_idx - 252)  # Use one year of data
        rets_window = rets.iloc[start_idx:end_idx][asset_names].dropna()
        if rets_window.shape[0] < 30:
            continue
        rho = compute_copula_corrs_fast(rets_window, method=method)
        rho_dict[date_t] = rho

    # Fill in correlation matrices for all dates
    rho_dates = sorted(rho_dict.keys())
    rho_filled = {}
    for i in range(len(rho_dates) - 1):
        start, end = rho_dates[i], rho_dates[i + 1]
        fill_dates = all_dates[(all_dates >= start) & (all_dates < end)]
        for d in fill_dates:
            rho_filled[d] = rho_dict[start]
    for d in all_dates[all_dates >= rho_dates[-1]]:
        rho_filled[d] = rho_dict[rho_dates[-1]]

    # Calculate portfolio volatility for each date
    sigmas_mat = cond_sigmas[asset_names].values
    sigmas = []

    for t in all_dates:
        if t not in rho_filled:
            sigmas.append(np.nan)
            continue

        rho = rho_filled[t]
        sigma_t = cond_sigmas.loc[t].values

        if rho.shape != (len(sigma_t), len(sigma_t)):
            sigmas.append(np.nan)
            continue

        cov = np.diag(sigma_t) @ rho @ np.diag(sigma_t)
        port_sigma = np.sqrt(weights @ cov @ weights)
        sigmas.append(port_sigma)

    return pd.Series(sigmas, index=all_dates)

# Set up assets and weights for portfolio modeling
rets = asset_log_returns.copy()
initial_nav = df.iloc[0][asset_log_returns.columns]
weights = (initial_nav / initial_nav.sum()).values
cond_sigmas = pd.DataFrame(index=rets.index)

# Get conditional volatilities for each asset
for asset in rets.columns:
    if asset in garch_vols:
        cond_sigmas[asset] = garch_vols[asset]
    else:
        cond_sigmas[asset] = rets[asset].rolling(window=250).std()
    cond_sigmas[asset] = cond_sigmas[asset].clip(upper=vol_cap)

# Calculate portfolio volatility using different copula methods
copula_sigma_results = {
    "t": copula_sigma_fast(cond_sigmas, rets, weights, method="t", update_freq=20),
    "Gaussian": copula_sigma_fast(cond_sigmas, rets, weights, method="gaussian", update_freq=20),
    "Clayton": copula_sigma_fast(cond_sigmas, rets, weights, method="clayton", update_freq=20),
    "Gumbel": copula_sigma_fast(cond_sigmas, rets, weights, method="gumbel", update_freq=20)
}

# Plot portfolio volatility for each copula type
for name, sigma_series in copula_sigma_results.items():
    plt.figure(figsize=(12, 4))
    plt.plot(sigma_series.dropna(), label=f"{name} Copula")
    plt.title(f"Portfolio Conditional Sigma via {name} Copula")
    plt.ylabel("Portfolio Sigma")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Calculate portfolio PnL
pnl_actual = np.log(portfolio_nav / portfolio_nav.shift(1)).dropna() * notional

# Find common dates across all models
common_idx_all = set(pnl_actual.index)
for sigma_series in copula_sigma_results.values():
    common_idx_all = common_idx_all.intersection(set(sigma_series.index))
common_idx_all = sorted(common_idx_all)

# Optimize degrees of freedom for Student-t distribution
sigma_ref = copula_sigma_results["t"].loc[common_idx_all]
pnl_ref = pnl_actual.loc[common_idx_all]
dof_range = range(3, 50)
backtest_summary = []

for dof in dof_range:
    z_val = student_t.ppf(confidence_level, df=dof)
    VaR_test = z_val * sigma_ref
    exceptions = (-pnl_ref > VaR_test * notional)
    total_days = exceptions.count()
    observed = exceptions.sum()
    expected = int((1 - confidence_level) * total_days)
    ratio = observed / expected if expected > 0 else float('inf')
    backtest_summary.append({
        'DoF': dof,
        'Observed': int(observed),
        'Expected': expected,
        'Ratio': round(ratio, 2),
        'Distance': abs(ratio - 1.0),
        'Mean VaR': int(VaR_test.mean() * notional)
    })

bt_df = pd.DataFrame(backtest_summary).sort_values('Distance')
optimal_dof = int(bt_df.iloc[0]['DoF'])
z_star = student_t.ppf(confidence_level, df=optimal_dof)

print("\n=== Optimized Student-t DoF (based on t-copula) ===")
print(tabulate(bt_df.head(10), headers='keys', tablefmt='pretty', showindex=False))
print(f"\nOptimal DoF: {optimal_dof}, z = {z_star:.4f}")

# Compare VaR backtest results across different copula methods
multi_var_summary = []
for copula_name, sigma_series in copula_sigma_results.items():
    sigma_aligned = sigma_series.loc[common_idx_all]
    pnl_aligned = pnl_actual.loc[common_idx_all]

    VaR_series = z_star * sigma_aligned
    exceptions = (-pnl_aligned > VaR_series * notional)

    summary = {
        'Copula': copula_name,
        'Total Days': len(VaR_series),
        'Observed Exc.': int(exceptions.sum()),
        'Expected Exc.': int((1 - confidence_level) * len(VaR_series)),
        'Mean VaR': int(VaR_series.mean() * notional),
        'Max VaR': int(VaR_series.max() * notional),
        'Min VaR': int(VaR_series.min() * notional)
    }
    multi_var_summary.append(summary)

print("\n=== Multi-Copula VaR Backtest Summary (Shared DoF) ===")
print(tabulate(multi_var_summary, headers='keys', tablefmt='pretty'))

# Plot VaR vs actual PnL for each copula type
for copula_name, sigma_series in copula_sigma_results.items():
    sigma_aligned = sigma_series.loc[common_idx_all]
    pnl_aligned = pnl_actual.loc[common_idx_all]
    VaR_series = z_star * sigma_aligned

    plt.figure(figsize=(12, 5))
    plt.plot(pnl_aligned, label='Actual PnL')
    plt.plot(VaR_series * notional, label=f'{copula_name} Copula VaR', color='red')
    plt.title(f'VaR vs Actual PnL ({copula_name} Copula, df={optimal_dof})')
    plt.xlabel('Date')
    plt.ylabel('EUR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Prepare data for further analysis
copula_sigma_series = {
    name: pd.Series(values, index=cond_sigmas.index)
    for name, values in copula_sigma_results.items()
}

pnl_actual_aligned = pnl_actual.loc[common_idx_all]

# Calculate VaR statistics for each copula type
copula_var_dict = {}
for name, sigma_series in copula_sigma_results.items():
    sigma_aligned = sigma_series.loc[common_idx_all]
    VaR = z_star * sigma_aligned
    copula_var_dict[name] = {
        'mean': VaR.mean() * notional,
        'max': VaR.max() * notional,
        'min': VaR.min() * notional
    }

# Calculate empirical VaR for comparison
empirical_var = np.percentile(-pnl_actual_aligned, 1)
x_min = np.percentile(-pnl_actual_aligned, 0.5)
x_max = np.percentile(-pnl_actual_aligned, 99.5)

# Plot histogram of losses with VaR estimates
plt.figure(figsize=(12, 6))
plt.hist(-pnl_actual_aligned, bins=50, color='skyblue', edgecolor='black', alpha=0.6, label='Actual Losses')
plt.axvline(empirical_var, color='red', linestyle='--', linewidth=2,
            label=f'Empirical 99% VaR: €{abs(empirical_var):,.0f}')

colors = ['green', 'orange', 'blue', 'purple']
for (copula, stats), color in zip(copula_var_dict.items(), colors):
    plt.axvline(-stats['mean'], color=color, linestyle='--', linewidth=2,
                label=f'{copula} Copula Mean VaR: €{int(stats["mean"]):,}')

plt.xlim(x_min, x_max)
plt.title("Histogram of Actual Losses with Copula-based VaR Levels")
plt.xlabel("Loss (EUR)")
plt.ylabel("Frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Binomial test for VaR model validation
total_days = len(common_idx_all)
expected = int((1 - confidence_level) * total_days)
std_dev = np.sqrt(total_days * (1 - confidence_level) * confidence_level)
x_min = max(0, int(expected - 4 * std_dev))
x_max = int(expected + 4 * std_dev)
x_vals = np.arange(x_min, x_max)
binom_probs = binom.pmf(x_vals, total_days, 1 - confidence_level)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, binom_probs, label='Expected Binomial Distribution', color='blue')

for name, sigma in copula_sigma_series.items():
    sigma_aligned = sigma.loc[common_idx_all]
    VaR_series = z_star * sigma_aligned
    pnl_aligned = pnl_actual.loc[common_idx_all]
    exceptions = (-pnl_aligned > VaR_series * notional)
    obs = int(exceptions.sum())
    plt.axvline(obs, linestyle='--', linewidth=2, label=f'{name}: {obs} Exceptions')

plt.axvline(expected, color='green', linestyle='-', label='Expected Exceptions')
plt.title(f'Binomial Test of 99% 1-Day VaR Violations\n(Total Days: {total_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert data to uniform scale for copula fitting
def to_uniform(data):
    return np.array([
        pd.Series(col).rank(method="average") / (len(col) + 1)
        for col in data.T
    ]).T

# Get number of parameters in copula model
def get_num_params(copula):
    try:
        return len(copula.getParameter())
    except Exception:
        return 1

# Fit a specific copula type and calculate information criteria
def fit_and_score_copula(name, sample):
    copula_factories = {
        'Gaussian': ot.NormalCopulaFactory(),
        't': ot.StudentCopulaFactory(),
        'Clayton': ot.ClaytonCopulaFactory()
    }
    try:
        factory = copula_factories[name]
        copula = factory.build(sample)
        log_likelihood = np.array(copula.computeLogPDF(sample)).sum()
        n = sample.getSize()
        k = get_num_params(copula)
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        return log_likelihood, aic, bic
    except Exception as e:
        print(f"Copula {name} fitting failed: {e}")
        return np.nan, np.nan, np.nan

# Evaluate different copula types using AIC/BIC
def evaluate_copulas_aicbic(data):
    copula_types = ['Gaussian', 't', 'Clayton']
    results = []

    for i, j in combinations(range(data.shape[1]), 2):
        pair_data = data[:, [i, j]]
        u_data = to_uniform(pair_data)
        ot_sample = ot.Sample(u_data.tolist())

        for cop in copula_types:
            ll, aic, bic = fit_and_score_copula(cop, ot_sample)
            results.append({
                'Pair': f'{i}-{j}',
                'Copula': cop,
                'LogLikelihood': ll,
                'AIC': aic,
                'BIC': bic
            })

    return pd.DataFrame(results)

# Compare copula types using AIC/BIC
returns_df = asset_log_returns[garch_vols.keys()].dropna().values
copula_results = evaluate_copulas_aicbic(returns_df)
copula_results = copula_results.replace([-np.inf, np.inf], np.nan)
gof_summary = copula_results.groupby('Copula')[['LogLikelihood', 'AIC', 'BIC']].mean().sort_values('AIC')

print("\n=== Copula Goodness-of-Fit (Average Log-Likelihood / AIC / BIC) ===")
print(gof_summary)

# Plot AIC/BIC comparison
plt.figure(figsize=(10, 5))
gof_summary[['AIC', 'BIC']].plot(kind='bar')
plt.title("Average AIC and BIC by Copula Type")
plt.ylabel("Information Criterion")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show() 