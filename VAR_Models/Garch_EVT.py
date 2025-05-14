import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import argparse

PORTFOLIO_RETURNS_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_returns_history.csv"
COLUMN_NAME = "Return"
WINDOW_LENGTHS = [500]
ALPHA = 0.01
NOTIONAL = 10000000

def load_data(url):
    """Simple function to load data with parse_dates"""
    return pd.read_csv(url, parse_dates=['Date'])

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Value at Risk using GARCH+EVT')
    return parser.parse_args()

def fit_gpd(exceedances, threshold):
    """
    Fit a Generalized Pareto Distribution to the exceedances
    """
    # Scale exceedances by 100 to avoid convergence issues (as in the R code)
    scaled_exceedances = 100 * (exceedances - threshold)
    
    def negative_log_likelihood(params):
        shape, scale = params
        n = len(scaled_exceedances)
        
        # Handle shape parameter close to zero separately
        if abs(shape) < 1e-6:
            return n * np.log(scale) + np.sum(scaled_exceedances) / scale
        
        # Check validity of parameters
        if scale <= 0 or (shape < 0 and max(scaled_exceedances) >= -scale/shape):
            return np.inf
        
        return n * np.log(scale) + (1 + 1/shape) * np.sum(np.log(1 + shape * scaled_exceedances / scale))
    
    # Initial parameters (shape, scale)
    initial_params = [0.1, np.std(scaled_exceedances)]
    
    # Optimize the negative log-likelihood
    result = minimize(negative_log_likelihood, initial_params, method='Nelder-Mead')
    
    return result.x[0], result.x[1]  # return shape and scale parameters

def calculate_var(returns, window_length=250, alpha=0.01):
    """
    Calculate VaR using ARMA+GARCH with EVT and compare with other methods
    """
    n = len(returns)
    results = {}
    
    # Ensure we have enough data points
    if n <= window_length:
        raise ValueError(f"Not enough data points. Need more than {window_length} observations.")
    
    # Initialize storage for VaR estimates
    var_garch_evt = np.zeros(n - window_length)
    var_garch_t = np.zeros(n - window_length)
    
    # For storing GARCH model residuals
    all_residuals = []
    
    # Rolling window estimation
    for t in range(window_length, n):
        # Extract window data
        window_data = returns[t-window_length:t].copy()
        
        # Try with different optimizers if one fails
        try:
            model = arch_model(window_data, vol='GARCH', p=1, q=1, dist='studentsT')
            model_fit = model.fit(disp='off', show_warning=False)
        except:
            try:
                # Try with a different optimizer
                model = arch_model(window_data, vol='GARCH', p=1, q=1, dist='studentsT')
                model_fit = model.fit(disp='off', options={'maxiter': 1000}, show_warning=False)
            except:
                # Fallback to normal distribution if t-dist fails
                model = arch_model(window_data, vol='GARCH', p=1, q=1, dist='normal')
                model_fit = model.fit(disp='off', show_warning=False)
        
        # Store model results
        residuals = model_fit.resid
        all_residuals.append(residuals)
        
        # Get forecast for next period
        forecast = model_fit.forecast(horizon=1)
        conditional_vol = np.sqrt(forecast.variance.iloc[-1, 0])
        
        
        # GARCH with Student's t VaR
        dof = model_fit.params['nu']  # degrees of freedom for Student's t
        t_quantile = stats.t.ppf(alpha, dof)
        var_garch_t[t-window_length] = model_fit.params['mu'] + conditional_vol * t_quantile
        
        # GARCH with EVT VaR
        # Set threshold at 95th percentile
        u = np.percentile(residuals, 95)
        exceedances = residuals[residuals > u]
        
        if len(exceedances) > 10:  # Ensure enough data for GPD fitting
            # Fit GPD to exceedances
            shape, scale = fit_gpd(exceedances, u)
            
            # Calculate VaR using GPD
            N_u = len(exceedances)
            N = len(residuals)
            gpd_var = (100*u + (scale/shape) * (((alpha/(N_u/N))**(-shape) - 1)))/100
            
            # Standardize to get quantile for GARCH model
            std_gpd_var = -(gpd_var - np.mean(residuals)) / np.std(residuals)
            
            # Calculate final VaR
            var_garch_evt[t-window_length] = model_fit.params['mu'] + conditional_vol * std_gpd_var
        else:
            # Fallback to Student's t if not enough exceedances
            var_garch_evt[t-window_length] = var_garch_t[t-window_length]
    
    # Calculate number of violations
    actual_returns = returns[window_length:]
    violations_garch_evt = np.sum(actual_returns < var_garch_evt)
    violations_garch_t = np.sum(actual_returns < var_garch_t)
    
    # Calculate binomial test bounds
    lower_bound = stats.binom.ppf(0.025, len(actual_returns), alpha)
    upper_bound = stats.binom.ppf(0.975, len(actual_returns), alpha)
    
    # Store results
    results = {
        'violations': {
            'GARCH+EVT': violations_garch_evt,
            'GARCH+t': violations_garch_t,
        },
        'bounds': {
            'lower': lower_bound,
            'upper': upper_bound
        },
        'var_estimates': {
            'GARCH+EVT': var_garch_evt,
            'GARCH+t': var_garch_t,
        },
        'actual_returns': actual_returns
    }
    
    return results

def sanitize_var_series(var_series, threshold=1e6):
    var_series = np.nan_to_num(var_series, nan=0.0, posinf=0.0, neginf=0.0)
    var_series[np.abs(var_series) > threshold] = 0.0
    return var_series

def plot_results(results):
    actual_returns = results['actual_returns']
    var_garch_evt = sanitize_var_series(results['var_estimates']['GARCH+EVT'])
    var_garch_t = sanitize_var_series(results['var_estimates']['GARCH+t'])

    x = np.arange(len(actual_returns))
    plt.figure(figsize=(14, 6))
    plt.plot(x, actual_returns, color='gray', label='Returns', linewidth=1)
    plt.plot(x, var_garch_evt, color='green', label='GARCH+EVT VaR', linewidth=1)
    plt.plot(x, var_garch_t, color='blue', label='GARCH+t VaR', linewidth=1)

    # Highlight exceptions
    evt_violations = actual_returns < var_garch_evt
    t_violations = actual_returns < var_garch_t
    plt.scatter(x[evt_violations], actual_returns[evt_violations], color='red', label='Violations EVT', s=10)
    plt.scatter(x[t_violations], actual_returns[t_violations], color='orange', label='Violations t', s=10)

    plt.title('Value at Risk (VaR) Estimates with Violations')
    plt.xlabel('Time')
    plt.ylabel('Returns / VaR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('var_comparison_clean.png', dpi=300)
    plt.show()

def print_results(results):
    violations = results['violations']
    bounds = results['bounds']
    n_returns = len(results['actual_returns'])
    
    print("\n==== VaR Backtest Results ====")
    print(f"Total observations: {n_returns}")
    print(f"Expected violations at {ALPHA*100:.2f}% VaR: {ALPHA * n_returns:.1f}")
    print(f"Acceptance region: [{bounds['lower']:.1f}, {bounds['upper']:.1f}]")
    print("\nNumber of violations:")
    
    for method, count in violations.items():
        status = "ACCEPTED" if bounds['lower'] <= count <= bounds['upper'] else "REJECTED"
        lr_stat, p_val = kupiec_pof_test(count, n_returns, ALPHA)
        print(f"  {method}: {count} ({count/n_returns*100:.2f}%) - {status}")
        print(f"     → Kupiec POF: LR = {lr_stat:.2f}, p = {p_val:.4f}")

def kupiec_pof_test(violations, total_obs, alpha):
    """
    Kupiec Proportion of Failures (POF) Test
    H0: The proportion of violations equals alpha
    """
    pi = violations / total_obs
    if pi == 0 or pi == 1:
        return np.nan, 1.0  # Avoid log(0) issues
    
    log_likelihood_null = total_obs * np.log(1 - alpha) + violations * np.log(alpha / (1 - alpha))
    log_likelihood_alt = (violations * np.log(pi) + 
                          (total_obs - violations) * np.log(1 - pi))
    LR_pof = -2 * (log_likelihood_null - log_likelihood_alt)
    p_value = 1 - stats.chi2.cdf(LR_pof, df=1)
    return LR_pof, p_value

# Binomial Test Plot
from scipy.stats import binom

def plot_binomial_test(results, alpha=0.01):
    actual = results['violations']['GARCH+EVT']
    total_days = len(results['actual_returns'])
    expected = int(alpha * total_days)

    std_dev = np.sqrt(total_days * alpha * (1 - alpha))
    x = np.arange(int(expected - 4*std_dev), int(expected + 4*std_dev))
    pmf = binom.pmf(x, total_days, alpha)

    plt.figure(figsize=(12, 6))
    plt.plot(x, pmf, 'b-', lw=2, label='Expected Binomial Distribution')
    plt.axvline(expected, color='green', linestyle='--', label=f'Expected: {expected}')
    plt.axvline(actual, color='red', label=f'Actual: {actual}')
    plt.title(f'Binomial Test of 99% 1-Day VaR Violations (Scaled to $10M)\n(Total Days: {total_days})')
    plt.xlabel('Number of Exceptions')
    plt.ylabel('Probability Mass')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("binomial_test_evt.png", dpi=300)
    plt.show()

def plot_evt_pnl_distribution(actual_returns, var_series, alpha=0.01, scale=1e7):
    scaled_returns = actual_returns * scale
    scaled_var = np.percentile(scaled_returns,1)

    plt.figure(figsize=(12, 6))
    plt.hist(scaled_returns, bins=60, color='skyblue', edgecolor='black')
    plt.axvline(scaled_var, color='red', linestyle='--', linewidth=2,
                label=f'Avg GARCH+EVT VaR @ 99% ($10M): USD {int(abs(scaled_var)):,}')
    plt.title('GARCH+EVT: Portfolio PnL Distribution (Scaled to $10M)')
    plt.xlabel('Profit and Loss (USD, $10M)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evt_pnl_distribution.png", dpi=300)
    plt.show() 

# warning suppression 
from arch.utility.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    args = parse_arguments()
    
    print(f"Loading data from {PORTFOLIO_RETURNS_URL}...")
    df = load_data(PORTFOLIO_RETURNS_URL)
    
    if COLUMN_NAME not in df.columns:
        raise ValueError(f"Column '{COLUMN_NAME}' not found in CSV file")
    returns = df[COLUMN_NAME].values
    
    for window_length in WINDOW_LENGTHS:
        print(f"\nCalculating VaR with window length = {window_length}, alpha = {ALPHA}...")
        results = calculate_var(returns, window_length, ALPHA)
        
        print_results(results)
        plot_results(results)
        plot_binomial_test(results, ALPHA)
        plot_evt_pnl_distribution(results['actual_returns'], results['var_estimates']['GARCH+EVT'], ALPHA)
       