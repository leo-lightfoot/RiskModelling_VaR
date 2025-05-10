import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import argparse

# Suppress all warnings
warnings.filterwarnings('ignore')

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
    #var_hist_sim = np.zeros(n - window_length)
    
    # For storing GARCH model residuals
    all_residuals = []
    
    # Rolling window estimation
    for t in range(window_length, n):
        # Extract window data
        window_data = returns[t-window_length:t].copy()
        
        # Fit GARCH model
        model = arch_model(window_data, vol='GARCH', p=1, q=1, dist='studentsT')
        model_fit = model.fit(disp='off')
        
        # Store model results
        residuals = model_fit.resid
        all_residuals.append(residuals)
        
        # Get forecast for next period
        forecast = model_fit.forecast(horizon=1)
        conditional_vol = np.sqrt(forecast.variance.iloc[-1, 0])
        
        # Historical simulation VaR
        #var_hist_sim[t-window_length] = np.percentile(window_data, alpha*100)
        
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
    #violations_hist_sim = np.sum(actual_returns < var_hist_sim)
    
    # Calculate binomial test bounds
    lower_bound = stats.binom.ppf(0.025, len(actual_returns), alpha)
    upper_bound = stats.binom.ppf(0.975, len(actual_returns), alpha)
    
    # Store results
    results = {
        'violations': {
            'GARCH+EVT': violations_garch_evt,
            'GARCH+t': violations_garch_t,
           # 'HistSim': violations_hist_sim
        },
        'bounds': {
            'lower': lower_bound,
            'upper': upper_bound
        },
        'var_estimates': {
            'GARCH+EVT': var_garch_evt,
            'GARCH+t': var_garch_t,
            #'HistSim': var_hist_sim
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
    print(f"Expected violations at {alpha*100:.2f}% VaR: {alpha * n_returns:.1f}")
    print(f"Acceptance region: [{bounds['lower']:.1f}, {bounds['upper']:.1f}]")
    print("\nNumber of violations:")
    
    for method, count in violations.items():
        status = "ACCEPTED" if bounds['lower'] <= count <= bounds['upper'] else "REJECTED"
        print(f"  {method}: {count} ({count/n_returns*100:.2f}%) - {status}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Hardcoded parameters
    file_path = "Portfolio/portfolio_results/portfolio_returns_history.csv"
    column_name = "Return"
    window_lengths = [500]
    alpha = 0.01
    
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Select column with returns
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV file")
    returns = df[column_name].values
    
    # Run analysis for each window length
    for window_length in window_lengths:
        print(f"\nCalculating VaR with window length = {window_length}, alpha = {alpha}...")
        results = calculate_var(returns, window_length, alpha)
        
        # Print and plot results
        print_results(results)
        plot_results(results) 