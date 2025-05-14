import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

DATA_PATH = "https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/data_restructured.csv"
PORTFOLIO_RETURNS_PATH = "https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/portfolio_results/portfolio_returns_history.csv"
PORTFOLIO_RESULTS_DIR = "Stress_test"

CRISIS_PERIODS = {
    "2008_Financial_Crisis": ("2008-09-01", "2009-03-31"),
    "2012_Euro_Crisis": ("2011-07-01", "2012-07-31"),
    "2018_Tariff_War": ("2018-01-26", "2018-12-24"),
    "COVID_19_Pandemic": ("2020-02-15", "2020-04-30"),
    "Russia_Ukraine_War": ("2022-02-24", "2022-06-30")
}

from Pricing import Portfolio, load_data

def load_portfolio_returns():
    try:
        response = requests.get(PORTFOLIO_RETURNS_PATH)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            print(f"Failed to download portfolio returns: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error loading portfolio returns: {e}")
        return None

def load_risk_factors():
    try:
        response = requests.get(DATA_PATH)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
            data.set_index('date', inplace=True)
            return data
        else:
            print(f"Failed to download risk factors: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error loading risk factors: {e}")
        return None

def calculate_drawdowns(returns_series):
    cum_returns = (1 + returns_series).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1
    return drawdowns

def identify_crisis_periods(returns_df):
    returns_series = returns_df['Return']
    drawdowns = calculate_drawdowns(returns_series)
    
    crisis_details = {}
    for crisis_name, (start_date, end_date) in CRISIS_PERIODS.items():
        try:
            period_mask = (drawdowns.index >= start_date) & (drawdowns.index <= end_date)
            period_drawdowns = drawdowns.loc[period_mask]
            
            if not period_drawdowns.empty:
                worst_date = period_drawdowns.idxmin()
                worst_drawdown = period_drawdowns.min()
                
                recovery_mask = (drawdowns.index > worst_date) & (drawdowns > -0.01)
                recovery_dates = drawdowns.loc[recovery_mask].index
                recovery_date = recovery_dates[0] if not recovery_dates.empty else None
                
                crisis_details[crisis_name] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "worst_date": worst_date,
                    "worst_drawdown": worst_drawdown,
                    "recovery_date": recovery_date
                }
            else:
                print(f"No data found for {crisis_name} period")
        except Exception as e:
            print(f"Error analyzing {crisis_name}: {e}")
    
    return crisis_details

def analyze_risk_factor_impact(risk_factors_df, returns_df, crisis_details):
    results = {}
    
    for crisis_name, details in crisis_details.items():
        start_date = details["start_date"]
        end_date = details["end_date"]
        
        mask = (risk_factors_df.index >= start_date) & (risk_factors_df.index <= end_date)
        crisis_risk_factors = risk_factors_df.loc[mask]
        
        mask = (returns_df.index >= start_date) & (returns_df.index <= end_date)
        crisis_returns = returns_df.loc[mask]
        
        if crisis_risk_factors.empty or crisis_returns.empty:
            print(f"Insufficient data for {crisis_name}")
            continue
        
        merged_data = pd.merge(
            crisis_returns,
            crisis_risk_factors,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        if merged_data.empty:
            print(f"No matching dates for {crisis_name}")
            continue
        
        correlations = merged_data.corr()['Return'].drop('Return')
        
        risk_factor_changes = {}
        for column in crisis_risk_factors.columns:
            try:
                start_value = crisis_risk_factors[column].iloc[0]
                end_value = crisis_risk_factors[column].iloc[-1]
                if pd.notna(start_value) and pd.notna(end_value) and start_value != 0:
                    pct_change = (end_value - start_value) / start_value
                    risk_factor_changes[column] = pct_change
            except Exception as e:
                print(f"Error calculating change for {column}: {e}")
        
        impact_scores = {}
        for factor, corr in correlations.items():
            if pd.notna(corr) and factor in risk_factor_changes:
                impact_scores[factor] = abs(corr) * abs(risk_factor_changes[factor])
        
        sorted_impact = {k: v for k, v in sorted(impact_scores.items(), key=lambda item: item[1], reverse=True) if pd.notna(v)}
        
        results[crisis_name] = {
            "correlations": correlations,
            "pct_changes": risk_factor_changes,
            "impact_scores": sorted_impact
        }
    
    return results

def plot_crisis_drawdowns(returns_df, crisis_details):
    returns_series = returns_df['Return']
    drawdowns = calculate_drawdowns(returns_series)
    
    try:
        plt.figure(figsize=(15, 8))
        plt.plot(drawdowns, color='gray', alpha=0.5, label='Drawdowns')
        
        colors = ['red', 'orange', 'purple', 'blue']
        for i, (crisis_name, details) in enumerate(crisis_details.items()):
            start_date = pd.to_datetime(details["start_date"])
            end_date = pd.to_datetime(details["end_date"])
            
            mask = (drawdowns.index >= start_date) & (drawdowns.index <= end_date)
            crisis_drawdowns = drawdowns.loc[mask]
            
            if not crisis_drawdowns.empty:
                plt.plot(crisis_drawdowns, color=colors[i % len(colors)], linewidth=2, label=crisis_name)
                
                worst_date = details["worst_date"]
                if worst_date in drawdowns.index:
                    worst_value = drawdowns.loc[worst_date]
                    plt.scatter(worst_date, worst_value, color=colors[i % len(colors)], s=100, zorder=5)
                    plt.text(worst_date, worst_value, f"{worst_value:.2%}", ha='right', va='bottom')
        
        plt.title('Portfolio Drawdowns During Crisis Periods')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{PORTFOLIO_RESULTS_DIR}/crisis_drawdowns.png")
    except Exception as e:
        print(f"Error plotting crisis drawdowns: {e}")
    
def plot_key_risk_factors(risk_factors_df, crisis_details, top_factors_by_crisis):
    for crisis_name, factors in top_factors_by_crisis.items():
        if crisis_name not in crisis_details:
            print(f"Crisis {crisis_name} not found in crisis_details")
            continue
            
        start_date = pd.to_datetime(crisis_details[crisis_name]["start_date"])
        end_date = pd.to_datetime(crisis_details[crisis_name]["end_date"])
        
        try:
            plt.figure(figsize=(15, 8))
            
            mask = (risk_factors_df.index >= start_date) & (risk_factors_df.index <= end_date)
            crisis_data = risk_factors_df.loc[mask]
            
            if crisis_data.empty:
                print(f"No data for {crisis_name}")
                continue
            
            for factor in factors:
                if factor in crisis_data.columns:
                    normalized = crisis_data[factor] / crisis_data[factor].iloc[0] * 100
                    plt.plot(crisis_data.index, normalized, label=factor)
            
            plt.title(f'Top Risk Factors During {crisis_name} (Normalized to 100)')
            plt.ylabel('Normalized Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{PORTFOLIO_RESULTS_DIR}/{crisis_name}_risk_factors.png")
        except Exception as e:
            print(f"Error plotting key risk factors for {crisis_name}: {e}")

def main():
    print("Loading data...")
    portfolio_returns = load_portfolio_returns()
    risk_factors = load_risk_factors()
    
    if portfolio_returns is None or risk_factors is None:
        print("Failed to load required data.")
        return
    
    print("Identifying crisis periods and calculating drawdowns...")
    crisis_details = identify_crisis_periods(portfolio_returns)
    
    print("Crisis periods identified:")
    for crisis, details in crisis_details.items():
        print(f"\n{crisis}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\nAnalyzing risk factor impact...")
    impact_results = analyze_risk_factor_impact(risk_factors, portfolio_returns, crisis_details)
    
    top_factors_by_crisis = {}
    for crisis, results in impact_results.items():
        if "impact_scores" in results and results["impact_scores"]:
            sorted_factors = list(results["impact_scores"].keys())[:5]
            top_factors_by_crisis[crisis] = sorted_factors
            
            print(f"\nTop risk factors during {crisis}:")
            for i, factor in enumerate(sorted_factors, 1):
                pct_change = results["pct_changes"].get(factor, "N/A")
                if isinstance(pct_change, float):
                    pct_change = f"{pct_change:.2%}"
                
                impact = results["impact_scores"].get(factor, "N/A")
                corr = results["correlations"].get(factor, "N/A")
                
                print(f"{i}. {factor}: Impact Score = {impact:.4f}, Correlation = {corr:.4f}")
    
    print("\nGenerating visualizations...")
    plot_crisis_drawdowns(portfolio_returns, crisis_details)
    plot_key_risk_factors(risk_factors, crisis_details, top_factors_by_crisis)
    
    print("\nAnalysis complete. Results saved to the portfolio_results directory.")

if __name__ == "__main__":
    main()
