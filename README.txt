Project Setup Guide

GitHub Repository
- You can also access the codebase here: Market-Risk-Modelling-VaR (https://github.com/leo-lightfoot/Market-Risk-Modelling-VaR)

Steps to Run the Project

1. Ensure "Data Restructured.csv" (20 years of risk factor data) is present. (Currently it used link directly from Github for ease of collaboration and working on same dataset. Local path can also be set in script.)

2. Run Pricing Code
   - Execute Pricing.py first.
   - Outputs generated:
     - combined_instrument_returns.csv
     - portfolio_returns/ folder

4. Understand Generated Outputs
   - combined_instrument_returns.csv: Log returns of instruments.
   - portfolio_returns/ contains:
     - portfolio_nav_history.csv: Portfolio + individual instrument NAVs.
     - portfolio_returns_history.csv: 20-year portfolio returns.
     - portfolio_cumulative_nav_log.png: Log NAV graph (starts at 10M notional).

5. Run VAR Analyses
   - Navigate to VAR_Codes/ folder.
   - Run respective scripts for different Value at Risk methods.

6. Run Historical Stress Test and Stressed Var
   - Execute historical_stress_test.py to view stress test results.
   - Execute stressed_var_maxplot.py to view stressed Var during crisis period.
	
