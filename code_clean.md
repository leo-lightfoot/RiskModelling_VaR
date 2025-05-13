1. Do not use os library for paths. always use a static path to github.
    Portfolio_nav_history = https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/portfolio_results/portfolio_nav_history.csv
    Portfolio_returns_history = https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/portfolio_results/portfolio_returns_history.csv
    Data_restructured = https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/main/data_restructured.csv
2. No verbose comments. Only single line comments when needed.
3. Always place all data reading and path variable near the top of teh script.
4. The script should be readable. so follow simple convention of code.
5. anything that is not being used in the script should be removed.
6. Do not edit anything unless asked.
7. never change the logic in the script.
8. For loading data from GitHub, use requests library and StringIO to properly handle HTTP responses:
   ```python
   import requests
   from io import StringIO
   
   response = requests.get(URL)
   if response.status_code == 200:
       df = pd.read_csv(StringIO(response.text))
   ```
