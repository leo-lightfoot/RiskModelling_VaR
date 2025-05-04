import pandas as pd

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')

# Print column names
print("Available columns:")
for col in data.columns:
    print(f"- {col}") 