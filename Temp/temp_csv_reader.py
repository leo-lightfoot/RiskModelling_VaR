import pandas as pd

# Load the data
data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')

# Print column names
print("Column names in the CSV file:")
print(data.columns.tolist())

# Print date info
print("\nDate column format sample:")
print(data['Date'].head())

# Get all FX columns
fx_columns = [col for col in data.columns if 'fx_' in col.lower()]
print("\nAll FX rate columns:", fx_columns)

# If there's no direct INR/USD, we might need to derive it or get data from another source
# Check the existing FX columns
for col in fx_columns:
    print(f"\nFirst 5 rows of {col}:")
    print(data[col].head())

# Print MIBOR data
print("\nFirst 5 rows of MIBOR rate:")
print(data['MIBOR '].head())

# Print Fed Funds rate
print("\nFirst 5 rows of Fed Funds rate:")
print(data['fed_funds_rate'].head()) 