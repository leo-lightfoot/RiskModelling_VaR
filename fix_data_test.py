import pandas as pd
import numpy as np

# Read the data file
print("Reading data file...")
try:
    data = pd.read_csv('data_restructured.csv')
    print("File read successfully.")
    
    # Print the first few rows and column names
    print("\nColumn names:")
    print(data.columns.tolist())
    
    print("\nFirst 3 rows:")
    print(data.head(3))
    
    # Check for issues in the data
    print("\nChecking for issues...")
    
    # Count NaN values and 'Data Unavailable' in each column
    na_counts = data.isna().sum()
    du_counts = data.apply(lambda x: (x == 'Data Unavailable').sum() if x.dtype == 'object' else 0)
    
    print("\nColumns with NaN values:")
    print(na_counts[na_counts > 0])
    
    print("\nColumns with 'Data Unavailable':")
    print(du_counts[du_counts > 0])
    
    # Check for column name issues
    print("\nLooking for column name issues...")
    for col in data.columns:
        if "fxc" in col and "tor" in col:
            print(f"Potential column name issue: {col}")
        if "spread" in col and "," in col:
            print(f"Potential column name issue: {col}")
        if "bitcoin" in col and "priceld" in col:
            print(f"Potential column name issue: {col}")
    
    # Create a fixed version with corrected column names
    print("\nCreating fixed version...")
    fixed_data = data.copy()
    
    # Fix problematic column names
    rename_dict = {}
    for col in fixed_data.columns:
        if "fxc" in col and "tor" in col:
            parts = col.split("fxc")
            potential_fix = parts[0] + "fx_" + parts[1]
            rename_dict[col] = potential_fix
            print(f"Will rename: {col} to {potential_fix}")
        
        if "priceld_credit" in col:
            potential_fix = "bitcoin_usd_price,high_yield_credit_spread"
            rename_dict[col] = potential_fix
            print(f"Will split column: {col} into {potential_fix}")
    
    # Apply column rename
    for old_col, new_col in rename_dict.items():
        if "," in new_col:  # This indicates the column needs to be split
            print(f"Splitting column: {old_col}")
            # Code to split column would go here
        else:
            if old_col in fixed_data.columns:
                fixed_data = fixed_data.rename(columns={old_col: new_col})
                print(f"Renamed: {old_col} to {new_col}")
    
    # Convert 'Data Unavailable' to NaN
    for col in fixed_data.columns:
        if fixed_data[col].dtype == 'object':
            mask = fixed_data[col] == 'Data Unavailable'
            if mask.sum() > 0:
                fixed_data.loc[mask, col] = np.nan
                print(f"Converted 'Data Unavailable' to NaN in column: {col}")
    
    # Save the fixed data
    # fixed_data.to_csv('data_restructured_fixed.csv', index=False)
    # print("\nFixed data saved to 'data_restructured_fixed.csv'")
    
    print("\nSample of columns the S&P futures script needs:")
    cols_needed = ['date', 'sp500_spot', 'fed_funds_rate']
    for col in cols_needed:
        if col in fixed_data.columns:
            print(f"Column '{col}' exists in data")
        else:
            similar_cols = [c for c in fixed_data.columns if col.lower() in c.lower()]
            print(f"Column '{col}' not found. Similar columns: {similar_cols}")
    
except Exception as e:
    print(f"Error: {e}") 