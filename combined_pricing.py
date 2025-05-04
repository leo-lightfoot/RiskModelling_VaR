import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
import os
import calendar

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')

###############################################################################
#                         DATA LOADING AND CLEANING                           #
###############################################################################

# Load the data
def load_data():
    """Load market data from the CSV file"""
    try:
        data = pd.read_csv(r'C:\Users\abdul\Desktop\Github Repos\RiskModelling_VaR\data_restructured.csv')
        
        # Convert date to datetime format
        data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
        
        # Set date as index
        data.set_index('date', inplace=True)
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to handle "Data Unavailable" values
def clean_numeric_data(df, columns):
    """Clean numeric data in the dataframe"""
    for col in columns:
        # Replace "Data Unavailable" with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Standard function to calculate NAV and log returns
def calculate_nav_and_returns(prices, start_value=100):
    """Calculate NAV and log returns from price series"""
    # Calculate daily returns
    daily_returns = prices.pct_change(fill_method=None)
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Calculate NAV starting with start_value
    nav = pd.Series(index=prices.index, data=np.nan)
    nav.iloc[0] = start_value
    
    for i in range(1, len(nav)):
        if not np.isnan(daily_returns.iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + daily_returns.iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return nav, log_returns

# Function to save results in standardized format
def save_results(result, output_folder, filename):
    """Save results from a pricing function to a CSV file and create visualization
    
    This version handles both the new DataFrame format from price_forward_contract
    and the original format with separate nav and log_returns inputs.
    
    Args:
        result: Either a DataFrame from price_forward_contract or a tuple of (nav, log_returns, additional_data, num_rollovers)
        output_folder: Folder to save results in
        filename: Filename for the CSV output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Check the type of result to determine how to handle it
    if isinstance(result, pd.DataFrame):
        # Assume this is the new format from price_forward_contract
        output_df = result
        
        # Extract key series for plotting
        nav = result['NAV']
        
        # For Forex and Fixed Income, only keep NAV and log return columns
        if output_folder.startswith('Forex') or output_folder.startswith('Fixed_Income'):
            # Identify NAV and log return columns
            nav_cols = [col for col in output_df.columns if col.startswith('NAV_')]
            log_return_cols = [col for col in output_df.columns if col.startswith('Log_Return_')]
            keep_cols = nav_cols + log_return_cols
            if 'Date' in output_df.columns:
                keep_cols.insert(0, 'Date')
            
            # Filter the DataFrame to keep only necessary columns
            output_df = output_df[keep_cols]
        
        # Save to CSV
        try:
            output_df.to_csv(f'{output_folder}/{filename}', index=True)
        except PermissionError:
            print(f"Permission error when saving to {output_folder}/{filename} - file may be open in another application")
            return output_df
        
        # Get the instrument name from the filename
        instrument_name = filename.split('.')[0]
        
    else:
        # Assume this is the old format with tuple of (nav, log_returns, additional_data, num_rollovers)
        nav, log_returns, additional_data, num_rollovers = result
        
        # Ensure NAV series has a name (if not already set)
        if nav.name is None:
            nav.name = filename.split('.')[0]
            
        # Ensure log_returns series has a name (if not already set)
        if hasattr(log_returns, 'name') and log_returns.name is None:
            log_returns.name = nav.name
        
        # Build output DataFrame
        output_df = pd.DataFrame({
            'Date': nav.index,
            f'NAV_{nav.name}': nav.values,
            f'Log_Return_{nav.name}': log_returns.values,
        })
        
        # For non-Forex and non-Fixed Income instruments, add additional data columns
        if additional_data is not None and not (output_folder.startswith('Forex') or output_folder.startswith('Fixed_Income')):
            for col_name, col_data in additional_data.items():
                output_df[col_name] = col_data
        
        # Save to CSV
        try:
            output_df.to_csv(f'{output_folder}/{filename}', index=False)
        except PermissionError:
            print(f"Permission error when saving to {output_folder}/{filename} - file may be open in another application")
            return output_df
        
        # Get the instrument name from the NAV series name
        instrument_name = nav.name
    
    # Plot NAV over time
    try:
        plt.figure(figsize=(14, 8))
        plt.plot(nav.index, nav, label=instrument_name, linewidth=2)
        plt.title(f'NAV of {instrument_name} Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('NAV (log scale)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', ls='--')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.yscale('log')
        date_form = DateFormatter("%Y-%m")
        plt.gca().xaxis.set_major_formatter(date_form)
        plt.savefig(f'{output_folder}/{filename.replace(".csv", "")}_nav.png', bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {instrument_name}: {str(e)}")
    
    return output_df

###############################################################################
#                                EQUITY PRICING                               #
###############################################################################

def price_equities(data, output_folder='Equity'):
    """Price equity securities and save results"""
    print("Pricing equity securities...")
    
    # Identify the equity columns
    equity_columns = [col for col in data.columns if col.startswith('eq_')]
    print(f"Equity columns: {equity_columns}")
    
    # Clean the equity data
    data = clean_numeric_data(data, equity_columns)
    
    # Build a single DataFrame for all equity instruments
    output_df = pd.DataFrame(index=data.index)
    
    # Process each equity
    for col in equity_columns:
        # Calculate NAV and log returns
        equity_name = col.replace('eq_', '')
        nav, log_returns = calculate_nav_and_returns(data[col])
        nav.name = equity_name  # Set the name for the series
        
        # Add to output DataFrame
        output_df[f'NAV_{equity_name}'] = nav.values
        output_df[f'Log_Return_{equity_name}'] = log_returns.values
    
    # Add a combined NAV column for plotting
    combined_nav = pd.Series(index=data.index)
    combined_nav.iloc[0] = 100
    for i in range(1, len(combined_nav)):
        # Average return across all equities
        avg_return = 0
        count = 0
        for col in equity_columns:
            equity_name = col.replace('eq_', '')
            if not np.isnan(output_df[f'NAV_{equity_name}'].iloc[i] / output_df[f'NAV_{equity_name}'].iloc[i-1] - 1):
                avg_return += output_df[f'NAV_{equity_name}'].iloc[i] / output_df[f'NAV_{equity_name}'].iloc[i-1] - 1
                count += 1
        
        if count > 0:
            avg_return = avg_return / count
            combined_nav.iloc[i] = combined_nav.iloc[i-1] * (1 + avg_return)
        else:
            combined_nav.iloc[i] = combined_nav.iloc[i-1]
    
    output_df['NAV'] = combined_nav
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to CSV directly (not using save_results since this is a special case)
    output_df.to_csv(f'{output_folder}/equity_data.csv', index=True)
    
    # Plot NAV over time
    plt.figure(figsize=(14, 8))
    for col in equity_columns:
        equity_name = col.replace('eq_', '')
        plt.plot(data.index, output_df[f'NAV_{equity_name}'], label=equity_name)
    
    plt.title('NAV of Equities Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('NAV (log scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', ls='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    date_form = DateFormatter("%Y-%m")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.savefig(f'{output_folder}/equity_nav_log.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Equity pricing complete.")
    return output_df

###############################################################################
#                          FIXED INCOME PRICING                               #
###############################################################################

def price_10y_treasury_bond(data, output_folder='Fixed_Income/10_year_bond'):
    """Price 10-year Treasury bond and save results"""
    print("Pricing 10-year Treasury bond...")
    
    # Bond pricing parameters
    NOTIONAL = 100
    COUPON_RATE = 0.02  # 2% annual
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 10  # 10 years
    
    # Clean the 10-year yield data
    data = clean_numeric_data(data, ['10y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['10y_treasury_yield'] = data['10y_treasury_yield'] / 100
    
    # Initialize variables for bond pricing
    data['bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2):
        """Calculate bond price using the yield to maturity"""
        coupon_payment = (coupon_rate / frequency) * NOTIONAL
        periods = years_to_maturity * frequency
        period_yield = yield_rate / frequency
        
        # Calculate present value of coupon payments
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
        
        # Calculate present value of face value
        face_value_pv = NOTIONAL / (1 + period_yield)**periods
        
        return coupon_pv + face_value_pv
    
    # Calculate bond prices with roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate bond price
        if not np.isnan(data.loc[current_date, '10y_treasury_yield']):
            data.loc[current_date, 'bond_price'] = calculate_bond_price(
                data.loc[current_date, '10y_treasury_yield'],
                COUPON_RATE,
                time_to_maturity,
                FREQUENCY
            )
    
    # Calculate returns
    data['daily_returns'] = data['bond_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['bond_price'] / data['bond_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['bond_price'].iloc[i] - data['bond_price'].iloc[i-1]) / data['bond_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index, name='10_year_bond')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Bond_Price': data['bond_price'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'Yield': data['10y_treasury_yield'].values * 100  # Convert back to percentage
    }
    
    # Save results using standardized function
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="10_year_bond_data.csv"
    )
    
    print(f"10-year Treasury bond pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

def price_lqd_etf(data, output_folder='Fixed_Income/LQD_ETF'):
    """Price LQD Corporate Bond ETF and save results"""
    print("Pricing LQD Corporate Bond ETF...")
    
    # Identify the LQD ETF column
    lqd_column = 'lqd_corporate_bond_etf'
    
    # Clean the LQD ETF data
    data = clean_numeric_data(data, [lqd_column])
    
    # Calculate NAV and log returns
    nav, log_returns = calculate_nav_and_returns(data[lqd_column])
    nav.name = 'LQD_ETF'  # Set name for the series
    log_returns.name = 'LQD_ETF'  # Set name for consistency
    
    # Save results using standardized function
    output_df = save_results(
        result=(nav, log_returns, None, None),
        output_folder=output_folder,
        filename="lqd_etf_data.csv"
    )
    
    print("LQD ETF pricing complete.")
    return output_df

def price_10y_tips(data, output_folder='Fixed_Income/10_year_TIPS'):
    """Price 10-year TIPS and save results"""
    print("Pricing 10-year TIPS...")
    
    # TIPS pricing parameters
    NOTIONAL = 100
    COUPON_RATE = 0.0125  # 1.25% annual
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 10  # 10 years
    
    # Identify the TIPS yield column
    tips_column = 'Real_10Y_yield'
    
    # Clean the TIPS yield and CPI data
    data = clean_numeric_data(data, [tips_column, 'CPI'])
    
    # Convert yield from percentage to decimal
    data['Real_10Y_yield'] = data[tips_column] / 100
    
    # Convert annual inflation percentage to monthly rate
    data['monthly_inflation'] = data['CPI'] / 12 / 100
    data['monthly_inflation'] = data['monthly_inflation'].ffill()
    
    # Initialize variables for TIPS pricing
    data['tips_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    data['inflation_factor'] = np.nan
    
    def calculate_tips_price(real_yield, coupon_rate, years_to_maturity, frequency=2, inflation_factor=1.0):
        """
        Calculate TIPS price using real yield and inflation adjustment
        """
        if years_to_maturity <= 0:
            return NOTIONAL * inflation_factor
        
        # Adjusted principal for inflation
        adj_principal = NOTIONAL * inflation_factor
        
        # Calculate payments
        periods = int(years_to_maturity * frequency)
        period_yield = real_yield / frequency
        period_coupon = coupon_rate / frequency
        coupon_payment = adj_principal * period_coupon
        
        # Present value of coupon payments
        if abs(period_yield) < 1e-10:  # Handle zero yield case
            coupon_pv = coupon_payment * periods
        else:
            coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
        
        # Present value of principal
        principal_pv = adj_principal / (1 + period_yield)**periods
        
        return coupon_pv + principal_pv
    
    # Track rollover dates and inflation base
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    start_date = data.index[0]
    
    # Calculate TIPS prices with roll-over logic
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            start_date = current_date  # Reset base date for inflation calculation
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate inflation factor from issue date to current date
        months_elapsed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
        relevant_dates = data.iloc[:i+1]
        relevant_dates = relevant_dates[(relevant_dates.index >= start_date) & (relevant_dates.index <= current_date)]
        
        if len(relevant_dates) > 0:
            avg_monthly_inflation = relevant_dates['monthly_inflation'].mean()
            inflation_factor = (1 + avg_monthly_inflation) ** months_elapsed
        else:
            inflation_factor = 1.0
        
        data.loc[current_date, 'inflation_factor'] = inflation_factor
        
        # Calculate TIPS price
        if not pd.isna(data.loc[current_date, 'Real_10Y_yield']):
            data.loc[current_date, 'tips_price'] = calculate_tips_price(
                data.loc[current_date, 'Real_10Y_yield'],
                COUPON_RATE,
                time_to_maturity,
                FREQUENCY,
                inflation_factor
            )
    
    # Calculate returns
    data['daily_returns'] = data['tips_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['tips_price'] / data['tips_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Keep same value on roll dates (no artificial jumps)
            data.loc[data.index[i], 'daily_returns'] = 0
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index, name='10Y_TIPS')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not pd.isna(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'TIPS_Price': data['tips_price'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'Yield': data['Real_10Y_yield'].values * 100,  # Convert back to percentage
        'Inflation_Factor': data['inflation_factor'].values,
        'CPI': data['CPI'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="10_year_tips_data.csv"
    )
    
    print(f"10-year TIPS pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

def price_1y_eur_zcb(data, output_folder='Fixed_Income/1_year_EUR_bond'):
    """Price 1-year EUR Zero Coupon Bond and save results"""
    print("Pricing 1-year EUR Zero Coupon Bond...")
    
    # Bond parameters
    NOTIONAL = 100  # EUR
    MATURITY = 1  # 1 year
    
    # Clean the 1-year EUR yield and FX data
    data = clean_numeric_data(data, ['1_year_euro_yield_curve', 'fx_eurusd_rate'])
    
    # Convert yield from percentage to decimal
    data['1_year_euro_yield_curve'] = data['1_year_euro_yield_curve'] / 100
    
    def calculate_zcb_price(yield_rate, years_to_maturity):
        """Calculate zero coupon bond price using continuous compounding"""
        return NOTIONAL * np.exp(-yield_rate * years_to_maturity)
    
    # Initialize variables for bond pricing
    data['zcb_price_eur'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate bond price in EUR
        if not np.isnan(data.loc[current_date, '1_year_euro_yield_curve']):
            data.loc[current_date, 'zcb_price_eur'] = calculate_zcb_price(
                data.loc[current_date, '1_year_euro_yield_curve'],
                time_to_maturity
            )
    
    # Calculate returns in EUR
    data['daily_returns_eur'] = data['zcb_price_eur'].pct_change(fill_method=None)
    data['log_returns_eur'] = np.log(data['zcb_price_eur'] / data['zcb_price_eur'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['zcb_price_eur'].iloc[i] - data['zcb_price_eur'].iloc[i-1]) / data['zcb_price_eur'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns_eur'] = roll_yield
    
    # Compute NAV in EUR
    nav_eur = pd.Series(index=data.index, name='1_year_eur_zcb')
    nav_eur.iloc[0] = NOTIONAL
    for i in range(1, len(nav_eur)):
        if not np.isnan(data['daily_returns_eur'].iloc[i]):
            nav_eur.iloc[i] = nav_eur.iloc[i-1] * (1 + data['daily_returns_eur'].iloc[i])
        else:
            nav_eur.iloc[i] = nav_eur.iloc[i-1]
    
    # Convert NAV to USD
    nav_usd = pd.Series(index=data.index, name='1_year_EUR_ZCB')
    nav_usd = nav_eur * data['fx_eurusd_rate']
    
    # Calculate USD returns
    data['daily_returns_usd'] = nav_usd.pct_change(fill_method=None)
    data['log_returns_usd'] = np.log(nav_usd / nav_usd.shift(1))
    log_returns_usd = pd.Series(data['log_returns_usd'].values, index=data.index, name='1_year_EUR_ZCB')
    
    # Additional data for output
    additional_data = {
        'NAV_EUR': nav_eur.values,
        'ZCB_Price_EUR': data['zcb_price_eur'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'Yield': data['1_year_euro_yield_curve'].values * 100,  # Convert back to percentage
        'EUR_USD_Rate': data['fx_eurusd_rate'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav_usd, log_returns_usd, additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="1_year_eur_zcb_data.csv"
    )
    
    print(f"1-year EUR Zero Coupon Bond pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

def price_high_yield_corp_debt(data, output_folder='Fixed_Income/High_Yield_CorpDebt'):
    """Price High Yield Corporate Debt and save results"""
    print("Pricing High Yield Corporate Debt...")
    
    # Bond parameters
    NOTIONAL = 100  # USD
    MATURITY = 5  # Typical high-yield bond fund has ~5 year effective duration
    COUPON_RATE = 0.065  # 6.5% typical for high yield corporate bonds
    FREQUENCY = 2  # Semiannual coupon payments
    
    # Clean the yield data
    data = clean_numeric_data(data, ['10y_treasury_yield', 'high_yield_credit spread'])
    
    # Convert yield from percentage to decimal
    data['10y_treasury_yield'] = data['10y_treasury_yield'] / 100
    data['high_yield_credit spread'] = data['high_yield_credit spread'] / 100
    
    # Calculate high yield rate (treasury + high yield spread)
    data['high_yield_rate'] = data['10y_treasury_yield'] + data['high_yield_credit spread']
    
    def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2):
        """Calculate bond price using the yield to maturity with coupon payments"""
        if years_to_maturity <= 0 or np.isnan(yield_rate):
            return np.nan
            
        coupon_payment = (coupon_rate / frequency) * NOTIONAL
        periods = years_to_maturity * frequency
        period_yield = yield_rate / frequency
        
        # Calculate present value of coupon payments
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield if period_yield > 0 else coupon_payment * periods
        
        # Calculate present value of face value
        face_value_pv = NOTIONAL / (1 + period_yield)**periods
        
        return coupon_pv + face_value_pv
    
    # Initialize variables for bond pricing
    data['bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with monthly roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(months=1)  # Monthly roll-over
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over (monthly)
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(months=1)
        
        # Calculate days to maturity for the 5-year bond
        full_maturity = current_date + pd.DateOffset(years=MATURITY)
        days_to_maturity = (full_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate bond price with coupon payments
        if not np.isnan(data.loc[current_date, 'high_yield_rate']):
            data.loc[current_date, 'bond_price'] = calculate_bond_price(
                data.loc[current_date, 'high_yield_rate'],
                COUPON_RATE,
                time_to_maturity,
                FREQUENCY
            )
    
    # Calculate returns
    data['daily_returns'] = data['bond_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['bond_price'] / data['bond_price'].shift(1))
    
    # Add coupon accrual component to returns
    daily_coupon_return = COUPON_RATE / 252  # Daily equivalent of annual coupon
    data['daily_returns'] = data['daily_returns'] + daily_coupon_return
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['bond_price'].iloc[i] - data['bond_price'].iloc[i-1]) / data['bond_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield + daily_coupon_return
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='High_Yield_Corp_Debt')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns'].values, index=data.index, name='High_Yield_Corp_Debt')
    
    # Additional data for output
    additional_data = {
        'Bond_Price': data['bond_price'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'High_Yield_Rate': data['high_yield_rate'].values * 100,  # Convert back to percentage
        'Treasury_Yield': data['10y_treasury_yield'].values * 100,
        'Credit_Spread': data['high_yield_credit spread'].values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="high_yield_corp_debt_data.csv"
    )
    
    print(f"High Yield Corporate Debt pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

def price_5y_green_bond(data, output_folder='Fixed_Income/5year_corp_green_bond'):
    """Price 5-year Green Bond and save results"""
    print("Pricing 5-year Green Bond...")
    
    # Bond parameters
    NOTIONAL = 100
    COUPON_RATE = 0.025  # 2.5% annual
    FREQUENCY = 1        # Annual payments
    MATURITY = 5         # 5 years
    GREENIUM = -0.002    # -20 bps spread
    
    # Clean the yield data
    data = clean_numeric_data(data, ['5y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100
    
    def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=1):
        """
        Calculate bond price using yield to maturity
        """
        if np.isnan(yield_rate) or years_to_maturity <= 0:
            return np.nan
            
        coupon_payment = (coupon_rate / frequency) * NOTIONAL
        periods = int(np.round(years_to_maturity * frequency))
        period_yield = yield_rate / frequency

        # Present value of coupon payments
        if period_yield == 0:
            coupon_pv = coupon_payment * periods
        else:
            coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield

        # Present value of face value
        face_value_pv = NOTIONAL / (1 + period_yield)**periods

        return coupon_pv + face_value_pv
    
    # Initialize variables for bond pricing
    data['bond_price_green'] = np.nan
    data['days_to_maturity_green'] = 0
    data['roll_date_green'] = False
    
    # Calculate bond prices with roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date_green'] = True
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity_green'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate effective yield (Treasury + greenium)
        base_yield = data.loc[current_date, '5y_treasury_yield']
        if not np.isnan(base_yield):
            effective_yield = base_yield + GREENIUM
            data.loc[current_date, 'bond_price_green'] = calculate_bond_price(
                yield_rate=effective_yield,
                coupon_rate=COUPON_RATE,
                years_to_maturity=time_to_maturity,
                frequency=FREQUENCY
            )
    
    # Calculate returns
    data['daily_returns_green'] = data['bond_price_green'].pct_change(fill_method=None)
    data['log_returns_green'] = np.log(data['bond_price_green'] / data['bond_price_green'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date_green'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['bond_price_green'].iloc[i] - data['bond_price_green'].iloc[i-1]) / data['bond_price_green'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns_green'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='Green_Bond')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns_green'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns_green'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns_green'].values, index=data.index, name='Green_Bond')
    
    # Additional data for output
    additional_data = {
        'Bond_Price': data['bond_price_green'].values,
        'Days_to_Maturity': data['days_to_maturity_green'].values,
        'Roll_Date': data['roll_date_green'].values,
        'Yield_Treasury_5Y': data['5y_treasury_yield'].values * 100,  # Convert back to percentage
        'Effective_Yield': (data['5y_treasury_yield'] + GREENIUM).values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['roll_date_green'].sum()),
        output_folder=output_folder,
        filename="5y_green_bond_data.csv"
    )
    
    print(f"5-year Green Bond pricing complete. Number of roll-overs: {data['roll_date_green'].sum()}")
    return output_df

def price_30y_revenue_bond(data, output_folder='Fixed_Income/Revenue_Bond'):
    """Price 30-year Revenue Bond and save results"""
    print("Pricing 30-year Revenue Bond...")
    
    # Bond parameters
    NOTIONAL = 100
    COUPON_RATE = 0.04  # 4% annual
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 30  # 30 years
    SPREAD = 0.008  # 80 bps spread
    
    # Clean the yield data
    data = clean_numeric_data(data, ['30Y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['30Y_treasury_yield'] = data['30Y_treasury_yield'] / 100
    
    def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2):
        """
        Calculate bond price using the yield to maturity
        """
        if np.isnan(yield_rate) or years_to_maturity <= 0:
            return np.nan
            
        coupon_payment = (coupon_rate / frequency) * NOTIONAL
        periods = years_to_maturity * frequency
        period_yield = yield_rate / frequency

        if period_yield == 0:
            return coupon_payment * periods + NOTIONAL

        coupon_pv = coupon_payment * (1 - (1 + period_yield) ** -periods) / period_yield
        face_value_pv = NOTIONAL / (1 + period_yield) ** periods

        return coupon_pv + face_value_pv
    
    # Initialize variables for bond pricing
    data['rev_bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate effective yield (Treasury + spread)
        treasury_yield = data.loc[current_date, '30Y_treasury_yield']
        if not np.isnan(treasury_yield):
            effective_yield = treasury_yield + SPREAD
            data.loc[current_date, 'rev_bond_price'] = calculate_bond_price(
                effective_yield,
                COUPON_RATE,
                time_to_maturity,
                FREQUENCY
            )
    
    # Calculate returns
    data['rev_daily_returns'] = data['rev_bond_price'].pct_change(fill_method=None)
    data['rev_log_returns'] = np.log(data['rev_bond_price'] / data['rev_bond_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['rev_bond_price'].iloc[i] - data['rev_bond_price'].iloc[i-1]) / data['rev_bond_price'].iloc[i-1]
            data.loc[data.index[i], 'rev_daily_returns'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='Revenue_Bond')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['rev_daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['rev_daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['rev_log_returns'].values, index=data.index, name='Revenue_Bond')
    
    # Additional data for output
    additional_data = {
        'Bond_Price': data['rev_bond_price'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'Effective_Yield': (data['30Y_treasury_yield'] + SPREAD).values * 100  # Convert back to percentage
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="30_year_revenue_bond_data.csv"
    )
    
    print(f"30-year Revenue Bond pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

###############################################################################
#                          DERIVATIVES PRICING                                #
###############################################################################

def price_sp500_futures_1m(data, output_folder='Derivatives/1M_S&P_Futures'):
    """Price 1-month S&P 500 futures contracts"""
    print("Pricing S&P 500 Futures...")
    
    # Futures pricing parameters
    DIVIDEND_YIELD = 0.018  # 1.8% annual dividend yield
    START_VALUE = 100
    
    # Function to get next expiry date
    def get_next_expiry(date):
        # S&P 500 futures typically expire on the third Friday of the contract month
        # For simplicity, we'll use the 15th of each month
        current_year = date.year
        current_month = date.month
        
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year
        
        return datetime(next_year, next_month, 15)
    
    # Clean the S&P 500 spot and Fed funds rate data
    data = clean_numeric_data(data, ['sp500_index', 'fed_funds_rate'])
    
    # Convert Fed funds rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    
    # Calculate futures prices with roll-over logic
    current_expiry = get_next_expiry(data.index[0])
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = get_next_expiry(current_date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price
        if not np.isnan(data.loc[current_date, 'sp500_index']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'sp500_index'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] - DIVIDEND_YIELD) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new contract)
            roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index, name='sp500_futures_1m')
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="sp500_futures_1m.csv"
    )
    
    print(f"S&P 500 Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_returns'], additional_data, data['roll_date'].sum())

def price_vix_futures(data, output_folder='Derivatives/VIX_FrontMonth_futures'):
    """Price VIX futures contracts using a mean-reverting model"""
    print("Pricing VIX Futures...")
    
    # Start value for NAV
    START_VALUE = 100
    
    # Clean the VIX data
    data = clean_numeric_data(data, ['vix_index_level'])
    
    # Calculate 30-day moving average (target mean for mean reversion)
    vix_mean = data['vix_index_level'].rolling(window=30).mean()
    
    # Set parameters
    theta = 0.2           # Speed of mean reversion
    sigma = data['vix_index_level'].pct_change().std() * data['vix_index_level'].mean()
    contract_days = 21    # Approx 1 month (trading days)
    initial_price = data['vix_index_level'].iloc[0]
    
    # Initialize simulation
    vix_futures_price = []
    current_price = initial_price
    
    for i in range(len(data)):
        if i < 30:
            # Not enough history for moving average
            vix_futures_price.append(np.nan)
            continue
        
        # Start of a new future contract?
        if (i - 30) % contract_days == 0:
            current_price = data['vix_index_level'].iloc[i]  # Assume it realizes to spot
            days_left = contract_days
        
        # Get long-term mean from moving average
        long_term_mean = vix_mean.iloc[i]
        
        # Mean-reverting update
        shock = sigma * np.random.normal()
        current_price += theta * (long_term_mean - current_price) + shock
        vix_futures_price.append(current_price)
    
    # Create futures price series
    data['contract_price'] = pd.Series(vix_futures_price, index=data.index)
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Identify roll dates (every contract_days days after initial 30 days)
    data['roll_date'] = False
    for i in range(30, len(data), contract_days):
        if i < len(data):
            data.iloc[i, data.columns.get_loc('roll_date')] = True
    
    # Calculate days to expiry
    data['days_to_expiry'] = 0
    for i in range(len(data)):
        if i < 30:
            data.iloc[i, data.columns.get_loc('days_to_expiry')] = np.nan
            continue
        
        # Calculate how many days since the last roll date
        days_since_roll = (i - 30) % contract_days
        data.iloc[i, data.columns.get_loc('days_to_expiry')] = contract_days - days_since_roll
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='vix_futures')
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'VIX_Spot': data['vix_index_level'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    direct_output_df = pd.DataFrame({
        'Date': nav.index,
        'NAV_VIX_Futures_1M': nav.values,
        'Log_Return_VIX_Futures_1M': data['log_returns'].values
    })
    direct_output_df.to_csv(f'{output_folder}/vix_futures_data.csv', index=False)
    
    print(f"VIX Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_returns'], additional_data, data['roll_date'].sum())

def price_crude_oil_futures(data, output_folder='Derivatives/1M_Crude_Oil_Futures'):
    """Price 1-month crude oil futures contracts"""
    print("Pricing Crude Oil Futures...")
    
    # Futures pricing parameters
    STORAGE_COST = 0.02/12  # Monthly storage cost (2% annual)
    CONVENIENCE_YIELD = 0.01/12  # Monthly convenience yield (1% annual)
    START_VALUE = 100
    
    # Function to get next expiry date for crude oil futures
    def get_next_expiry(date):
        # Crude oil futures typically expire around the 20th of each month
        current_year = date.year
        current_month = date.month
        
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year
        
        return datetime(next_year, next_month, 20)
    
    # Clean the crude oil spot and interest rate data
    data = clean_numeric_data(data, ['crude_oil_wti_spot', 'fed_funds_rate'])
    
    # Convert interest rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    
    # Calculate futures prices with roll-over logic
    current_expiry = get_next_expiry(data.index[0])
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = get_next_expiry(current_date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price using the cost-of-carry model
        if not np.isnan(data.loc[current_date, 'crude_oil_wti_spot']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'crude_oil_wti_spot'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST - CONVENIENCE_YIELD) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new contract)
            roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='crude_oil_futures')
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'Spot_Price': data['crude_oil_wti_spot'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="crude_oil_futures.csv"
    )
    
    print(f"Crude Oil Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_returns'], additional_data, data['roll_date'].sum())

def price_gold_futures(data, output_folder='Derivatives/3M_Gold_Futures'):
    """Price 3-month gold futures contracts"""
    print("Pricing Gold Futures...")
    
    # Futures pricing parameters
    STORAGE_COST = 0.01/4  # Quarterly storage cost (1% annual)
    CONVENIENCE_YIELD = 0  # No convenience yield for gold
    START_VALUE = 100
    
    # Function to get next expiry date for gold futures
    def get_next_expiry(date):
        # Gold futures typically expire on a quarterly cycle (Feb, Apr, Jun, Aug, Oct, Dec)
        # For simplicity, we'll use a 3-month cycle from the start date
        current_month = date.month
        current_year = date.year
        
        # Calculate next quarter month
        next_month = ((current_month - 1 + 3) % 12) + 1
        next_year = current_year + (1 if next_month < current_month else 0)
        
        return datetime(next_year, next_month, 15)  # Mid-month approximation
    
    # Clean the gold spot and interest rate data
    data = clean_numeric_data(data, ['gold_spot_price', 'fed_funds_rate'])
    
    # Convert interest rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    
    # Calculate futures prices with roll-over logic
    current_expiry = get_next_expiry(data.index[0])
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = get_next_expiry(current_date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price using the cost-of-carry model
        if not np.isnan(data.loc[current_date, 'gold_spot_price']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'gold_spot_price'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST - CONVENIENCE_YIELD) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new contract)
            roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='gold_futures')
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'Spot_Price': data['gold_spot_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="gold_futures.csv"
    )
    
    print(f"Gold Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_returns'], additional_data, data['roll_date'].sum())

def price_soybean_futures(data, output_folder='Derivatives/6M_Soybean_Futures'):
    """Price 6-month soybean futures contracts"""
    print("Pricing Soybean Futures...")
    
    # Futures pricing parameters
    STORAGE_COST = 0.03/2  # Semi-annual storage cost (3% annual)
    CONVENIENCE_YIELD = 0.02/2  # Semi-annual convenience yield (2% annual)
    START_VALUE = 100
    
    # Function to get next expiry date for soybean futures
    def get_next_expiry(date):
        # Soybean futures typically expire on a specific schedule
        # For simplicity, we'll use a 6-month cycle from the start date
        current_month = date.month
        current_year = date.year
        
        # Calculate next 6-month point
        next_month = ((current_month - 1 + 6) % 12) + 1
        next_year = current_year + (1 if next_month < current_month else 0)
        
        return datetime(next_year, next_month, 15)  # Mid-month approximation
    
    # Clean the soybean spot and interest rate data
    data = clean_numeric_data(data, ['soybean_spot_usd', 'fed_funds_rate'])
    
    # Convert interest rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    
    # Calculate futures prices with roll-over logic
    current_expiry = get_next_expiry(data.index[0])
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = get_next_expiry(current_date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price using the cost-of-carry model
        if not np.isnan(data.loc[current_date, 'soybean_spot_usd']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'soybean_spot_usd'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST - CONVENIENCE_YIELD) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new contract)
            roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='soybean_futures')
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'Spot_Price': data['soybean_spot_usd'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="soybean_futures.csv"
    )
    
    print(f"Soybean Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_returns'], additional_data, data['roll_date'].sum())

###############################################################################
#                              FOREX PRICING                                  #
###############################################################################

def price_gbpusd_6m_forward(data, output_folder='Forex/GBP_USD_6M_Forward'):
    """Price 6-month GBP/USD forward contract"""
    print("Pricing GBP/USD 6-month forward contract...")
    
    # Forward contract parameters
    NOTIONAL_GBP = 1.0  # Notional amount in GBP
    DAYS_IN_YEAR = 365
    DAYS_FORWARD = 182  # Approximately 6 months
    
    # Function to get next expiry date
    def get_next_expiry(date):
        # For a 6-month forward, expiry is 182 days from entry
        return date + timedelta(days=DAYS_FORWARD)
    
    # Clean the GBP/USD spot and interest rate data
    data = clean_numeric_data(data, ['fx_gbpusd_rate', 'fed_funds_rate', 'factor_GBP_sonia'])
    
    # Rename columns for clarity
    data_fx = data.copy()
    data_fx.rename(columns={
        'fx_gbpusd_rate': 'spot',
        'fed_funds_rate': 'usd_rate',
        'factor_GBP_sonia': 'gbp_rate'
    }, inplace=True)
    
    # Convert rates from percentage to decimal
    data_fx['usd_rate'] = data_fx['usd_rate'] / 100
    data_fx['gbp_rate'] = data_fx['gbp_rate'] / 100
    
    # Drop rows with NaN values in essential columns
    data_fx = data_fx.dropna(subset=['spot', 'usd_rate', 'gbp_rate'])
    
    # Initialize columns for forward analysis
    data_fx['forward_rate'] = np.nan
    data_fx['days_to_expiry'] = np.nan
    data_fx['roll_date'] = False
    data_fx['contract_pnl'] = np.nan  # P&L of each individual contract
    data_fx['active_contract'] = False  # Flag for the currently active contract
    
    # Create NAV series starting at 100
    nav_series = pd.Series(index=data_fx.index, data=np.nan)
    nav_series.iloc[0] = 100
    
    entry_dates = []
    forward_rates = []
    expiry_dates = []
    
    # Process each date sequentially
    current_contract = None
    for i in range(len(data_fx)):
        current_date = data_fx.index[i]
        
        # If we don't have an active contract or the current contract has expired, enter a new one
        if current_contract is None or current_date >= current_contract['expiry_date']:
            # Mark roll date if we're rolling over (not the first contract)
            if current_contract is not None:
                data_fx.loc[current_date, 'roll_date'] = True
            
            # Calculate the forward rate using interest rate parity
            spot = data_fx.loc[current_date, 'spot']
            r_gbp = data_fx.loc[current_date, 'gbp_rate']
            r_usd = data_fx.loc[current_date, 'usd_rate']
            T = DAYS_FORWARD / DAYS_IN_YEAR
            
            # Forward rate formula: Spot * (1 + r_foreign) / (1 + r_base)
            forward_rate = spot * (1 + r_usd * T) / (1 + r_gbp * T)
            
            # Store contract details
            current_contract = {
                'entry_date': current_date,
                'expiry_date': current_date + timedelta(days=DAYS_FORWARD),
                'forward_rate': forward_rate,
                'notional_gbp': NOTIONAL_GBP
            }
            
            # Store dates and rates for debugging/analysis
            entry_dates.append(current_date)
            forward_rates.append(forward_rate)
            expiry_dates.append(current_contract['expiry_date'])
            
            # Mark this row as an active contract entry point
            data_fx.loc[current_date, 'active_contract'] = True
        
        # Update data for the current date
        data_fx.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
        data_fx.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
        
        # Calculate unrealized P&L (mark-to-market)
        # For a GBP/USD forward, buying GBP and selling USD:
        # P&L = Notional * (Spot - Forward Rate)
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_GBP * (spot - forward_rate)
        data_fx.loc[current_date, 'contract_pnl'] = pnl
    
    # Calculate daily returns based on P&L changes
    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_GBP
    data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    
    # Fill NaN values in daily returns with zeros (for roll dates)
    # Using recommended approach to avoid FutureWarning
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav_series = pd.Series(index=data_fx.index)
    nav_series.iloc[0] = 100  # Start with 100
    for i in range(1, len(data_fx)):
        prev_nav = nav_series.iloc[i-1]
        daily_return = data_fx['daily_return'].iloc[i]
        nav_series.iloc[i] = prev_nav * (1 + daily_return)
    
    # Compute log returns of NAV
    log_returns = np.log(nav_series / nav_series.shift(1))
    
    # Build output DataFrame
    output_df = pd.DataFrame({
        'Date': data_fx.index,
        'NAV_GBPUSD_6M_Forward': nav_series.values,
        'Log_Return_GBPUSD_6M_Forward': log_returns.values,
        'Forward_Rate': data_fx['forward_rate'].values,
        'Spot_Rate': data_fx['spot'].values,
        'Days_to_Expiry': data_fx['days_to_expiry'].values,
        'Roll_Date': data_fx['roll_date'].values,
        'Contract_PnL': data_fx['contract_pnl'].values
    })
    
    # Make NAV series with name for the standardized saving function
    nav_named = pd.Series(nav_series.values, index=data_fx.index, name='GBPUSD_6M_Forward')
    log_returns_named = pd.Series(log_returns.values, index=data_fx.index, name='GBPUSD_6M_Forward')
    
    # Additional data for standardized output
    additional_data = {
        'Forward_Rate': data_fx['forward_rate'].values,
        'Spot_Rate': data_fx['spot'].values,
        'Days_to_Expiry': data_fx['days_to_expiry'].values,
        'Roll_Date': data_fx['roll_date'].values,
        'Contract_PnL': data_fx['contract_pnl'].values
    }
    
    # Save results using standardized function
    save_results(
        result=(nav_named, log_returns_named, additional_data, data_fx['roll_date'].sum()),
        output_folder=output_folder,
        filename="gbpusd_6m_forward_data.csv"
    )
    
    print(f"GBP/USD 6-month forward pricing complete. Number of contracts: {len(entry_dates)}, Number of roll-overs: {data_fx['roll_date'].sum()}")
    return (nav_named, log_returns_named, additional_data, data_fx['roll_date'].sum())

def price_usdinr_3m_forward(data, output_folder='Forex/INR_USD_3M_Forward_Short'):
    """Price 3-month USD/INR forward contract (short position)"""
    print("Pricing USD/INR 3-month forward contract (short position)...")
    
    # Forward contract parameters
    NOTIONAL_INR = 10000.0  # Notional amount in INR (10000 INR)
    DAYS_IN_YEAR = 365
    DAYS_FORWARD = 91  # Approximately 3 months
    
    # Function to get next expiry date
    def get_next_expiry(date):
        # For a 3-month forward, expiry is 91 days from entry
        return date + timedelta(days=DAYS_FORWARD)
    
    # Clean the USD/INR spot and interest rate data
    data = clean_numeric_data(data, ['USD_INR', 'fed_funds_rate', 'MIBOR '])
    
    # Rename columns for clarity
    data_fx = data.copy()
    data_fx.rename(columns={
        'USD_INR': 'spot',
        'fed_funds_rate': 'usd_rate',
        'MIBOR ': 'inr_rate'  # Note the space in 'MIBOR '
    }, inplace=True)
    
    # Convert rates from percentage to decimal
    data_fx['usd_rate'] = data_fx['usd_rate'] / 100
    data_fx['inr_rate'] = data_fx['inr_rate'] / 100
    
    # Drop rows with NaN values in essential columns
    data_fx = data_fx.dropna(subset=['spot', 'usd_rate', 'inr_rate'])
    
    # Initialize columns for forward analysis
    data_fx['forward_rate'] = np.nan
    data_fx['days_to_expiry'] = np.nan
    data_fx['roll_date'] = False
    data_fx['contract_pnl'] = np.nan  # P&L of each individual contract
    data_fx['active_contract'] = False  # Flag for the currently active contract
    
    # Create NAV series starting at 100
    nav_series = pd.Series(index=data_fx.index, data=np.nan)
    nav_series.iloc[0] = 100
    
    entry_dates = []
    forward_rates = []
    expiry_dates = []
    
    # Process each date sequentially
    current_contract = None
    for i in range(len(data_fx)):
        current_date = data_fx.index[i]
        
        # If we don't have an active contract or the current contract has expired, enter a new one
        if current_contract is None or current_date >= current_contract['expiry_date']:
            # Mark roll date if we're rolling over (not the first contract)
            if current_contract is not None:
                data_fx.loc[current_date, 'roll_date'] = True
            
            # Calculate the forward rate using interest rate parity
            spot = data_fx.loc[current_date, 'spot']
            r_inr = data_fx.loc[current_date, 'inr_rate']
            r_usd = data_fx.loc[current_date, 'usd_rate']
            T = DAYS_FORWARD / DAYS_IN_YEAR
            
            # Forward rate formula: Spot * (1 + r_base) / (1 + r_foreign)
            # For USD/INR, USD is the base and INR is the foreign currency
            forward_rate = spot * (1 + r_usd * T) / (1 + r_inr * T)
            
            # Store contract details
            current_contract = {
                'entry_date': current_date,
                'expiry_date': current_date + timedelta(days=DAYS_FORWARD),
                'forward_rate': forward_rate,
                'notional_inr': NOTIONAL_INR
            }
            
            # Store dates and rates for debugging/analysis
            entry_dates.append(current_date)
            forward_rates.append(forward_rate)
            expiry_dates.append(current_contract['expiry_date'])
            
            # Mark this row as an active contract entry point
            data_fx.loc[current_date, 'active_contract'] = True
        
        # Update data for the current date
        data_fx.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
        data_fx.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
        
        # Calculate unrealized P&L (mark-to-market)
        # For a USD/INR forward, selling USD and buying INR:
        # P&L = Notional * (Spot - Forward Rate) / Spot
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_INR * (spot - forward_rate) / spot  # Adjusted for short position
        data_fx.loc[current_date, 'contract_pnl'] = pnl
    
    # Calculate daily returns based on P&L changes
    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_INR
    data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    
    # Fill NaN values in daily returns with zeros (for roll dates)
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav_series = pd.Series(index=data_fx.index)
    nav_series.iloc[0] = 100  # Start with 100
    for i in range(1, len(data_fx)):
        prev_nav = nav_series.iloc[i-1]
        daily_return = data_fx['daily_return'].iloc[i]
        nav_series.iloc[i] = prev_nav * (1 + daily_return)
    
    # Compute log returns of NAV
    log_returns = np.log(nav_series / nav_series.shift(1))
    
    # Make NAV series with name for the standardized saving function
    nav_named = pd.Series(nav_series.values, index=data_fx.index, name='USDINR_3M_Forward_Short')
    log_returns_named = pd.Series(log_returns.values, index=data_fx.index, name='USDINR_3M_Forward_Short')
    
    # Additional data for standardized output
    additional_data = {
        'Forward_Rate': data_fx['forward_rate'].values,
        'Spot_Rate': data_fx['spot'].values,
        'Days_to_Expiry': data_fx['days_to_expiry'].values,
        'Roll_Date': data_fx['roll_date'].values,
        'Contract_PnL': data_fx['contract_pnl'].values
    }
    
    # Save results using standardized function
    save_results(
        result=(nav_named, log_returns_named, additional_data, data_fx['roll_date'].sum()),
        output_folder=output_folder,
        filename="usdinr_3m_forward_data.csv"
    )
    
    print(f"USD/INR 3-month forward (short) pricing complete. Number of contracts: {len(entry_dates)}, Number of roll-overs: {data_fx['roll_date'].sum()}")
    return (nav_named, log_returns_named, additional_data, data_fx['roll_date'].sum())

###############################################################################
#                              MAIN EXECUTION                                 #
###############################################################################

def main():
    """Main function to price all securities"""
    print("Starting combined pricing script for all instruments...")
    
    # Load market data
    data = load_data()
    if data is None:
        print("Failed to load market data. Exiting.")
        return
    
    print(f"Loaded market data with {len(data)} rows.")
    print(f"Available columns: {', '.join(data.columns.tolist())}")
    
    # Create a dictionary to store all pricing results
    pricing_results = {}
    
    # 1. Price equity securities
    try:
        equity_results = price_equities(data)
        pricing_results['Equity'] = equity_results
    except Exception as e:
        print(f"Error pricing equities: {str(e)}")
    
    # 2. Price fixed income securities
    # 2.1. 10-year Treasury Bond
    try:
        if '10y_treasury_yield' in data.columns:
            bond_results = price_10y_treasury_bond(data)
            pricing_results['10y_Bond'] = bond_results
        else:
            print("Skipping 10y Treasury Bond: Required column '10y_treasury_yield' not found in dataset")
    except Exception as e:
        print(f"Error pricing 10-year bond: {str(e)}")
        
    # 2.2. LQD ETF
    try:
        if 'lqd_corporate_bond_etf' in data.columns:
            lqd_results = price_lqd_etf(data)
            pricing_results['LQD_ETF'] = lqd_results
        else:
            print("Skipping LQD ETF: Required column 'lqd_corporate_bond_etf' not found in dataset")
    except Exception as e:
        print(f"Error pricing LQD ETF: {str(e)}")
        
    # 2.3. 10-year TIPS
    try:
        if 'Real_10Y_yield' in data.columns:
            tips_results = price_10y_tips(data)
            pricing_results['10y_TIPS'] = tips_results
        else:
            print("Skipping 10y TIPS: Required column 'Real_10Y_yield' not found in dataset")
    except Exception as e:
        print(f"Error pricing 10-year TIPS: {str(e)}")
        
    # 2.4. 1-year EUR Zero Coupon Bond
    try:
        if '1_year_euro_yield_curve' in data.columns and 'fx_eurusd_rate' in data.columns:
            eur_zcb_results = price_1y_eur_zcb(data)
            pricing_results['1y_EUR_ZCB'] = eur_zcb_results
        else:
            missing = []
            for col in ['1_year_euro_yield_curve', 'fx_eurusd_rate']:
                if col not in data.columns:
                    missing.append(col)
            print(f"Skipping 1-year EUR Zero Coupon Bond: Required columns {missing} not found in dataset")
    except Exception as e:
        print(f"Error pricing 1-year EUR Zero Coupon Bond: {str(e)}")
        
    # 2.5. High Yield Corporate Debt
    try:
        if 'high_yield_credit spread' in data.columns and '10y_treasury_yield' in data.columns:
            hy_corp_debt_results = price_high_yield_corp_debt(data)
            pricing_results['High_Yield_Corp_Debt'] = hy_corp_debt_results
        else:
            missing = []
            for col in ['high_yield_credit spread', '10y_treasury_yield']:
                if col not in data.columns:
                    missing.append(col)
            print(f"Skipping High Yield Corporate Debt: Required columns {missing} not found in dataset")
    except Exception as e:
        print(f"Error pricing High Yield Corporate Debt: {str(e)}")
    
    # 2.6. 5-year Green Bond
    try:
        if '5y_treasury_yield' in data.columns:
            green_bond_results = price_5y_green_bond(data)
            pricing_results['5y_Green_Bond'] = green_bond_results
        else:
            print("Skipping 5-year Green Bond: Required column '5y_treasury_yield' not found in dataset")
    except Exception as e:
        print(f"Error pricing 5-year Green Bond: {str(e)}")
    
    # 2.7. 30-year Revenue Bond
    try:
        if '30Y_treasury_yield' in data.columns:
            revenue_bond_results = price_30y_revenue_bond(data)
            pricing_results['30y_Revenue_Bond'] = revenue_bond_results
        else:
            print("Skipping 30-year Revenue Bond: Required column '30Y_treasury_yield' not found in dataset")
    except Exception as e:
        print(f"Error pricing 30-year Revenue Bond: {str(e)}")
    
    # 3. Price derivatives
    # 3.1. S&P 500 Futures (1 month)
    try:
        if 'sp500_index' in data.columns and 'fed_funds_rate' in data.columns:
            sp500_futures_results = price_sp500_futures_1m(data)
            pricing_results['SP500_Futures_1M'] = sp500_futures_results
        else:
            print("Skipping S&P 500 Futures: Required columns ('sp500_index', 'fed_funds_rate') not found in dataset")
    except Exception as e:
        print(f"Error pricing S&P 500 futures: {str(e)}")
        
    # 3.2. VIX Futures (front month)
    try:
        if 'vix_index_level' in data.columns and 'fed_funds_rate' in data.columns:
            vix_futures_results = price_vix_futures(data)
            pricing_results['VIX_Futures'] = vix_futures_results
        else:
            print("Skipping VIX Futures: Required columns ('vix_index_level', 'fed_funds_rate') not found in dataset")
    except Exception as e:
        print(f"Error pricing VIX futures: {str(e)}")
        
    # 3.3. Crude Oil Futures (1 month)
    try:
        if 'crude_oil_wti_spot' in data.columns and 'fed_funds_rate' in data.columns:
            oil_futures_results = price_crude_oil_futures(data)
            pricing_results['Crude_Oil_Futures_1M'] = oil_futures_results
        else:
            print("Skipping Crude Oil Futures: Required columns ('crude_oil_wti_spot', 'fed_funds_rate') not found in dataset")
    except Exception as e:
        print(f"Error pricing Crude Oil futures: {str(e)}")
        
    # 3.4. Gold Futures (3 month)
    try:
        if 'gold_spot_price' in data.columns and 'fed_funds_rate' in data.columns:
            gold_futures_results = price_gold_futures(data)
            pricing_results['Gold_Futures_3M'] = gold_futures_results
        else:
            print("Skipping Gold Futures: Required columns ('gold_spot_price', 'fed_funds_rate') not found in dataset")
    except Exception as e:
        print(f"Error pricing Gold futures: {str(e)}")
        
    # 3.5. Soybean Futures (6 month)
    try:
        if 'soybean_spot_usd' in data.columns and 'fed_funds_rate' in data.columns:
            soybean_futures_results = price_soybean_futures(data)
            pricing_results['Soybean_Futures_6M'] = soybean_futures_results
        else:
            print("Skipping Soybean Futures: Required columns ('soybean_spot_usd', 'fed_funds_rate') not found in dataset")
    except Exception as e:
        print(f"Error pricing Soybean futures: {str(e)}")
    
    # 4. Price forex
    # 4.1. GBP/USD Forward (6 month)
    try:
        if 'fx_gbpusd_rate' in data.columns and 'fed_funds_rate' in data.columns and 'factor_GBP_sonia' in data.columns:
            gbpusd_results = price_gbpusd_6m_forward(data)
            pricing_results['GBPUSD_Forward'] = gbpusd_results
        else:
            missing = []
            for col in ['fx_gbpusd_rate', 'fed_funds_rate', 'factor_GBP_sonia']:
                if col not in data.columns:
                    missing.append(col)
            print(f"Skipping GBP/USD Forward: Required columns {missing} not found in dataset")
    except Exception as e:
        print(f"Error pricing GBP/USD forward: {str(e)}")
        
    # 4.2. USD/INR Forward (3 month, short position)
    try:
        if 'USD_INR' in data.columns and 'fed_funds_rate' in data.columns and 'MIBOR ' in data.columns:
            usdinr_results = price_usdinr_3m_forward(data)
            pricing_results['USDINR_Forward'] = usdinr_results
        else:
            missing = []
            for col in ['USD_INR', 'fed_funds_rate', 'MIBOR ']:
                if col not in data.columns:
                    missing.append(col)
            print(f"Skipping USD/INR Forward: Required columns {missing} not found in dataset")
    except Exception as e:
        print(f"Error pricing USD/INR forward: {str(e)}")
    
    # Summary of successfully priced instruments
    print("\nSuccessfully priced instruments:")
    for instrument in pricing_results.keys():
        print(f"- {instrument}")
    
    print("\nAll pricing complete. Results saved to respective directories.")
    print("You can now run the Portfolio.py script to analyze the combined portfolio.")

if __name__ == "__main__":
    main() 