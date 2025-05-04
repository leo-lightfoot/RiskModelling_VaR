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
    daily_returns = prices.pct_change()
    
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
        
        # Save to CSV
        output_df.to_csv(f'{output_folder}/{filename}', index=True)
        
        # Get the instrument name from the filename
        instrument_name = filename.split('.')[0]
        
    else:
        # Assume this is the old format with tuple of (nav, log_returns, additional_data, num_rollovers)
        nav, log_returns, additional_data, num_rollovers = result
        
        # Build output DataFrame
        output_df = pd.DataFrame({
            'Date': nav.index,
            f'NAV_{nav.name}': nav.values,
            f'Log_Return_{nav.name}': log_returns.values,
        })
        
        # Add any additional data columns
        if additional_data is not None:
            for col_name, col_data in additional_data.items():
                output_df[col_name] = col_data
        
        # Save to CSV
        output_df.to_csv(f'{output_folder}/{filename}', index=False)
        
        # Get the instrument name from the NAV series name
        instrument_name = nav.name
    
    # Plot NAV over time
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
    
    # Identify the TIPS yield column - use Real_10Y_yield instead of 10Y_TIPS_yld
    tips_column = 'Real_10Y_yield'
    
    # Clean the TIPS yield data
    data = clean_numeric_data(data, [tips_column])
    
    # Convert yield from percentage to decimal
    data['10y_tips_yield'] = data[tips_column] / 100
    
    # Bond pricing parameters
    NOTIONAL = 100
    COUPON_RATE = 0.01  # 1% annual for TIPS
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 10  # 10 years
    
    # Calculate inflation adjustment (simplified - using CPI data if available)
    if 'CPI' in data.columns:
        data = clean_numeric_data(data, ['CPI'])
        # Calculate cumulative inflation factor
        data['inflation_factor'] = (1 + data['CPI'] / 100).cumprod()
        data['inflation_factor'] = data['inflation_factor'] / data['inflation_factor'].iloc[0]
    else:
        # If no inflation data available, use a simplified model (e.g., 2% annual inflation)
        days_since_start = [(date - data.index[0]).days for date in data.index]
        data['inflation_factor'] = [1.02 ** (days / 365) for days in days_since_start]
    
    # Initialize variables for TIPS pricing
    data['tips_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    def calculate_tips_price(yield_rate, coupon_rate, years_to_maturity, inflation_factor, frequency=2):
        """Calculate TIPS price using real yield and inflation adjustment"""
        # Adjust notional for inflation
        adjusted_notional = NOTIONAL * inflation_factor
        
        # Calculate real coupon payment (adjusted for inflation)
        coupon_payment = (coupon_rate / frequency) * adjusted_notional
        
        # Number of remaining coupon payments
        periods = years_to_maturity * frequency
        
        # Discount rate for each period
        period_yield = yield_rate / frequency
        
        # Present value of coupon payments
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
        
        # Present value of inflation-adjusted principal
        principal_pv = adjusted_notional / (1 + period_yield)**periods
        
        return coupon_pv + principal_pv
    
    # Calculate TIPS prices with roll-over logic
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
        
        # Calculate TIPS price
        if not np.isnan(data.loc[current_date, '10y_tips_yield']):
            data.loc[current_date, 'tips_price'] = calculate_tips_price(
                data.loc[current_date, '10y_tips_yield'],
                COUPON_RATE,
                time_to_maturity,
                data.loc[current_date, 'inflation_factor'],
                FREQUENCY
            )
    
    # Calculate returns
    data['daily_returns'] = data['tips_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['tips_price'] / data['tips_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield
            roll_yield = (data['tips_price'].iloc[i] - data['tips_price'].iloc[i-1]) / data['tips_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='10Y_TIPS')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'TIPS_Price': data['tips_price'].values,
        'Days_to_Maturity': data['days_to_maturity'].values,
        'Roll_Date': data['roll_date'].values,
        'Real_Yield': data['10y_tips_yield'].values * 100,  # Convert back to percentage
        'Inflation_Factor': data['inflation_factor'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="10_year_tips_data.csv"
    )
    
    print(f"10-year TIPS pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return output_df

###############################################################################
#                          DERIVATIVES PRICING                                #
###############################################################################

def price_futures_contract(data, spot_column, rate_column='fed_funds_rate', 
                          expiry_func=None, storage_cost=0, convenience_yield=0,
                          start_value=100, output_prefix='futures'):
    """
    Generic function to price futures contracts with consistent methodology
    
    Args:
        data: DataFrame with market data
        spot_column: Column name for the spot price
        rate_column: Column name for the interest rate
        expiry_func: Function that determines expiry dates
        storage_cost: Annual storage cost rate
        convenience_yield: Annual convenience yield rate
        start_value: Initial NAV value
        output_prefix: Prefix for output column names
        
    Returns:
        DataFrame with pricing results
    """
    # Clean the spot and interest rate data
    clean_cols = [spot_column, rate_column]
    data = clean_numeric_data(data, clean_cols)
    
    # Convert interest rate from percentage to decimal
    data[rate_column] = data[rate_column] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    
    # Calculate futures prices with roll-over logic
    current_expiry = expiry_func(data.index[0])
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = expiry_func(current_date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price using the cost-of-carry model
        # Futures = Spot * exp((r + storage - convenience_yield) * T)
        if not np.isnan(data.loc[current_date, spot_column]) and not np.isnan(data.loc[current_date, rate_column]):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, spot_column] * np.exp(
                (data.loc[current_date, rate_column] + storage_cost - convenience_yield) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new contract)
            roll_yield = (data['contract_price'].iloc[i] - data['contract_price'].iloc[i-1]) / data['contract_price'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index, name=f'{output_prefix}')
    nav.iloc[0] = start_value
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
    
    return nav, data['log_returns'], additional_data, data['roll_date'].sum()

def get_sp500_expiry(date):
    """Calculate next expiry date for S&P 500 futures"""
    year = date.year
    month = date.month
    
    # S&P 500 futures expire on the third Friday of March, June, September, and December
    # Determine which quarter we're in
    if month < 3:
        expiry_month = 3  # March
    elif month < 6:
        expiry_month = 6  # June
    elif month < 9:
        expiry_month = 9  # September
    elif month < 12:
        expiry_month = 12  # December
    else:  # December, so next expiry is March of next year
        expiry_month = 3
        year += 1
    
    # Find the third Friday of the expiry month
    first_day = datetime(year, expiry_month, 1)
    # Find the first Friday
    first_friday = first_day + timedelta(days=((4 - first_day.weekday()) % 7))
    # Add two weeks to get the third Friday
    third_friday = first_friday + timedelta(days=14)
    
    return third_friday

def price_sp500_futures_1m(data):
    """Price 1-month S&P 500 futures contracts"""
    print("Pricing S&P 500 Futures...")
    
    # Use the generic futures pricing function
    result = price_futures_contract(
        data=data,
        spot_column='sp500_index',
        rate_column='fed_funds_rate',
        expiry_func=get_sp500_expiry,
        storage_cost=0,  # No storage cost for index futures
        convenience_yield=0.005/12,  # Monthly convenience yield (dividend-like)
        start_value=1000,
        output_prefix='sp500_futures'
    )
    
    # Save results
    save_results(result, 'Derivatives/1M_S&P_Futures', 'sp500_futures.csv')
    
    return result

def get_vix_expiry(date):
    """Calculate next expiry date for VIX futures"""
    year = date.year
    month = date.month
    
    # VIX futures expire on the Wednesday that is 30 days prior to the third Friday
    # of the calendar month immediately following the month in which the contract expires
    
    # First, determine the next month
    if month == 12:
        next_month = 1
        next_year = year + 1
    else:
        next_month = month + 1
        next_year = year
    
    # Find the third Friday of the next month
    first_day = datetime(next_year, next_month, 1)
    # Find the first Friday
    first_friday = first_day + timedelta(days=((4 - first_day.weekday()) % 7))
    # Add two weeks to get the third Friday
    third_friday = first_friday + timedelta(days=14)
    
    # Go back 30 days to get settlement date (nearest Wednesday)
    expiry = third_friday - timedelta(days=30)
    # Adjust to Wednesday (3 is Wednesday)
    days_to_add = (3 - expiry.weekday()) % 7
    expiry = expiry + timedelta(days=days_to_add)
    
    return expiry

def price_vix_futures(data):
    """Price VIX futures contracts"""
    print("Pricing VIX Futures...")
    
    # Use the generic futures pricing function with VIX-specific parameters
    result = price_futures_contract(
        data=data,
        spot_column='vix_index',
        rate_column='fed_funds_rate',
        expiry_func=get_vix_expiry,
        storage_cost=0,  # No storage cost for volatility index
        convenience_yield=0,  # No convenience yield for VIX
        start_value=1000,
        output_prefix='vix_futures'
    )
    
    # Save results
    save_results(result, 'Derivatives/VIX_FrontMonth_futures', 'vix_futures.csv')
    
    return result

def get_crude_oil_expiry(date):
    """Calculate next expiry date for crude oil futures"""
    year = date.year
    month = date.month
    
    # Move to next month for expiry
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1
    
    # Crude oil futures expire on the 20th of each month
    expiry = datetime(year, month, 20)
    
    # Adjust for weekends
    while expiry.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        expiry = expiry - timedelta(days=1)
    
    return expiry

def price_crude_oil_futures(data):
    """Price 1-month crude oil futures contracts"""
    print("Pricing Crude Oil Futures...")
    
    # Use the generic futures pricing function
    result = price_futures_contract(
        data=data,
        spot_column='crude_oil_wti_spot',
        rate_column='fed_funds_rate',
        expiry_func=get_crude_oil_expiry,
        storage_cost=0.02/12,  # Monthly storage cost as fraction of spot
        convenience_yield=0.01/12,  # Monthly convenience yield
        start_value=1000,
        output_prefix='crude_oil_futures'
    )
    
    # Save results
    save_results(result, 'Derivatives/1M_Crude_Oil_Futures', 'crude_oil_futures.csv')
    
    return result

def get_gold_expiry(date):
    """Calculate next expiry date for gold futures"""
    year = date.year
    month = date.month
    
    # Gold futures expire quarterly (Mar, Jun, Sep, Dec)
    if month < 3:
        expiry_month = 3
    elif month < 6:
        expiry_month = 6
    elif month < 9:
        expiry_month = 9
    elif month < 12:
        expiry_month = 12
    else:  # If it's December, go to next year March
        expiry_month = 3
        year += 1
    
    # Gold futures expire on the third last business day of the month
    last_day = calendar.monthrange(year, expiry_month)[1]
    expiry = datetime(year, expiry_month, last_day)
    
    # Count back business days
    business_days_back = 0
    while business_days_back < 3:
        expiry = expiry - timedelta(days=1)
        if expiry.weekday() < 5:  # Not weekend
            business_days_back += 1
    
    return expiry

def price_gold_futures(data):
    """Price 3-month gold futures contracts"""
    print("Pricing Gold Futures...")
    
    # Use the generic futures pricing function
    result = price_futures_contract(
        data=data,
        spot_column='gold_spot_price',
        rate_column='fed_funds_rate',
        expiry_func=get_gold_expiry,
        storage_cost=0.01/4,  # Quarterly storage cost as fraction of spot
        convenience_yield=0.005/4,  # Quarterly convenience yield
        start_value=1000,
        output_prefix='gold_futures'
    )
    
    # Save results
    save_results(result, 'Derivatives/3M_Gold_Futures', 'gold_futures.csv')
    
    return result

def get_soybean_expiry(date):
    """Calculate next expiry date for soybean futures"""
    year = date.year
    month = date.month
    
    # Soybean futures typically have 6-month expirations
    # Move forward 6 months
    expiry_month = ((month - 1 + 6) % 12) + 1
    expiry_year = year + ((month + 6) > 12)
    
    # Soybean futures typically expire on the 15th of the expiry month
    expiry = datetime(expiry_year, expiry_month, 15)
    
    # Adjust for weekends
    while expiry.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        expiry = expiry + timedelta(days=1)
    
    return expiry

def price_soybean_futures(data):
    """Price 6-month soybean futures contracts"""
    print("Pricing Soybean Futures...")
    
    # Use the generic futures pricing function
    result = price_futures_contract(
        data=data,
        spot_column='soybean_spot_usd',
        rate_column='fed_funds_rate',
        expiry_func=get_soybean_expiry,
        storage_cost=0.03/2,  # Semi-annual storage cost as fraction of spot
        convenience_yield=0.02/2,  # Semi-annual convenience yield
        start_value=1000,
        output_prefix='soybean_futures'
    )
    
    # Save results
    save_results(result, 'Derivatives/6M_Soybean_Futures', 'soybean_futures.csv')
    
    return result

###############################################################################
#                              FOREX PRICING                                  #
###############################################################################

def price_forward_contract(data, base_column, domestic_rate_column, foreign_rate_column=None, 
                       expiry_func=None, tenor_days=180, start_value=1000, output_prefix='forward'):
    """Generic function to price forward contracts
    
    Args:
        data: DataFrame with market data
        base_column: Column name for the base rate (e.g., spot exchange rate)
        domestic_rate_column: Column name for domestic interest rate
        foreign_rate_column: Column name for foreign interest rate (None for non-FX forwards)
        expiry_func: Function to calculate expiry date (optional)
        tenor_days: Fixed tenor in days (used if expiry_func is None)
        start_value: Initial value for NAV calculation
        output_prefix: Prefix for output columns
        
    Returns:
        DataFrame with pricing results
    """
    print(f"Pricing {output_prefix} forward contract...")
    
    # Clean the input data
    columns_to_clean = [base_column, domestic_rate_column]
    if foreign_rate_column:
        columns_to_clean.append(foreign_rate_column)
    
    data_clean = clean_numeric_data(data, columns_to_clean)
    
    # Initialize result dataframe
    result = pd.DataFrame(index=data_clean.index)
    result['forward_price'] = np.nan
    result['days_to_expiry'] = np.nan
    result['roll_date'] = False
    
    # If expiry_func is not provided, use fixed tenor
    if expiry_func is None:
        def default_expiry_func(date):
            return date + timedelta(days=tenor_days)
        expiry_func = default_expiry_func
    
    # Calculate forwards with roll-over logic
    current_expiry = expiry_func(data_clean.index[0])
    
    for i, date in enumerate(data_clean.index):
        # Check if we need to roll over
        if date >= current_expiry:
            result.loc[date, 'roll_date'] = True
            current_expiry = expiry_func(date)
        
        # Calculate days to expiry
        days_to_expiry = (current_expiry - date).days
        result.loc[date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Forward pricing formula depends on the type
        if foreign_rate_column:  # FX forward
            # Convert rates from percentage to decimal if needed
            dom_rate = data_clean.loc[date, domestic_rate_column]
            for_rate = data_clean.loc[date, foreign_rate_column]
            
            # Use percentage rates (convert if they appear to be in decimal form)
            if dom_rate < 0.2:  # Assuming rates are not over 20%
                dom_rate *= 100
            if for_rate < 0.2:
                for_rate *= 100
                
            # Convert to decimal for calculation
            dom_rate = dom_rate / 100  
            for_rate = for_rate / 100
            
            # FX Forward = Spot * (1 + r_domestic)^T / (1 + r_foreign)^T
            spot = data_clean.loc[date, base_column]
            if not np.isnan(spot) and not np.isnan(dom_rate) and not np.isnan(for_rate):
                result.loc[date, 'forward_price'] = spot * (
                    (1 + dom_rate) ** time_to_maturity / 
                    (1 + for_rate) ** time_to_maturity
                )
        else:  # Non-FX forward (commodity, etc.)
            # Forward = Spot * (1 + r_domestic)^T
            spot = data_clean.loc[date, base_column]
            dom_rate = data_clean.loc[date, domestic_rate_column]
            
            # Use percentage rate (convert if it appears to be in decimal form)
            if dom_rate < 0.2:  # Assuming rates are not over 20%
                dom_rate *= 100
                
            # Convert to decimal for calculation
            dom_rate = dom_rate / 100
            
            if not np.isnan(spot) and not np.isnan(dom_rate):
                result.loc[date, 'forward_price'] = spot * (1 + dom_rate) ** time_to_maturity
    
    # Calculate returns with roll-over adjustment
    result['daily_returns'] = result['forward_price'].pct_change(fill_method=None)
    result['log_returns'] = np.log(result['forward_price'] / result['forward_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(result)):
        if result['roll_date'].iloc[i]:
            # On roll dates, we want returns to reflect the actual price change
            roll_yield = (result['forward_price'].iloc[i] - result['forward_price'].iloc[i-1]) / result['forward_price'].iloc[i-1]
            result.loc[result.index[i], 'daily_returns'] = roll_yield
            if result['forward_price'].iloc[i-1] > 0 and result['forward_price'].iloc[i] > 0:
                result.loc[result.index[i], 'log_returns'] = np.log(result['forward_price'].iloc[i] / result['forward_price'].iloc[i-1])
    
    # Compute NAV
    nav = pd.Series(index=result.index, name=f'{output_prefix}_nav')
    nav.iloc[0] = start_value
    
    for i in range(1, len(nav)):
        if not np.isnan(result['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + result['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Add NAV to the result
    result['NAV'] = nav
    
    return result

def get_gbpusd_expiry(date):
    """Calculate 6-month expiry date for GBP/USD forward contract"""
    # Standard 6-month tenor from the given date
    expiry = date + timedelta(days=180)
    
    # Adjust for weekends (forwards typically settle on business days)
    while expiry.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        expiry = expiry + timedelta(days=1)
    
    return expiry

def price_gbpusd_6m_forward(data):
    """Price 6-month GBP/USD forward contract"""
    print("Pricing GBP/USD 6-month forward contract...")
    
    # Use the generic forward pricing function
    result = price_forward_contract(
        data=data,
        base_column='GBP_USD',  # Spot exchange rate
        domestic_rate_column='fed_funds_rate',  # USD interest rate
        foreign_rate_column='UK_6M_rate',  # GBP interest rate
        expiry_func=get_gbpusd_expiry,
        start_value=1000,
        output_prefix='gbpusd_forward'
    )
    
    # Save results
    save_results(result, 'Forex/GBP_USD_6M_Forward', 'gbpusd_forward.csv')
    
    return result

def get_usdinr_expiry(date):
    """Calculate 3-month expiry date for USD/INR forward contract"""
    # Standard 3-month (91 days) tenor from the given date
    expiry = date + timedelta(days=91)
    
    # Adjust for weekends (forwards typically settle on business days)
    while expiry.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
        expiry = expiry + timedelta(days=1)
    
    return expiry

def price_usdinr_3m_forward(data):
    """Price 3-month USD/INR forward contract (short position)"""
    print("Pricing USD/INR 3-month forward contract (short position)...")
    
    # Use the generic forward pricing function with USD/INR-specific parameters
    result = price_forward_contract(
        data=data,
        base_column='USD_INR',  # Spot exchange rate
        domestic_rate_column='fed_funds_rate',  # USD interest rate
        foreign_rate_column='MIBOR ',  # INR interest rate (note the space)
        expiry_func=get_usdinr_expiry,
        tenor_days=91,  # 3 months
        start_value=1000,
        output_prefix='usdinr_forward'
    )
    
    # Save results
    save_results(result, 'Forex/INR_USD_3M_Forward_Short', 'usdinr_forward.csv')
    
    return result

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
            # Update the data columns
            data['vix_index'] = data['vix_index_level']
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
            # Update the data columns
            data['GBP_USD'] = data['fx_gbpusd_rate']
            data['UK_6M_rate'] = data['factor_GBP_sonia']
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