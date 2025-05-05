import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
import os
import calendar
from scipy.stats import norm  # Required for option pricing

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
    """Save results from a pricing function to a CSV file and create visualization"""
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
    
    # Plot NAV over time - Only if we're not running a batch job or if explicitly requested
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

def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2, notional=100):
    """Calculate bond price using the yield to maturity"""
    if years_to_maturity <= 0 or np.isnan(yield_rate):
        return np.nan
        
    coupon_payment = (coupon_rate / frequency) * notional
    periods = int(np.round(years_to_maturity * frequency))
    period_yield = yield_rate / frequency
    
    # Handle zero yield case
    if abs(period_yield) < 1e-10:
        coupon_pv = coupon_payment * periods
    else:
        # Calculate present value of coupon payments
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
    
    # Calculate present value of face value
    face_value_pv = notional / (1 + period_yield)**periods
    
    return coupon_pv + face_value_pv

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
                yield_rate=data.loc[current_date, '10y_treasury_yield'],
                coupon_rate=COUPON_RATE,
                years_to_maturity=time_to_maturity,
                frequency=FREQUENCY,
                notional=NOTIONAL
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
        """Calculate TIPS price using real yield and inflation adjustment"""
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
                yield_rate=data.loc[current_date, 'high_yield_rate'],
                coupon_rate=COUPON_RATE,
                years_to_maturity=time_to_maturity,
                frequency=FREQUENCY,
                notional=NOTIONAL
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
                frequency=FREQUENCY,
                notional=NOTIONAL
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
    FREQUENCY = 2       # Semi-annual payments
    MATURITY = 30       # 30 years
    REVENUE_SPREAD = 0.0075  # +75 bps spread over 30Y Treasuries
    
    # Clean the yield data
    data = clean_numeric_data(data, ['30Y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['30Y_treasury_yield'] = data['30Y_treasury_yield'] / 100
    
    # Initialize variables for bond pricing
    data['bond_price_revenue'] = np.nan
    data['days_to_maturity_revenue'] = 0
    data['roll_date_revenue'] = False
    
    # Calculate bond prices with roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(years=MATURITY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date_revenue'] = True
            current_maturity = current_date + pd.DateOffset(years=MATURITY)
        
        # Calculate days to maturity
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity_revenue'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate effective yield (Treasury + revenue spread)
        base_yield = data.loc[current_date, '30Y_treasury_yield']
        if not np.isnan(base_yield):
            effective_yield = base_yield + REVENUE_SPREAD
            data.loc[current_date, 'bond_price_revenue'] = calculate_bond_price(
                yield_rate=effective_yield,
                coupon_rate=COUPON_RATE,
                years_to_maturity=time_to_maturity,
                frequency=FREQUENCY,
                notional=NOTIONAL
            )
    
    # Calculate returns
    data['daily_returns_revenue'] = data['bond_price_revenue'].pct_change()
    data['log_returns_revenue'] = np.log(data['bond_price_revenue'] / data['bond_price_revenue'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date_revenue'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['bond_price_revenue'].iloc[i] - data['bond_price_revenue'].iloc[i-1]) / data['bond_price_revenue'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns_revenue'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='Revenue_Bond')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns_revenue'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns_revenue'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns_revenue'].values, index=data.index, name='Revenue_Bond')
    
    # Additional data for output
    additional_data = {
        'Bond_Price': data['bond_price_revenue'].values,
        'Days_to_Maturity': data['days_to_maturity_revenue'].values,
        'Roll_Date': data['roll_date_revenue'].values,
        'Yield_Treasury_30Y': data['30Y_treasury_yield'].values * 100,  # Convert back to percentage
        'Effective_Yield': (data['30Y_treasury_yield'] + REVENUE_SPREAD).values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['roll_date_revenue'].sum()),
        output_folder=output_folder,
        filename="30_year_revenue_bond_data.csv"
    )
    
    print(f"30-year Revenue Bond pricing complete. Number of roll-overs: {data['roll_date_revenue'].sum()}")
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
    nav = pd.Series(index=data.index, name='VIX_Futures_1M')
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
    
    # Save results using the standardized function
    output_df = save_results(
        result=(nav, data['log_returns'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="vix_futures_data.csv"
    )
    
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
    """Price Gold Futures and save results"""
    print("Pricing Gold Futures...")
    
    # Clean the data
    data = clean_numeric_data(data, ['gold_spot_price'])
    
    # Initialize variables
    data['gold_futures_roll_date'] = False
    data['days_to_expiry'] = 0
    
    # Set contract specifications
    CONTRACT_MONTHS = [2, 4, 6, 8, 10, 12]  # Feb, Apr, Jun, Aug, Oct, Dec
    CONTRACT_SIZE = 100  # troy ounces
    STORAGE_COST = 0.005  # 0.5% annual storage cost
    
    # Calculate pricing based on cost-of-carry model
    data['gold_futures_price'] = np.nan
    
    # Calculate days to expiry and identify roll dates
    expiry_date = None
    for i in range(len(data)):
        current_date = data.index[i]
        current_month = current_date.month
        current_year = current_date.year
        
        # Find the next contract month
        next_contract_month = None
        for month in CONTRACT_MONTHS:
            if month > current_month:
                next_contract_month = month
                break
        
        # If no future month found, use the first month of next year
        if next_contract_month is None:
            next_contract_month = CONTRACT_MONTHS[0]
            next_contract_year = current_year + 1
        else:
            next_contract_year = current_year
        
        # Calculate expiry date (third Friday of the contract month)
        if expiry_date is None or current_date >= expiry_date:
            # Mark as roll date if this is a new contract
            if expiry_date is not None:
                data.loc[current_date, 'gold_futures_roll_date'] = True
            
            # Calculate new expiry date
            first_day = pd.Timestamp(year=next_contract_year, month=next_contract_month, day=1)
            weekday = first_day.weekday()
            days_until_friday = (4 - weekday) % 7
            first_friday = first_day + pd.Timedelta(days=days_until_friday)
            expiry_date = first_friday + pd.Timedelta(days=14)  # Third Friday
        
        # Calculate days to expiry
        days_to_expiry = (expiry_date - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate futures price using cost-of-carry model
        if not np.isnan(data.loc[current_date, 'gold_spot_price']):
            time_to_expiry = days_to_expiry / 365.0  # Time in years
            
            # If fed_funds_rate is available, use it, otherwise use a default rate of 2%
            if 'fed_funds_rate' in data.columns and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
                interest_rate = data.loc[current_date, 'fed_funds_rate'] / 100
            else:
                interest_rate = 0.02  # Default 2%
                
            # Futures price = Spot price * e^((r + s) * T)
            # where r = interest rate, s = storage cost, T = time to expiry
            data.loc[current_date, 'gold_futures_price'] = data.loc[current_date, 'gold_spot_price'] * np.exp(
                (interest_rate + STORAGE_COST) * time_to_expiry
            )
    
    # Calculate daily returns
    data['daily_returns'] = data['gold_futures_price'].pct_change(fill_method=None)
    
    # Replace roll date returns with 0
    for i in range(1, len(data)):
        if data['gold_futures_roll_date'].iloc[i]:
            data.loc[data.index[i], 'daily_returns'] = 0
    
    # Calculate log returns
    data['log_returns'] = np.log(1 + data['daily_returns'])
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='Gold_Futures')
    nav.iloc[0] = 100  # Starting with $100
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns'].values, index=data.index, name='Gold_Futures')
    
    # Additional data for output
    additional_data = {
        'Futures_Price': data['gold_futures_price'].values,
        'Spot_Price': data['gold_spot_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['gold_futures_roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['gold_futures_roll_date'].sum()),
        output_folder=output_folder,
        filename="gold_futures.csv"
    )
    
    print(f"Gold Futures pricing complete. Number of roll-overs: {data['gold_futures_roll_date'].sum()}")
    return output_df

def price_soybean_futures(data, output_folder='Derivatives/6M_Soybean_Futures'):
    """Price 6-month soybean futures contracts"""
    print("Pricing Soybean Futures...")
    
    # Futures pricing parameters
    BASE_STORAGE_COST = 0.04     # 4% base annual storage cost
    SEASONAL_AMPLITUDE = 0.02    # 2% seasonal fluctuation
    CONVENIENCE_YIELD = 0.02     # 2% annual convenience yield
    START_VALUE = 100
    PHASE_SHIFT = 10             # Peak storage cost in October (month 10)
    
    # Function to get next expiry date for soybean futures
    def get_next_expiry(date):
        # Soybean futures typically expire in March, May, July, August, September, November
        expiry_months = [3, 5, 7, 8, 9, 11]
        current_year = date.year
        current_month = date.month
        
        # Find next expiry month
        next_month = None
        for month in expiry_months:
            if month > current_month:
                next_month = month
                break
        
        if next_month is None:
            next_month = expiry_months[0]
            current_year += 1
        
        # Set expiry to 15th of the month
        return datetime(current_year, next_month, 15)
    
    # Clean the soybean spot and interest rate data
    data = clean_numeric_data(data, ['soybean_spot_usd', 'fed_funds_rate'])
    
    # Convert interest rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Add month column for seasonality
    data['month'] = data.index.month
    
    # Calculate seasonal storage cost
    def seasonal_storage_cost(month):
        return BASE_STORAGE_COST + SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (month - PHASE_SHIFT) / 12)
    
    data['seasonal_storage_cost'] = data['month'].apply(seasonal_storage_cost)
    
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
        
        # Calculate futures price using the cost-of-carry model with seasonal storage cost
        if not np.isnan(data.loc[current_date, 'soybean_spot_usd']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'soybean_spot_usd'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] + 
                 data.loc[current_date, 'seasonal_storage_cost'] - 
                 CONVENIENCE_YIELD) * time_to_maturity
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
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns'].values, index=data.index, name='soybean_futures')
    
    # Additional data for output
    additional_data = {
        'Contract_Price': data['contract_price'].values,
        'Spot_Price': data['soybean_spot_usd'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values
    }
    
    # Save results
    output_df = save_results(
        result=(nav, log_returns, additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="soybean_futures.csv"
    )
    
    print(f"Soybean Futures pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, log_returns, additional_data, data['roll_date'].sum())

def price_costco_itm_call_option(data, output_folder='Derivatives/Costco_ITM_Call_option'):
    """Price 3-month ITM Call option on Costco stock"""
    print("Pricing Costco ITM Call Option...")
    
    # Constants
    NOTIONAL = 100
    MATURITY_DAYS = 90
    ANNUAL_BASIS = 365.0
    TRANSACTION_COST = 0.005  # 0.5% transaction cost on rolling
    
    def black_scholes_call(S, K, T, r, sigma, q):
        """Price European call using Black-Scholes-Merton"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    # Clean the required data
    cols = ['costco_stock_price', 'COST_IVOL_3MMA', 'fed_funds_rate', 'COST_DIV_YIELD']
    data = clean_numeric_data(data, cols)
    
    # Convert interest rates and volatility to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['COST_DIV_YIELD'] = data['COST_DIV_YIELD'] / 100
    data['COST_IVOL_3MMA'] = data['COST_IVOL_3MMA'] / 100
    
    # Initialize columns
    data['option_price'] = np.nan
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['strike_price'] = np.nan
    
    # Rolling logic
    start_date = data.index[0]
    current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
    current_strike = 0.9 * data.loc[start_date, 'costco_stock_price']  # Initial strike at 90% of spot
    
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Set strike price for the current date (fixed until next roll)
        data.loc[current_date, 'strike_price'] = current_strike
        
        # Check if we need to roll the option
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
            # Update strike price only on roll dates to 90% of current spot
            current_strike = 0.9 * data.loc[current_date, 'costco_stock_price']
        
        days_to_expiry = (current_expiry - current_date).days
        T = days_to_expiry / ANNUAL_BASIS
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        S = data.loc[current_date, 'costco_stock_price']
        sigma = data.loc[current_date, 'COST_IVOL_3MMA']
        r = data.loc[current_date, 'fed_funds_rate']
        q = data.loc[current_date, 'COST_DIV_YIELD']
        K = data.loc[current_date, 'strike_price']
    
        if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r) and not np.isnan(q):
            price = black_scholes_call(S, K, T, r, sigma, q)
            data.loc[current_date, 'option_price'] = price
    
    # Calculate returns
    data['daily_return'] = data['option_price'].pct_change()
    data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust returns on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            prev = data['option_price'].iloc[i - 1]  # Value of old option
            curr = data['option_price'].iloc[i]      # Value of new option
            if prev > 0:
                # Apply transaction cost when rolling the position
                roll_yield = (curr - prev) / prev - TRANSACTION_COST
                data.loc[data.index[i], 'daily_return'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='Costco_ITM_Call')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_return'].iloc[i]):
            nav.iloc[i] = nav.iloc[i - 1] * (1 + data['daily_return'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i - 1]
    
    # Additional data for output
    additional_data = {
        'Option_Price': data['option_price'].values,
        'Strike_Price': data['strike_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values,
        'Costco_Spot': data['costco_stock_price'].values,
        'Implied_Volatility': data['COST_IVOL_3MMA'].values * 100,
        'Fed_Funds_Rate': data['fed_funds_rate'].values * 100,
        'Dividend_Yield': data['COST_DIV_YIELD'].values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_return'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="costco_3m_itm_call_option_data.csv"
    )
    
    print(f"Costco ITM Call Option pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_return'], additional_data, data['roll_date'].sum())

def price_eurusd_atm_call_option(data, output_folder='Derivatives/EUR-USD_ATM_Call_Option'):
    """Price 1-month ATM Call option on EUR/USD"""
    print("Pricing EUR/USD ATM Call Option...")
    
    # Constants
    NOTIONAL = 100
    MATURITY_DAYS = 30
    ANNUAL_BASIS = 365.0
    
    def garman_kohlhagen_call(S, K, T, r_d, r_f, sigma):
        """Price a European FX Call option using Garman-Kohlhagen model"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        return call_price
    
    # Clean the required data
    cols = ['fx_eurusd_rate', 'EUR-USD_IVOL', 'fed_funds_rate', 'Euro_STR']
    data = clean_numeric_data(data, cols)
    
    # Convert interest rates and vol to decimals
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['Euro_STR'] = data['Euro_STR'] / 100
    data['EUR-USD_IVOL'] = data['EUR-USD_IVOL'] / 100
    
    # Initialize columns
    data['option_price'] = np.nan
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    
    # Rolling logic
    start_date = data.index[0]
    current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
    
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Roll if we reach expiry
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
        
        # Time to expiry
        days_to_expiry = (current_expiry - current_date).days
        T = days_to_expiry / ANNUAL_BASIS
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Get inputs
        S = data.loc[current_date, 'fx_eurusd_rate']
        sigma = data.loc[current_date, 'EUR-USD_IVOL']
        r_d = data.loc[current_date, 'fed_funds_rate']
        r_f = data.loc[current_date, 'Euro_STR']
        K = S  # ATM option
        
        # Price option if data is available
        if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r_d) and not np.isnan(r_f):
            price = garman_kohlhagen_call(S, K, T, r_d, r_f, sigma)
            data.loc[current_date, 'option_price'] = price
    
    # Compute returns
    data['daily_return'] = data['option_price'].pct_change()
    data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            previous = data['option_price'].iloc[i-1]
            current = data['option_price'].iloc[i]
            if previous > 0:
                roll_yield = (current - previous) / previous
                data.loc[data.index[i], 'daily_return'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='EURUSD_ATM_Call')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_return'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_return'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Option_Price': data['option_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values,
        'EURUSD_Spot': data['fx_eurusd_rate'].values,
        'Vol_1M': data['EUR-USD_IVOL'].values * 100,
        'USD_Rate': data['fed_funds_rate'].values * 100,
        'EUR_Rate': data['Euro_STR'].values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_return'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="1m_eurusd_call_option_data.csv"
    )
    
    print(f"EUR/USD ATM Call Option pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_return'], additional_data, data['roll_date'].sum())

def price_xom_itm_put_option(data, output_folder='Derivatives/Exxon_ITM_Put_option'):
    """Price 3-month ITM Put option on Exxon Mobil stock"""
    print("Pricing Exxon Mobil ITM Put Option...")
    
    # Constants
    NOTIONAL = 100
    MATURITY_DAYS = 90
    ANNUAL_BASIS = 365.0
    TRANSACTION_COST = 0.005  # 0.5% transaction cost on rolling
    
    def black_scholes_put(S, K, T, r, sigma, q):
        """Price European put using Black-Scholes-Merton"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    # Clean the required data
    cols = ['exxonmobil_stock_price', 'XOM_IVOL_3MMA', 'fed_funds_rate', 'XOM_DIV_YIELD']
    data = clean_numeric_data(data, cols)
    
    # Convert interest rates and volatility to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['XOM_DIV_YIELD'] = data['XOM_DIV_YIELD'] / 100
    data['XOM_IVOL_3MMA'] = data['XOM_IVOL_3MMA'] / 100
    
    # Initialize columns
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    data['roll_date'] = False
    data['days_to_expiry'] = 0
    
    # Rolling setup
    start_date = data.index[0]
    current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
    current_strike = 1.10 * data.loc[start_date, 'exxonmobil_stock_price']  # 110% ITM strike
    
    for i in range(len(data)):
        current_date = data.index[i]
        data.loc[current_date, 'strike_price'] = current_strike
    
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
            current_strike = 1.10 * data.loc[current_date, 'exxonmobil_stock_price']
        
        days_to_expiry = (current_expiry - current_date).days
        T = days_to_expiry / ANNUAL_BASIS
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        S = data.loc[current_date, 'exxonmobil_stock_price']
        sigma = data.loc[current_date, 'XOM_IVOL_3MMA']
        r = data.loc[current_date, 'fed_funds_rate']
        q = data.loc[current_date, 'XOM_DIV_YIELD']
        K = data.loc[current_date, 'strike_price']
        
        if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r) and not np.isnan(q):
            data.loc[current_date, 'option_price'] = black_scholes_put(S, K, T, r, sigma, q)
    
    # Returns
    data['daily_return'] = data['option_price'].pct_change()
    data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Apply transaction cost at roll
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            prev = data['option_price'].iloc[i - 1]
            curr = data['option_price'].iloc[i]
            if prev > 0:
                roll_yield = (curr - prev) / prev - TRANSACTION_COST
                data.loc[data.index[i], 'daily_return'] = roll_yield
    
    # NAV tracking
    nav = pd.Series(index=data.index, name='XOM_ITM_Put')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_return'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_return'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Additional data for output
    additional_data = {
        'Option_Price': data['option_price'].values,
        'Strike_Price': data['strike_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values,
        'ExxonMobil_Spot': data['exxonmobil_stock_price'].values,
        'Implied_Volatility': data['XOM_IVOL_3MMA'].values * 100,
        'Fed_Funds_Rate': data['fed_funds_rate'].values * 100,
        'Dividend_Yield': data['XOM_DIV_YIELD'].values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_return'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="xom_3m_itm_put_option_data.csv"
    )
    
    print(f"Exxon Mobil ITM Put Option pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_return'], additional_data, data['roll_date'].sum())

def price_usdjpy_atm_put_option(data, output_folder='Derivatives/USD-JPY_ATM_Put_Option'):
    """Price 3-month ATM Put option on USD/JPY"""
    print("Pricing USD/JPY ATM Put Option...")
    
    # Constants
    NOTIONAL = 100
    MATURITY_DAYS = 90
    ANNUAL_BASIS = 365.0
    
    def garman_kohlhagen_put(S, K, T, r_d, r_f, sigma):
        """Price a European FX Put option using Garman-Kohlhagen model (price in USD)"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        return put_price
    
    # Clean the required data
    cols = ['fx_usdjpy_rate', 'USD-JPY_IVOL', 'fed_funds_rate', 'Basic_Loan_Rate_JPY']
    data = clean_numeric_data(data, cols)
    
    # Convert interest rates and vol to decimals
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['Basic_Loan_Rate_JPY'] = data['Basic_Loan_Rate_JPY'] / 100
    data['USD-JPY_IVOL'] = data['USD-JPY_IVOL'] / 100
    
    # Initialize pricing columns
    data['option_price'] = np.nan
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    
    # Rolling logic
    start_date = data.index[0]
    current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
    
    for i in range(len(data)):
        current_date = data.index[i]
        
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
        
        days_to_expiry = (current_expiry - current_date).days
        T = days_to_expiry / ANNUAL_BASIS
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        S = data.loc[current_date, 'fx_usdjpy_rate']
        sigma = data.loc[current_date, 'USD-JPY_IVOL']
        r_d = data.loc[current_date, 'fed_funds_rate']
        r_f = data.loc[current_date, 'Basic_Loan_Rate_JPY']
        K = S  # ATM Put → Strike = Spot
        
        if not np.isnan(S) and not np.isnan(sigma) and not np.isnan(r_d) and not np.isnan(r_f):
            price = garman_kohlhagen_put(S, K, T, r_d, r_f, sigma)
            data.loc[current_date, 'option_price'] = price
    
    # Calculate returns
    data['daily_return'] = data['option_price'].pct_change()
    data['log_return'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust returns on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            prev = data['option_price'].iloc[i - 1]
            curr = data['option_price'].iloc[i]
            if prev > 0:
                roll_yield = (curr - prev) / prev
                data.loc[data.index[i], 'daily_return'] = roll_yield
    
    # Compute NAV
    nav = pd.Series(index=data.index, name='USDJPY_ATM_Put')
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_return'].iloc[i]):
            nav.iloc[i] = nav.iloc[i - 1] * (1 + data['daily_return'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i - 1]
    
    # Additional data for output
    additional_data = {
        'Option_Price_USD': data['option_price'].values,
        'Days_to_Expiry': data['days_to_expiry'].values,
        'Roll_Date': data['roll_date'].values,
        'USDJPY_Spot': data['fx_usdjpy_rate'].values,
        'Vol_3M': data['USD-JPY_IVOL'].values * 100,
        'USD_Rate': data['fed_funds_rate'].values * 100,
        'JPY_Rate': data['Basic_Loan_Rate_JPY'].values * 100
    }
    
    # Save results
    output_df = save_results(
        result=(nav, data['log_return'], additional_data, data['roll_date'].sum()),
        output_folder=output_folder,
        filename="3m_usdjpy_put_option_data.csv"
    )
    
    print(f"USD/JPY ATM Put Option pricing complete. Number of roll-overs: {data['roll_date'].sum()}")
    return (nav, data['log_return'], additional_data, data['roll_date'].sum())

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

def price_instrument(data, pricing_func, required_cols, instrument_name, output_dict, custom_message=None):
    """Helper function to price an instrument with standardized error handling and column checking"""
    try:
        # Check if all required columns are present
        if all(col in data.columns for col in required_cols):
            # Call pricing function
            results = pricing_func(data)
            output_dict[instrument_name] = results
        else:
            # Identify missing columns
            missing = [col for col in required_cols if col not in data.columns]
            if custom_message:
                print(custom_message)
            else:
                print(f"Skipping {instrument_name}: Required columns {missing} not found in dataset")
    except Exception as e:
        print(f"Error pricing {instrument_name}: {str(e)}")

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
    price_instrument(data, price_10y_treasury_bond, ['10y_treasury_yield'], '10y_Bond', pricing_results)
    
    # 2.2. LQD ETF
    price_instrument(data, price_lqd_etf, ['lqd_corporate_bond_etf'], 'LQD_ETF', pricing_results)
    
    # 2.3. 10-year TIPS
    price_instrument(data, price_10y_tips, ['Real_10Y_yield'], '10y_TIPS', pricing_results)
    
    # 2.4. 1-year EUR Zero Coupon Bond
    price_instrument(data, price_1y_eur_zcb, ['1_year_euro_yield_curve', 'fx_eurusd_rate'], 
                     '1y_EUR_ZCB', pricing_results)
    
    # 2.5. High Yield Corporate Debt
    price_instrument(data, price_high_yield_corp_debt, ['high_yield_credit spread', '10y_treasury_yield'], 
                    'High_Yield_Corp_Debt', pricing_results)
    
    # 2.6. 5-year Green Bond
    price_instrument(data, price_5y_green_bond, ['5y_treasury_yield'], '5y_Green_Bond', pricing_results)
    
    # 2.7. 30-year Revenue Bond
    price_instrument(data, price_30y_revenue_bond, ['30Y_treasury_yield'], '30y_Revenue_Bond', pricing_results)
    
    # 3. Price derivatives
    # 3.1. S&P 500 Futures (1 month)
    price_instrument(data, price_sp500_futures_1m, ['sp500_index', 'fed_funds_rate'], 
                    'SP500_Futures_1M', pricing_results)
    
    # 3.2. VIX Futures (front month)
    price_instrument(data, price_vix_futures, ['vix_index_level'], 'VIX_Futures', pricing_results)
    
    # 3.3. Crude Oil Futures (1 month)
    price_instrument(data, price_crude_oil_futures, ['crude_oil_wti_spot', 'fed_funds_rate'], 
                    'Crude_Oil_Futures_1M', pricing_results)
    
    # 3.4. Gold Futures (3 month)
    price_instrument(data, price_gold_futures, ['gold_spot_price', 'fed_funds_rate'], 
                    'Gold_Futures_3M', pricing_results)
    
    # 3.5. Soybean Futures (6 month)
    price_instrument(data, price_soybean_futures, ['soybean_spot_usd', 'fed_funds_rate'], 
                    'Soybean_Futures_6M', pricing_results)
    
    # 3.6. Costco ITM Call Option (3 month)
    price_instrument(data, price_costco_itm_call_option, 
                    ['costco_stock_price', 'COST_IVOL_3MMA', 'fed_funds_rate', 'COST_DIV_YIELD'], 
                    'Costco_ITM_Call', pricing_results)
    
    # 3.7. EUR/USD ATM Call Option (1 month)
    price_instrument(data, price_eurusd_atm_call_option, 
                    ['fx_eurusd_rate', 'EUR-USD_IVOL', 'fed_funds_rate', 'Euro_STR'], 
                    'EURUSD_ATM_Call', pricing_results)
    
    # 3.8. Exxon Mobil ITM Put Option (3 month)
    price_instrument(data, price_xom_itm_put_option, 
                    ['exxonmobil_stock_price', 'XOM_IVOL_3MMA', 'fed_funds_rate', 'XOM_DIV_YIELD'], 
                    'XOM_ITM_Put', pricing_results)
    
    # 3.9. USD/JPY ATM Put Option (3 month)
    price_instrument(data, price_usdjpy_atm_put_option, 
                    ['fx_usdjpy_rate', 'USD-JPY_IVOL', 'fed_funds_rate', 'Basic_Loan_Rate_JPY'], 
                    'USDJPY_ATM_Put', pricing_results)
    
    # 4. Price forex
    # 4.1. GBP/USD Forward (6 month)
    price_instrument(data, price_gbpusd_6m_forward, 
                    ['fx_gbpusd_rate', 'fed_funds_rate', 'factor_GBP_sonia'], 
                    'GBPUSD_Forward', pricing_results)
    
    # 4.2. USD/INR Forward (3 month, short position)
    price_instrument(data, price_usdinr_3m_forward, 
                    ['USD_INR', 'fed_funds_rate', 'MIBOR '], 
                    'USDINR_Forward', pricing_results)
    
    # Summary of successfully priced instruments
    print("\nSuccessfully priced instruments:")
    for instrument in pricing_results.keys():
        print(f"- {instrument}")
    
    print("\nAll pricing complete. Results saved to respective directories.")
    print("You can now run the Portfolio.py script to analyze the combined portfolio.")

if __name__ == "__main__":
    main() 