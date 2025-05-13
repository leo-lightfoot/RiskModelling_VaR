import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import os
import matplotlib.pyplot as plt

# Hardcoded paths
DATA_PATH = 'data_restructured.csv' # Path to the data file with Risk factors
COMBINED_RETURNS_PATH = 'combined_instrument_returns.csv' # Individual instrument returns
PORTFOLIO_RESULTS_DIR = 'portfolio_results' # Portfolio Directory
PORTFOLIO_NAV_HISTORY_PATH = 'portfolio_results/portfolio_nav_history.csv' #Portfolio NAV history
PORTFOLIO_RETURNS_HISTORY_PATH = 'portfolio_results/portfolio_returns_history.csv'
PORTFOLIO_PLOT_PATH = 'portfolio_results/portfolio_cumulative_nav_log.png'

#DATA LOADING AND CLEANING   

def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
        data.set_index('date', inplace=True)
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_numeric_data(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_nav_and_returns(prices, start_value=100):
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


# EQUITY PRICING
def price_equities(data):
    # Identify the equity columns
    equity_columns = [col for col in data.columns if col.startswith('eq_')]

    data = clean_numeric_data(data, equity_columns)
    results = {}
    
    # Process each equity
    for col in equity_columns:
        equity_name = col.replace('eq_', '')
        nav, log_returns = calculate_nav_and_returns(data[col])
        results[f"NAV_{equity_name}"] = nav
        results[f"LogReturn_{equity_name}"] = log_returns
    
    return results

# FIXED INCOME

# Bond Pricing Function
def calculate_bond_price(yield_rate, coupon_rate, years_to_maturity, frequency=2, notional=100):
    if years_to_maturity <= 0 or np.isnan(yield_rate):
        return np.nan
        
    coupon_payment = (coupon_rate / frequency) * notional
    periods = int(np.round(years_to_maturity * frequency))
    period_yield = yield_rate / frequency
    
    if abs(period_yield) < 1e-10:   # Handle zero yield case
        coupon_pv = coupon_payment * periods
    else: # Calculate present value of coupon payments
        coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield

    face_value_pv = notional / (1 + period_yield)**periods
    
    return coupon_pv + face_value_pv

def price_10y_treasury_bond(data): # 10Y Treasury Bond

    NOTIONAL = 100  
    COUPON_RATE = 0.02  # 2% annual
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 10  # 10 years
    
    data = clean_numeric_data(data, ['10y_treasury_yield'])
    data['10y_treasury_yield'] = data['10y_treasury_yield'] / 100
    
    #Initialize variables for bond pricing
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]

    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    
    return {'NAV_10y_treasury': nav, 'LogReturn_10y_treasury': log_returns}

def price_lqd_etf(data): # LQD Corporate Bond ETF
    lqd_column = 'lqd_corporate_bond_etf'
    data = clean_numeric_data(data, [lqd_column])
    nav, log_returns = calculate_nav_and_returns(data[lqd_column])
    log_returns_series = pd.Series(log_returns.values, index=data.index)
    return {'NAV_lqd_etf': nav, 'LogReturn_lqd_etf': log_returns_series}

def price_10y_tips(data):

    NOTIONAL = 100
    COUPON_RATE = 0.0125  # 1.25% annual
    FREQUENCY = 2  # Semiannual payments
    MATURITY = 10  # 10 years
    
    tips_column = 'Real_10Y_yield'
    data = clean_numeric_data(data, [tips_column, 'CPI'])
    data['Real_10Y_yield'] = data[tips_column] / 100
    data['monthly_inflation'] = data['CPI'] / 12 / 100
    data['monthly_inflation'] = data['monthly_inflation'].ffill()
    
    # Initialize variables for TIPS pricing
    data['tips_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    data['inflation_factor'] = np.nan
    
    def calculate_tips_price(real_yield, coupon_rate, years_to_maturity, frequency=2, inflation_factor=1.0):
        if years_to_maturity <= 0:
            return NOTIONAL * inflation_factor

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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not pd.isna(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]

    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    return {'NAV_10y_tips': nav, 'LogReturn_10y_tips': log_returns}

def price_1y_eur_zcb(data):
    """Price 1-year EUR Zero Coupon Bond and return NAV and log returns"""
    # Bond parameters
    INITIAL_USD = 100  # USD
    MATURITY = 1  # 1 year
    FREQUENCY = 1  # Annual payment (zero coupon)
    
    def calculate_zcb_price(yield_rate, years_to_maturity):
        """Calculate zero coupon bond price using continuous compounding"""
        return np.exp(-yield_rate * years_to_maturity)
    
    # Clean the 1-year EUR yield and FX data
    data = clean_numeric_data(data, ['1_year_euro_yield_curve', 'fx_eurusd_rate'])
    
    # Convert yield from percentage to decimal
    data['1_year_euro_yield_curve'] = data['1_year_euro_yield_curve'] / 100
    
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
    data['daily_returns_eur'] = data['zcb_price_eur'].pct_change()
    data['log_returns_eur'] = np.log(data['zcb_price_eur'] / data['zcb_price_eur'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['zcb_price_eur'].iloc[i] - data['zcb_price_eur'].iloc[i-1]) / data['zcb_price_eur'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns_eur'] = roll_yield
    
    # Compute NAV in EUR
    nav_eur = pd.Series(index=data.index)
    # Convert initial USD to EUR using first available FX rate
    initial_eur = INITIAL_USD / data['fx_eurusd_rate'].iloc[0]
    nav_eur.iloc[0] = initial_eur
    for i in range(1, len(nav_eur)):
        if not np.isnan(data['daily_returns_eur'].iloc[i]):
            nav_eur.iloc[i] = nav_eur.iloc[i-1] * (1 + data['daily_returns_eur'].iloc[i])
        else:
            nav_eur.iloc[i] = nav_eur.iloc[i-1]
    
    # Convert NAV to USD
    nav_usd = nav_eur * data['fx_eurusd_rate']
    
    # Calculate USD returns
    data['daily_returns_usd'] = nav_usd.pct_change()
    data['log_returns_usd'] = np.log(nav_usd / nav_usd.shift(1))
    
    return {'NAV_1y_eur_zcb': nav_usd, 'LogReturn_1y_eur_zcb': data['log_returns_usd']}

def price_high_yield_corp_debt(data):
    """Price High Yield Corporate Debt and return NAV and log returns"""
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_high_yield_corp_debt': nav, 'LogReturn_high_yield_corp_debt': data['log_returns']}

def price_5y_green_bond(data):
    """Price 5-year Green Bond and return NAV and log returns"""
    # Green bond parameters
    NOTIONAL = 100  # USD
    MATURITY = 5  # 5 years
    COUPON_RATE = 0.025  # 2.5% annual, typically somewhat lower than conventional bonds
    FREQUENCY = 2  # Semiannual coupon payments
    GREEN_PREMIUM = 0.0010  # 10 bps "greenium" - premium for green bonds
    
    # Clean the yield data
    data = clean_numeric_data(data, ['5y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100
    
    # Apply green premium (lower yield due to high demand for green bonds)
    data['green_bond_yield'] = data['5y_treasury_yield'] - GREEN_PREMIUM
    
    # Initialize variables for bond pricing
    data['bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with quarterly roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(months=3)  # Quarterly roll-over
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over (quarterly)
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(months=3)
        
        # Calculate days to maturity for the 5-year bond
        full_maturity = current_date + pd.DateOffset(years=MATURITY)
        days_to_maturity = (full_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate bond price with coupon payments
        if not np.isnan(data.loc[current_date, 'green_bond_yield']):
            data.loc[current_date, 'bond_price'] = calculate_bond_price(
                yield_rate=data.loc[current_date, 'green_bond_yield'],
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_5y_green_bond': nav, 'LogReturn_5y_green_bond': data['log_returns']}

def price_30y_revenue_bond(data):
    """Price 30-year Revenue Bond and return NAV and log returns"""
    # Revenue bond parameters
    NOTIONAL = 100  # USD
    MATURITY = 30  # 30 years
    COUPON_RATE = 0.035  # 3.5% annual
    FREQUENCY = 2  # Semiannual coupon payments
    CREDIT_SPREAD = 0.0050  # 50 bps spread over Treasury
    
    # Clean the yield data
    data = clean_numeric_data(data, ['30Y_treasury_yield'])
    
    # Convert yield from percentage to decimal
    data['30Y_treasury_yield'] = data['30Y_treasury_yield'] / 100
    
    # Apply credit spread
    data['revenue_bond_yield'] = data['30Y_treasury_yield'] + CREDIT_SPREAD
    
    # Initialize variables for bond pricing
    data['bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with semiannual roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(months=6)  # Semiannual roll-over
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over (semiannually)
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(months=6)
        
        # Calculate days to maturity for the 30-year bond
        full_maturity = current_date + pd.DateOffset(years=MATURITY)
        days_to_maturity = (full_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_maturity / 365.0
        
        # Calculate bond price with coupon payments
        if not np.isnan(data.loc[current_date, 'revenue_bond_yield']):
            data.loc[current_date, 'bond_price'] = calculate_bond_price(
                yield_rate=data.loc[current_date, 'revenue_bond_yield'],
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_30y_revenue_bond': nav, 'LogReturn_30y_revenue_bond': data['log_returns']}

###############################################################################
#                          DERIVATIVES PRICING                                #
###############################################################################

def price_sp500_futures_1m(data):
    """Price 1-month S&P 500 futures contracts and return NAV and log returns"""
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_sp500_futures_1m': nav, 'LogReturn_sp500_futures_1m': data['log_returns']}

def price_vix_futures(data):
    """Price VIX futures contracts and return NAV and log returns"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
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
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_vix_futures': nav, 'LogReturn_vix_futures': data['log_returns']}

def price_crude_oil_futures(data):
    """Price 1-month crude oil futures contracts and return NAV and log returns"""
    # Futures pricing parameters
    STORAGE_COST = 0.005  # Monthly storage cost as fraction of price
    START_VALUE = 100
    
    # Function to get next expiry date
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
    
    # Clean the crude oil spot and Fed funds rate data
    data = clean_numeric_data(data, ['crude_oil_wti_spot', 'fed_funds_rate'])
    
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
        if not np.isnan(data.loc[current_date, 'crude_oil_wti_spot']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            # Apply contango - crude oil typically has higher futures prices due to storage costs
            contango_factor = (1 + data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST) ** time_to_maturity
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'crude_oil_wti_spot'] * contango_factor
    
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_crude_oil_futures': nav, 'LogReturn_crude_oil_futures': data['log_returns']}

def price_gold_futures(data):
    """Price 3-month gold futures contracts and return NAV and log returns"""
    # Futures pricing parameters
    STORAGE_COST = 0.0005  # Monthly storage cost as fraction of price (lower for gold)
    START_VALUE = 100
    CONTRACT_MONTHS = 3  # 3-month contract
    
    # Function to get next expiry date for 3-month contract
    def get_next_expiry(date):
        # Gold futures expire in Feb, Apr, Jun, Aug, Oct, Dec
        # For simplicity, just add 3 months
        return date + pd.DateOffset(months=CONTRACT_MONTHS)
    
    # Clean the gold spot and Fed funds rate data
    data = clean_numeric_data(data, ['gold_spot_price', 'fed_funds_rate'])
    
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
        if not np.isnan(data.loc[current_date, 'gold_spot_price']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            # Gold futures typically trade in contango due to storage costs
            contango_factor = (1 + data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST) ** time_to_maturity
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'gold_spot_price'] * contango_factor
    
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_gold_futures': nav, 'LogReturn_gold_futures': data['log_returns']}

def price_soybean_futures(data):
    """Price 6-month soybean futures contracts and return NAV and log returns"""
    # Futures pricing parameters
    START_VALUE = 100
    CONTRACT_MONTHS = 6  # 6-month contract
    
    # Function to get next expiry date
    def get_next_expiry(date):
        # Soybean futures typically expire in March, May, July, August, September, November
        # For simplicity, just add 6 months
        return date + pd.DateOffset(months=CONTRACT_MONTHS)
    
    # Function to calculate seasonal storage cost based on month
    def seasonal_storage_cost(month):
        # Harvest season (Oct-Nov) has lower storage costs
        # Pre-harvest season (Jul-Sept) has higher storage costs
        if month in [10, 11]:  # Harvest season
            return 0.0015  # Lower storage cost (0.15% per month)
        elif month in [7, 8, 9]:  # Pre-harvest
            return 0.0040  # Higher storage cost (0.40% per month)
        else:
            return 0.0025  # Standard storage cost (0.25% per month)
    
    # Clean the soybean spot and Fed funds rate data
    data = clean_numeric_data(data, ['soybean_spot_usd', 'fed_funds_rate'])
    
    # Convert Fed funds rate from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['contract_price'] = np.nan
    data['storage_cost'] = data.index.month.map(seasonal_storage_cost)
    
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
        if not np.isnan(data.loc[current_date, 'soybean_spot_usd']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            # Soybeans typically show seasonal patterns in futures markets
            contango_factor = (1 + data.loc[current_date, 'fed_funds_rate'] + data.loc[current_date, 'storage_cost']) ** time_to_maturity
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'soybean_spot_usd'] * contango_factor
    
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = START_VALUE
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_soybean_futures': nav, 'LogReturn_soybean_futures': data['log_returns']}

def price_costco_itm_call_option(data):
    """Price Costco ITM Call Option (3 month) and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    STRIKE_RATIO = 0.95  # ITM by 5%
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    TRANSACTION_COST = 0.0010  # 10 bps per transaction
    
    # Black-Scholes formula for call option pricing
    def black_scholes_call(S, K, T, r, sigma, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(0.0001, call)  # Ensure no zero values for log returns
    
    # Clean the data
    data = clean_numeric_data(data, ['costco_stock_price', 'COST_IVOL_3MMA', 'fed_funds_rate', 'COST_DIV_YIELD'])
    
    # Convert rate data from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['COST_DIV_YIELD'] = data['COST_DIV_YIELD'] / 100
    data['COST_IVOL_3MMA'] = data['COST_IVOL_3MMA'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    
    # Calculate option prices with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and stock price
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'costco_stock_price']):
            continue
        
        # Calculate strike price (ITM by 5%)
        spot = data.loc[current_date, 'costco_stock_price']
        strike = spot * STRIKE_RATIO
        data.loc[current_date, 'strike_price'] = strike
        
        # Calculate option price using Black-Scholes
        if not np.isnan(data.loc[current_date, 'costco_stock_price']) and not np.isnan(data.loc[current_date, 'COST_IVOL_3MMA']):
            data.loc[current_date, 'option_price'] = black_scholes_call(
                S=data.loc[current_date, 'costco_stock_price'],
                K=strike,
                T=time_to_maturity,
                r=data.loc[current_date, 'fed_funds_rate'],
                sigma=data.loc[current_date, 'COST_IVOL_3MMA'],
                q=data.loc[current_date, 'COST_DIV_YIELD']
            )
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate return with transaction costs
            data.loc[data.index[i], 'daily_returns'] = data.loc[data.index[i], 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_costco_itm_call': nav, 'LogReturn_costco_itm_call': data['log_returns']}

def price_eurusd_atm_call_option(data):
    """Price EUR/USD ATM Call Option (1 month) and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    DAYS_TO_EXPIRY = 30  # Approximately 1 month
    TRANSACTION_COST = 0.0005  # 5 bps per transaction
    
    # Garman-Kohlhagen formula for FX call option pricing (extension of Black-Scholes for FX options)
    def garman_kohlhagen_call(S, K, T, r_d, r_f, sigma):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        return max(0.0001, call)  # Ensure no zero values for log returns
    
    # Clean the data
    data = clean_numeric_data(data, ['fx_eurusd_rate', 'EUR-USD_IVOL', 'fed_funds_rate', 'Euro_STR'])
    
    # Convert rate data from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100  # USD interest rate
    data['Euro_STR'] = data['Euro_STR'] / 100  # EUR interest rate
    data['EUR-USD_IVOL'] = data['EUR-USD_IVOL'] / 100  # Implied volatility
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    
    # Calculate option prices with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and FX rate
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'fx_eurusd_rate']):
            continue
        
        # Calculate strike price (ATM)
        spot = data.loc[current_date, 'fx_eurusd_rate']
        strike = spot  # ATM strike = current spot
        data.loc[current_date, 'strike_price'] = strike
        
        # Calculate option price using Garman-Kohlhagen
        if not np.isnan(data.loc[current_date, 'fx_eurusd_rate']) and not np.isnan(data.loc[current_date, 'EUR-USD_IVOL']):
            data.loc[current_date, 'option_price'] = garman_kohlhagen_call(
                S=data.loc[current_date, 'fx_eurusd_rate'],
                K=strike,
                T=time_to_maturity,
                r_d=data.loc[current_date, 'fed_funds_rate'],  # Domestic (USD) rate
                r_f=data.loc[current_date, 'Euro_STR'],  # Foreign (EUR) rate
                sigma=data.loc[current_date, 'EUR-USD_IVOL']
            )
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate return with transaction costs
            data.loc[data.index[i], 'daily_returns'] = data.loc[data.index[i], 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_eurusd_atm_call': nav, 'LogReturn_eurusd_atm_call': data['log_returns']}

def price_xom_itm_put_option(data):
    """Price Exxon Mobil ITM Put Option (3 month) and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    STRIKE_RATIO = 1.05  # ITM by 5% for put option
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    TRANSACTION_COST = 0.0010  # 10 bps per transaction
    
    # Black-Scholes formula for put option pricing
    def black_scholes_put(S, K, T, r, sigma, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return max(0.0001, put)  # Ensure no zero values for log returns
    
    # Clean the data
    data = clean_numeric_data(data, ['exxonmobil_stock_price', 'XOM_IVOL_3MMA', 'fed_funds_rate', 'XOM_DIV_YIELD'])
    
    # Convert rate data from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['XOM_DIV_YIELD'] = data['XOM_DIV_YIELD'] / 100
    data['XOM_IVOL_3MMA'] = data['XOM_IVOL_3MMA'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    
    # Calculate option prices with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and stock price
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'exxonmobil_stock_price']):
            continue
        
        # Calculate strike price (ITM by 5% for put)
        spot = data.loc[current_date, 'exxonmobil_stock_price']
        strike = spot * STRIKE_RATIO
        data.loc[current_date, 'strike_price'] = strike
        
        # Calculate option price using Black-Scholes
        if not np.isnan(spot) and not np.isnan(data.loc[current_date, 'XOM_IVOL_3MMA']):
            data.loc[current_date, 'option_price'] = black_scholes_put(
                S=spot,
                K=strike,
                T=time_to_maturity,
                r=data.loc[current_date, 'fed_funds_rate'],
                sigma=data.loc[current_date, 'XOM_IVOL_3MMA'],
                q=data.loc[current_date, 'XOM_DIV_YIELD']
            )
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate return with transaction costs
            data.loc[data.index[i], 'daily_returns'] = data.loc[data.index[i], 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_xom_itm_put': nav, 'LogReturn_xom_itm_put': data['log_returns']}

def price_usdjpy_atm_put_option(data):
    """Price USD/JPY ATM Put Option (3 month) and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    TRANSACTION_COST = 0.0005  # 5 bps per transaction
    
    # Garman-Kohlhagen formula for FX put option pricing
    def garman_kohlhagen_put(S, K, T, r_d, r_f, sigma):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        return max(1.0, put)  # Set lower bound to 1.0
    
    # Clean the data
    data = clean_numeric_data(data, ['fx_usdjpy_rate', 'USD-JPY_IVOL', 'fed_funds_rate', 'Basic_Loan_Rate_JPY'])
    
    # Convert rate data from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100  # USD interest rate
    data['Basic_Loan_Rate_JPY'] = data['Basic_Loan_Rate_JPY'] / 100  # JPY interest rate
    data['USD-JPY_IVOL'] = data['USD-JPY_IVOL'] / 100  # Implied volatility
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    
    # Calculate option prices with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and FX rate
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'fx_usdjpy_rate']):
            continue
        
        # Calculate strike price (ATM)
        spot = data.loc[current_date, 'fx_usdjpy_rate']
        strike = spot  # ATM strike = current spot
        data.loc[current_date, 'strike_price'] = strike
        
        # Calculate option price using Garman-Kohlhagen
        if not np.isnan(spot) and not np.isnan(data.loc[current_date, 'USD-JPY_IVOL']):
            data.loc[current_date, 'option_price'] = garman_kohlhagen_put(
                S=spot,
                K=strike,
                T=time_to_maturity,
                r_d=data.loc[current_date, 'fed_funds_rate'],  # Domestic (USD) rate
                r_f=data.loc[current_date, 'Basic_Loan_Rate_JPY'],  # Foreign (JPY) rate
                sigma=data.loc[current_date, 'USD-JPY_IVOL']
            )
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate return with transaction costs
            data.loc[data.index[i], 'daily_returns'] = data.loc[data.index[i], 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_usdjpy_atm_put': nav, 'LogReturn_usdjpy_atm_put': data['log_returns']}

def price_variance_swap_dax(data):
    """Price DAX 30-day Variance Swap (fixed leg) and return NAV and log returns"""
    # Variance swap parameters
    NOTIONAL = 100
    DAYS_TO_EXPIRY = 30  # 30-day variance swap
    VARIANCE_STRIKE_ADJUSTMENT = 0.98  # Variance strike is typically set below fair value
    
    # Clean the DAX volatility data
    data = clean_numeric_data(data, ['DAX_Call_ivol_30D', 'DAX_Put_ivol_30D'])
    
    # Convert volatility from percentage to decimal
    data['DAX_Call_ivol_30D'] = data['DAX_Call_ivol_30D'] / 100
    data['DAX_Put_ivol_30D'] = data['DAX_Put_ivol_30D'] / 100
    
    # Calculate implied variance using average of call and put implied volatilities
    data['implied_variance'] = (data['DAX_Call_ivol_30D']**2 + data['DAX_Put_ivol_30D']**2) / 2
    
    # Initialize variables for swap pricing
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['variance_strike'] = np.nan
    data['pv_variance_leg'] = np.nan
    
    # Calculate variance swap values with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Set variance strike at the beginning of each swap period
        if data.loc[current_date, 'roll_date'] or i == 0:
            # Variance strike is set slightly below the implied variance (typical market practice)
            data.loc[current_date, 'variance_strike'] = data.loc[current_date, 'implied_variance'] * VARIANCE_STRIKE_ADJUSTMENT
        else:
            # Keep using the same variance strike until expiry
            if i > 0:
                data.loc[current_date, 'variance_strike'] = data.loc[data.index[i-1], 'variance_strike']
        
        # Calculate present value of fixed leg (we pay fixed, so positive value means profit)
        if not np.isnan(data.loc[current_date, 'variance_strike']) and not np.isnan(data.loc[current_date, 'implied_variance']):
            # PV of fixed leg = (implied variance - variance strike) * notional * remaining time
            data.loc[current_date, 'pv_variance_leg'] = (data.loc[current_date, 'implied_variance'] - 
                                                       data.loc[current_date, 'variance_strike']) * time_to_maturity * NOTIONAL
    
    # Calculate returns based on changes in PV
    data['daily_pnl'] = data['pv_variance_leg'].diff()
    data['daily_returns'] = data['daily_pnl'] / NOTIONAL
    data['log_returns'] = np.log(1 + data['daily_returns'])
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_dax_variance_swap': nav, 'LogReturn_dax_variance_swap': data['log_returns']}

def price_asian_put_option(data):
    """Price Nikkei 3-month Asian Put Option and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    STRIKE_RATIO = 1.02  # ITM by 2% for put
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    AVERAGING_FREQUENCY = 5  # Average price every 5 trading days
    TRANSACTION_COST = 0.0010  # 10 bps per transaction
    NUM_SIMULATIONS = 10000  # Number of simulations for Monte Carlo
    
    # Monte Carlo simulation for Asian put option pricing
    def price_asian_put_mc(S0, K, r, q, sigma, T, steps=63, paths=10000):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        dt = T / steps
        nudt = (r - q - 0.5 * sigma**2) * dt
        volsqrtdt = sigma * np.sqrt(dt)
        
        # Initialize price paths
        S = np.zeros((paths, steps+1))
        S[:, 0] = S0
        
        # Generate random paths
        Z = np.random.standard_normal((paths, steps))
        for t in range(1, steps+1):
            S[:, t] = S[:, t-1] * np.exp(nudt + volsqrtdt * Z[:, t-1])
        
        # Calculate average price path (every AVERAGING_FREQUENCY days)
        avg_days = np.arange(0, steps+1, AVERAGING_FREQUENCY)
        avg_prices = np.mean(S[:, avg_days], axis=1)
        
        # Calculate payoff for put option
        put_payoffs = np.maximum(K - avg_prices, 0)
        
        # Discount payoffs to present value
        price = np.exp(-r * T) * np.mean(put_payoffs)
        
        return max(0.0001, price)  # Ensure no zero values for log returns
    
    # Clean the data
    data = clean_numeric_data(data, ['Nikkei_spot', 'NKY_30D_ivol', 'NKY_Div_yield', 'Basic_Loan_Rate_JPY'])
    
    # Convert rate data from percentage to decimal
    data['NKY_30D_ivol'] = data['NKY_30D_ivol'] / 100
    data['NKY_Div_yield'] = data['NKY_Div_yield'] / 100
    data['Basic_Loan_Rate_JPY'] = data['Basic_Loan_Rate_JPY'] / 100
    
    # Initialize variables for roll-over logic
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    
    # Calculate option prices with roll-over logic
    next_expiry = data.index[0] + pd.DateOffset(days=DAYS_TO_EXPIRY)
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= next_expiry:
            data.loc[current_date, 'roll_date'] = True
            next_expiry = current_date + pd.DateOffset(days=DAYS_TO_EXPIRY)
        
        # Calculate days to expiry
        days_to_expiry = (next_expiry - current_date).days
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and stock price
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'Nikkei_spot']):
            continue
        
        # Calculate strike price (ITM by 2% for put)
        spot = data.loc[current_date, 'Nikkei_spot']
        strike = spot * STRIKE_RATIO
        data.loc[current_date, 'strike_price'] = strike
        
        # Calculate option price using Monte Carlo simulation
        if not np.isnan(spot) and not np.isnan(data.loc[current_date, 'NKY_30D_ivol']):
            data.loc[current_date, 'option_price'] = price_asian_put_mc(
                S0=spot,
                K=strike,
                r=data.loc[current_date, 'Basic_Loan_Rate_JPY'],
                q=data.loc[current_date, 'NKY_Div_yield'],
                sigma=data.loc[current_date, 'NKY_30D_ivol'],
                T=time_to_maturity,
                steps=int(days_to_expiry / 5) + 1,  # Number of averaging points
                paths=NUM_SIMULATIONS
            )
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates to account for transaction costs
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate return with transaction costs
            data.loc[data.index[i], 'daily_returns'] = data.loc[data.index[i], 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_nikkei_asian_put': nav, 'LogReturn_nikkei_asian_put': data['log_returns']}

def price_ford_cds(data):
    """Price Ford 5-year CDS and return NAV and log returns"""
    # CDS parameters
    NOTIONAL = 100
    MATURITY = 5  # 5-year CDS
    COUPON = 0.02  # 2% annual coupon (200 bps)
    RECOVERY_RATE = 0.40  # 40% recovery rate
    TRANSACTION_COST = 0.0005  # 5 bps per transaction
    
    # Function to calculate CDS value
    def cds_value(spread, hazard_rate, r, maturity):
        dt = 0.25  # Quarterly payments
        periods = int(maturity / dt)
        
        # Calculate premium leg (what protection buyer pays)
        premium_leg = 0
        for i in range(1, periods+1):
            t = i * dt
            survival_prob = np.exp(-hazard_rate * t)
            discount_factor = np.exp(-r * t)
            premium_leg += survival_prob * discount_factor * dt
        
        # Calculate protection leg (what protection seller pays upon default)
        protection_leg = 0
        for i in range(1, periods+1):
            t_before = (i-1) * dt
            t_after = i * dt
            survival_before = np.exp(-hazard_rate * t_before)
            survival_after = np.exp(-hazard_rate * t_after)
            default_prob = survival_before - survival_after
            mid_point = (t_before + t_after) / 2
            discount_factor = np.exp(-r * mid_point)
            protection_leg += (1 - RECOVERY_RATE) * default_prob * discount_factor
        
        # Value of CDS from protection seller perspective
        # spread - market coupon, COUPON - contract coupon
        return protection_leg - premium_leg * (spread / COUPON)
    
    # Clean the CDS spread and Treasury yield data
    data = clean_numeric_data(data, ['5_Y_ford_credit_spread', '5y_treasury_yield'])
    
    # Convert rates from percentage to decimal
    data['5_Y_ford_credit_spread'] = data['5_Y_ford_credit_spread'] / 100
    data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100
    
    # Initialize variables for CDS valuation
    data['cds_value'] = np.nan
    data['hazard_rate'] = np.nan
    data['days_to_maturity'] = MATURITY * 365
    
    # Calculate CDS values
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Calculate hazard rate from spread
        if not np.isnan(data.loc[current_date, '5_Y_ford_credit_spread']):
            # Approximate hazard rate from spread and recovery rate
            spread = data.loc[current_date, '5_Y_ford_credit_spread']
            hazard_rate = spread / (1 - RECOVERY_RATE)
            data.loc[current_date, 'hazard_rate'] = hazard_rate
            
            # Calculate CDS value
            if not np.isnan(data.loc[current_date, '5y_treasury_yield']):
                r = data.loc[current_date, '5y_treasury_yield']
                data.loc[current_date, 'cds_value'] = cds_value(spread, hazard_rate, r, MATURITY)
    
    # Calculate returns based on changes in CDS value
    data['daily_pnl'] = data['cds_value'].diff()
    data['daily_returns'] = data['daily_pnl'] / NOTIONAL
    data['log_returns'] = np.log(1 + data['daily_returns'])
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_ford_cds': nav, 'LogReturn_ford_cds': data['log_returns']}

def price_spx_barrier_option(data):
    """Price S&P 500 Knock-Out Call Option and return NAV and log returns"""
    # Option parameters
    NOTIONAL = 100
    MATURITY_DAYS = 30  # One month
    ANNUAL_BASIS = 365.0
    TRANSACTION_COST = 0.003  # 0.3%
    BARRIER_MULTIPLIER = 1.1  # 110% knock-out level
    MIN_OPTION_VALUE = 1e-6   # Minimum option value to prevent numerical issues
    MIN_NAV = 1.0  # Set a minimum NAV to avoid extremely small values in log scale
    
    # Black-Scholes formula for call option pricing
    def black_scholes_call(S, K, T, r, sigma, q):
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def monte_carlo_barrier_call(S, K, T, r, sigma, q, B, n_paths=10000, n_steps=21):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        dt = T / n_steps
        disc = np.exp(-r * T)
        S_paths = np.full((n_paths, n_steps + 1), S)
        barrier_breached = np.zeros(n_paths, dtype=bool)
        for t in range(1, n_steps + 1):
            z = np.random.normal(size=n_paths)
            S_paths[:, t] = S_paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            barrier_breached |= (S_paths[:, t] >= B)
        payoffs = np.where(barrier_breached, 0.0, np.maximum(S_paths[:, -1] - K, 0))
        return disc * np.mean(payoffs)
    
    # Clean the data
    data = clean_numeric_data(data, ['sp500_index', 'SPX_Div_yield', 'vix_index_level', 'fed_funds_rate'])
    
    # Convert rate data from percentage to decimal
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['SPX_Div_yield'] = data['SPX_Div_yield'] / 100
    data['vix_index_level'] = data['vix_index_level'] / 100  # VIX as volatility input
    
    # Initialize variables for option pricing
    data['days_to_expiry'] = 0
    data['roll_date'] = False
    data['option_price'] = np.nan
    data['strike_price'] = np.nan
    data['barrier_level'] = np.nan
    data['knocked_out'] = False
    
    # Rolling logic
    start_date = data.index[0]
    current_expiry = start_date + pd.Timedelta(days=MATURITY_DAYS)
    current_strike = data.loc[start_date, 'sp500_index']
    current_barrier = BARRIER_MULTIPLIER * current_strike
    
    for i in range(len(data)):
        current_date = data.index[i]
        spot = data.loc[current_date, 'sp500_index']
        
        data.loc[current_date, 'strike_price'] = current_strike
        data.loc[current_date, 'barrier_level'] = current_barrier
        
        if current_date >= current_expiry:
            data.loc[current_date, 'roll_date'] = True
            current_expiry = current_date + pd.Timedelta(days=MATURITY_DAYS)
            current_strike = data.loc[current_date, 'sp500_index']
            current_barrier = BARRIER_MULTIPLIER * current_strike
            # Reset the knocked out flag when rolling to a new option
            data.loc[current_date, 'knocked_out'] = False
        
        days_to_expiry = (current_expiry - current_date).days
        T = days_to_expiry / ANNUAL_BASIS
        data.loc[current_date, 'days_to_expiry'] = days_to_expiry
        
        sigma = data.loc[current_date, 'vix_index_level']
        r = data.loc[current_date, 'fed_funds_rate']
        q = data.loc[current_date, 'SPX_Div_yield']
        K = data.loc[current_date, 'strike_price']
        B = data.loc[current_date, 'barrier_level']
        
        if spot >= B:
            data.loc[current_date, 'option_price'] = 0.0
            data.loc[current_date, 'knocked_out'] = True
        elif not np.isnan(spot) and not np.isnan(sigma) and not np.isnan(r):
            price = monte_carlo_barrier_call(spot, K, T, r, sigma, q, B)
            data.loc[current_date, 'option_price'] = price
        
        # Add a step to propagate knocked_out status for the same option until roll date
        if i > 0 and not data.loc[current_date, 'roll_date'] and data.loc[data.index[i-1], 'knocked_out']:
            data.loc[current_date, 'knocked_out'] = True
            data.loc[current_date, 'option_price'] = 0.0
    
    # Calculate returns
    data['daily_returns'] = data['option_price'].pct_change()
    data['log_returns'] = np.log(data['option_price'] / data['option_price'].shift(1))
    
    # Adjust return on roll dates
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            prev = data['option_price'].iloc[i - 1]
            curr = data['option_price'].iloc[i]
            if prev > 0:
                data.loc[data.index[i], 'daily_returns'] = (curr - prev) / prev - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    
    for i in range(1, len(nav)):
        if data['roll_date'].iloc[i]:
            # Reset NAV to NOTIONAL on every roll date to avoid astronomic growth
            nav.iloc[i] = NOTIONAL
        else:
            ret = data['daily_returns'].iloc[i]
            new_val = nav.iloc[i - 1] * (1 + ret) if not np.isnan(ret) else nav.iloc[i - 1]
            nav.iloc[i] = max(new_val, MIN_NAV)  # Apply minimum value to avoid extremely small numbers
    
    return {'NAV_spx_knockout_call': nav, 'LogReturn_spx_knockout_call': data['log_returns']}

###############################################################################
#                              FOREX PRICING                                  #
###############################################################################

def price_gbpusd_6m_forward(data):
    """Price 6-month GBP/USD forward contract and return NAV and log returns"""
    # Forward contract parameters
    NOTIONAL_USD = 100.0  # Notional amount in USD
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
                'expiry_date': get_next_expiry(current_date),
                'forward_rate': forward_rate,
                'notional_usd': NOTIONAL_USD
            }
            
            # Mark this row as an active contract entry point
            data_fx.loc[current_date, 'active_contract'] = True
        
        # Update data for the current date
        data_fx.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
        data_fx.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
        
        # Calculate unrealized P&L (mark-to-market)
        # For a GBP/USD forward, buying GBP and selling USD:
        # P&L = Notional_USD * (1/spot - 1/forward_rate)
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_USD * (1/spot - 1/forward_rate)
        data_fx.loc[current_date, 'contract_pnl'] = pnl
    
    # Calculate daily returns based on P&L changes
    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_USD
    if len(data_fx) > 0:
        data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    
    # Fill NaN values in daily returns with zeros (for roll dates)
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav = pd.Series(index=data_fx.index)
    if len(nav) > 0:
        nav.iloc[0] = 100  # Start with 100
        for i in range(1, len(nav)):
            prev_nav = nav.iloc[i-1]
            daily_return = data_fx['daily_return'].iloc[i]
            nav.iloc[i] = prev_nav * (1 + daily_return)
    
    # Compute log returns of NAV
    log_returns = np.log(nav / nav.shift(1))
    
    return {'NAV_gbpusd_6m_forward': nav, 'LogReturn_gbpusd_6m_forward': log_returns}

def price_usdinr_3m_forward(data):
    """Price 3-month USD/INR forward contract (short position) and return NAV and log returns"""
    # Forward contract parameters
    NOTIONAL_USD = 100.0  # Notional amount in USD
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
    nav_series.iloc[0] = NOTIONAL_USD
    
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
                'notional_usd': NOTIONAL_USD
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
        # P&L = Notional_USD * (forward_rate/spot - 1)
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_USD * (forward_rate/spot - 1)  # Adjusted for USD/INR forward
        data_fx.loc[current_date, 'contract_pnl'] = pnl
    
    # Calculate daily returns based on P&L changes
    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_USD
    data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    
    # Fill NaN values in daily returns with zeros (for roll dates)
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav = pd.Series(index=data_fx.index)
    nav.iloc[0] = NOTIONAL_USD  # Start with 100 USD
    for i in range(1, len(data_fx)):
        prev_nav = nav.iloc[i-1]
        daily_return = data_fx['daily_return'].iloc[i]
        nav.iloc[i] = prev_nav * (1 + daily_return)
    
    # Compute log returns of NAV
    log_returns = np.log(nav / nav.shift(1))
    
    return {'NAV_usdinr_3m_forward': nav, 'LogReturn_usdinr_3m_forward': log_returns}

###############################################################################
#                             PORTFOLIO CLASS                                 #
###############################################################################

# Incorporate the Portfolio class directly
class Portfolio:
    def __init__(self, initial_capital=10000000.0, start_date=None):
        """Initialize portfolio with initial capital and start date"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.start_date = pd.to_datetime(start_date) if start_date else datetime.now()
        self.allocation = {}
        self.holdings = {}
        self.instruments = {}
        self.nav_history = pd.DataFrame(columns=['Date', 'NAV'])
        self.returns_history = pd.DataFrame(columns=['Date', 'Return'])
        self._load_pricing_data()
        
    def _load_pricing_data(self):
        """Load instrument pricing data from combined CSV file"""
        # Path to the combined instrument returns file
        combined_file = COMBINED_RETURNS_PATH
        
        try:
            print(f"Loading data from {combined_file}")
            # Load the combined data
            combined_data = pd.read_csv(combined_file, index_col='date', parse_dates=True)
            
            # Store the original trading dates from the source data
            self.trading_dates = combined_data.index.unique()
            print(f"Input file contains {len(self.trading_dates)} unique trading dates")
            
            # Parse all NAV and return columns
            nav_cols = [col for col in combined_data.columns if col.startswith('NAV_')]
            return_cols = [col for col in combined_data.columns if col.startswith('LogReturn_')]
            
            print(f"Found {len(nav_cols)} NAV columns and {len(return_cols)} return columns")
            
            # Create mapping between column names and formatted instrument names
            column_to_instrument = {
                'apple_price': 'Apple', 
                'lockheed_martin_price': 'Lockheed_martin',
                'nvidia_price': 'Nvidia', 
                'procter_gamble_price': 'Procter_gamble',
                'johnson_johnson_price': 'Johnson_johnson', 
                'toyota_price': 'Toyota',
                'nestle_price': 'Nestle', 
                'x_steel_price': 'X_steel',
                '10y_treasury': '10y_treasury', 
                'lqd_etf': 'Lqd_etf',
                '10y_tips': '10y_tips', 
                '1y_eur_zcb': '1y_eur_zcb',
                'high_yield_corp_debt': 'High_yield_corp_debt', 
                '5y_green_bond': '5y_green_bond',
                '30y_revenue_bond': '30y_revenue_bond', 
                'sp500_futures_1m': 'Sp500_futures_1m',
                'vix_futures': 'Vix_futures', 
                'crude_oil_futures': 'Crude_oil_futures',
                'gold_futures': 'Gold_futures', 
                'soybean_futures': 'Soybean_futures',
                'costco_itm_call': 'Costco_itm_call', 
                'eurusd_atm_call': 'Eurusd_atm_call',
                'xom_itm_put': 'Xom_itm_put', 
                'usdjpy_atm_put': 'Usdjpy_atm_put',
                'dax_variance_swap': 'Dax_variance_swap', 
                'nikkei_asian_put': 'Nikkei_asian_put',
                'ford_cds': 'Ford_cds', 
                'spx_knockout_call': 'Spx_knockout_call',
                'gbpusd_6m_forward': 'Gbpusd_6m_forward',
                'usdinr_3m_forward': 'Usdinr_3m_forward'
            }
            
            # Extract the individual instruments from the combined data
            for raw_name, instrument_name in column_to_instrument.items():
                nav_col = f'NAV_{raw_name}'
                return_col = f'LogReturn_{raw_name}'
                
                if nav_col in nav_cols and return_col in return_cols:
                    # Create a subset with just the NAV and return columns for this instrument
                    instrument_df = combined_data[[nav_col, return_col]].copy()
                    instrument_df.columns = ['NAV', 'Return']
                    
                    # Add to instruments dictionary
                    self.instruments[instrument_name] = instrument_df
                    print(f"Loaded {instrument_name} data with {len(instrument_df)} observations")
                else:
                    print(f"Warning: Could not find columns for {instrument_name}")
            
            # Print loaded instruments for debugging
            print(f"Loaded {len(self.instruments)} instruments: {list(self.instruments.keys())}")
            
            # Align datasets to common date range
            self._align_datasets()
            
        except FileNotFoundError:
            print(f"Error: Combined instrument data file not found at {combined_file}")
        except Exception as e:
            print(f"Error loading combined instrument data: {str(e)}")
    
    def _align_datasets(self):
        """Align all instrument datasets to a common date range"""
        if not self.instruments:
            return
            
        # Find common date range across all instruments
        start_dates = []
        end_dates = []
        
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                start_dates.append(data.index.min())
                end_dates.append(data.index.max())
        
        if not start_dates or not end_dates:
            return
            
        start_date = max(start_dates)
        end_date = min(end_dates)
        
        # If user provided start date is after the earliest data point, use that instead
        if self.start_date > start_date:
            start_date = self.start_date
        
        # Get the exact list of trading dates from the input file 
        # that fall within our date range
        if hasattr(self, 'trading_dates'):
            valid_trading_dates = self.trading_dates[
                (self.trading_dates >= start_date) & 
                (self.trading_dates <= end_date)
            ]
            print(f"Valid trading dates after alignment: {len(valid_trading_dates)}")
        else:
            # As a fallback, construct from the first instrument
            first_instrument = next(iter(self.instruments.values()))
            valid_trading_dates = first_instrument.index[
                (first_instrument.index >= start_date) & 
                (first_instrument.index <= end_date)
            ]
        
        # Filter all datasets to exactly the same set of trading dates
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                # Only keep rows for valid trading dates
                self.instruments[name] = data.reindex(valid_trading_dates)
        
        # Count actual trading days after alignment
        sample_instrument = next(iter(self.instruments.values()))
        if sample_instrument is not None and not sample_instrument.empty:
            num_trading_days = len(sample_instrument)
            print(f"Datasets aligned with {num_trading_days} trading days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        else:
            print("Warning: No data available after alignment")

    def add_instrument(self, name, data_path=None, data=None):
        """Add a new instrument to the portfolio
        
        Args:
            name: Name of the instrument
            data_path: Path to CSV data file (alternative to data)
            data: DataFrame with instrument data (alternative to data_path)
        """
        if data is not None:
            self.instruments[name] = data
        elif data_path is not None:
            # Load directly from specified CSV file
            try:
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                
                # Find NAV and Log Return columns
                nav_col = next((col for col in df.columns if 'NAV' in col), None)
                return_col = next((col for col in df.columns if 'Return' in col or 'return' in col), None)
                
                if nav_col and return_col:
                    # Create a subset with just the NAV and return columns
                    instrument_df = df[[nav_col, return_col]].copy()
                    instrument_df.columns = ['NAV', 'Return']
                    self.instruments[name] = instrument_df
                    print(f"Loaded {name} data with {len(instrument_df)} observations")
                else:
                    print(f"Error: Could not find NAV and Return columns in {data_path}")
            except Exception as e:
                print(f"Error loading {name} data: {str(e)}")
        else:
            print("Error: Either data_path or data must be provided")
            
        # Re-align datasets
        self._align_datasets()
        
        # Add to allocation with zero allocation initially
        if name not in self.allocation:
            self.allocation[name] = 0.0
            self.holdings[name] = 0.0

    def set_allocation(self, **allocations):
        """Set portfolio allocation between instruments
        
        Args:
            **allocations: Keyword arguments with instrument name and percentage
                           e.g., Apple=20.0, Gold=25.0, Cash=25.0
        """
        # Validate percentages sum to 100
        total_pct = sum(allocations.values())
        if abs(total_pct - 100.0) > 0.001:
            raise ValueError(f"Allocation percentages must sum to 100% (got {total_pct}%)")
            
        # Set allocation percentages
        for name, pct in allocations.items():
            self.allocation[name] = pct / 100.0
        
        # Calculate holdings based on allocation
        for name, alloc in self.allocation.items():
            self.holdings[name] = self.current_capital * alloc
        
        print(f"Portfolio allocation set to: {', '.join([f'{k}: {v}%' for k, v in allocations.items()])}")
        return self.allocation
    
    def calculate_returns(self, end_date=None, cash_rate=0.03):
        """Calculate portfolio returns from start date to end date
        
        Args:
            end_date: End date for calculation (str in 'YYYY-MM-DD' format or datetime)
            cash_rate: Annual cash return rate for cash holdings (default: 3%)
        
        Returns:
            DataFrame with portfolio NAV and return history
        """
        # Convert date strings to datetime if needed
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Get common date range for all instruments
        start_dates = []
        end_dates = []
        
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                start_dates.append(data.index.min())
                end_dates.append(data.index.max())
        
        if not start_dates or not end_dates:
            print("No instrument data available. Cannot calculate returns.")
            return None
            
        start_date = max(start_dates)
        if start_date < self.start_date:
            start_date = self.start_date
            
        if end_date is None:
            end_date = min(end_dates)
        
        print(f"Calculating portfolio returns from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get the trading dates from the aligned datasets
        # This ensures we're using the exact same set of dates across all instruments
        sample_instrument = next(iter(self.instruments.values()))
        if sample_instrument is None or sample_instrument.empty:
            print("Error: No data available for return calculation.")
            return None
            
        # Get dates from the sample instrument - these should all be aligned already
        trading_dates = sample_instrument.index.tolist()
        print(f"Using {len(trading_dates)} trading days from the aligned dataset")
        
        # Calculate daily cash return (compounded daily)
        daily_cash_return = (1 + cash_rate) ** (1/252) - 1
        
        # Store initial allocation amounts
        initial_allocations = {name: self.holdings.get(name, 0) for name in self.instruments.keys()}
        cash_allocation = self.holdings.get('Cash', 0)
        
        # Calculate initial prices and shares for each instrument
        instrument_shares = {}
        for name, amount in initial_allocations.items():
            if name in self.instruments and self.instruments[name] is not None:
                if not self.instruments[name].empty:
                    first_valid_date = self.instruments[name].first_valid_index()
                    if first_valid_date is not None and 'NAV' in self.instruments[name].columns:
                        initial_price = self.instruments[name].loc[first_valid_date, 'NAV']
                        if initial_price > 0:
                            # Calculate shares based on initial allocation and price
                            instrument_shares[name] = amount / initial_price
                        else:
                            instrument_shares[name] = 0
                    else:
                        instrument_shares[name] = 0
                else:
                    instrument_shares[name] = 0
            else:
                instrument_shares[name] = 0
        
        # Initialize tracking values
        instrument_navs = {}
        cash_nav = cash_allocation
        
        nav_records = []
        return_records = []
        
        prev_total_nav = sum(initial_allocations.values()) + cash_allocation
        prev_date = None
        
        for date in trading_dates:
            # For cash returns, calculate based on days since last trading day
            if prev_date is not None:
                days_since_last = (date - prev_date).days
                period_cash_return = (1 + cash_rate) ** (days_since_last/365) - 1
                cash_nav = cash_nav * (1 + period_cash_return)
            
            # Calculate current NAV for each instrument based on shares and current price
            for name, shares in instrument_shares.items():
                if name in self.instruments and self.instruments[name] is not None:
                    if date in self.instruments[name].index and 'NAV' in self.instruments[name].columns:
                        current_price = self.instruments[name].loc[date, 'NAV']
                        if not pd.isna(current_price):
                            instrument_navs[name] = shares * current_price
                        else:
                            # If NAV not available for this date, use previous NAV
                            instrument_navs[name] = instrument_navs.get(name, initial_allocations.get(name, 0))
                    else:
                        # If instrument data not available for this date, use previous NAV
                        instrument_navs[name] = instrument_navs.get(name, initial_allocations.get(name, 0))
                else:
                    # Instrument not in portfolio
                    instrument_navs[name] = instrument_navs.get(name, initial_allocations.get(name, 0))
            
            # Calculate total portfolio NAV
            total_nav = sum(instrument_navs.values()) + cash_nav
            
            # Calculate period returns
            total_return = (total_nav / prev_total_nav) - 1 if prev_total_nav > 0 else 0
            
            # Store values
            nav_record = {'Date': date, 'NAV': total_nav, 'Cash_NAV': cash_nav}
            for name, nav in instrument_navs.items():
                nav_record[f'{name}_NAV'] = nav
            
            nav_records.append(nav_record)
            
            return_records.append({
                'Date': date,
                'Return': total_return
            })
            
            # Update previous values for next iteration
            prev_total_nav = total_nav
            prev_date = date
        
        # Create history DataFrames
        self.nav_history = pd.DataFrame(nav_records).set_index('Date')
        self.returns_history = pd.DataFrame(return_records).set_index('Date')
        
        # Verify the number of rows in the output matches the expected number of trading days
        print(f"Generated {len(self.nav_history)} NAV records and {len(self.returns_history)} return records")
        
        # Update current capital to latest NAV
        self.current_capital = total_nav
        
        # Calculate simple performance metrics
        annualized_return = ((1 + self.returns_history['Return']).prod()) ** (252/len(self.returns_history)) - 1
        volatility = self.returns_history['Return'].std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(self.nav_history['NAV'])
        
        print(f"\nPortfolio Performance ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final NAV: ${total_nav:,.2f}")
        print(f"Number of trading days: {len(self.returns_history)}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        return self.nav_history
    
    def _calculate_max_drawdown(self, nav_series):
        """Calculate maximum drawdown from NAV series"""
        roll_max = nav_series.cummax()
        drawdown = (nav_series - roll_max) / roll_max
        return drawdown.min()
    
    def plot_performance(self, save_path=None):
        """Plot portfolio performance"""
        if self.nav_history.empty:
            print("No performance data available. Run calculate_returns() first.")
            return
            
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot NAV values with log scale
        ax1.plot(self.nav_history.index, self.nav_history['NAV'], 'b-', label='Portfolio NAV')
        for name in self.instruments.keys():
            if f'{name}_NAV' in self.nav_history.columns:
                # Only plot positions that have positive values for log scale
                if (self.nav_history[f'{name}_NAV'] > 0).all():
                    ax1.plot(self.nav_history.index, self.nav_history[f'{name}_NAV'], '-', label=f'{name} NAV')
            
        if 'Cash_NAV' in self.nav_history.columns:
            ax1.plot(self.nav_history.index, self.nav_history['Cash_NAV'], 'g-', label='Cash NAV')
            
        ax1.set_title('Portfolio Performance (Log Scale)')
        ax1.set_ylabel('NAV ($)')
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)
        
        # Plot allocation over time
        allocation_data = {}
        for name in list(self.instruments.keys()) + ['Cash']:
            nav_col = f'{name}_NAV'
            if nav_col in self.nav_history.columns:
                # Calculate percentage allocation (absolute value for plotting purposes)
                allocation_data[name] = self.nav_history[nav_col].abs() / self.nav_history['NAV'].abs() * 100
        
        if allocation_data:
            allocation_df = pd.DataFrame(allocation_data)
            
            # Normalize the allocation percentages to sum to 100%
            row_sums = allocation_df.sum(axis=1)
            for col in allocation_df.columns:
                allocation_df[col] = allocation_df[col] / row_sums * 100
            
            # Plot as lines instead of stacked area to handle negative values
            for col in allocation_df.columns:
                ax2.plot(allocation_df.index, allocation_df[col], '-', label=col)
        
        ax2.set_ylabel('Allocation (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Performance chart saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, folder_path=None):
        """Save portfolio results to CSV files"""
        if self.nav_history.empty:
            print("No performance data available. Run calculate_returns() first.")
            return
            
        if folder_path is None:
            folder_path = PORTFOLIO_RESULTS_DIR
            
        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Save NAV history with retry mechanism
        nav_path = PORTFOLIO_NAV_HISTORY_PATH
        try:
            self.nav_history.to_csv(nav_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {nav_path}. File may be in use.")
        
        # Save returns history with retry mechanism
        returns_path = PORTFOLIO_RETURNS_HISTORY_PATH
        try:
            self.returns_history.to_csv(returns_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {returns_path}. File may be in use.")
        
        print(f"Portfolio results saved to {folder_path}")

# Function to create a portfolio with custom instrument allocations
def create_portfolio_with_custom_allocation(instruments_allocation, initial_capital=1000000.0, start_date='2022-01-01', end_date='2024-12-31', cash_rate=0.025):
    """
    Create and calculate a portfolio with custom instrument allocations.
    
    Args:
        instruments_allocation (dict): Dictionary with instrument names as keys and allocation percentages as values.
                                      Example: {'Vix_futures': 30.0, 'Apple': 25.0}
        initial_capital (float): Initial capital for the portfolio.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        cash_rate (float): Annual cash return rate.
        
    Returns:
        Portfolio: Configured portfolio with calculated returns.
    """
    # Validate total allocation
    total_allocation = sum(instruments_allocation.values())
    if abs(total_allocation - 100.0) > 0.001:
        remaining = 100.0 - total_allocation
        print(f"Warning: Allocations sum to {total_allocation}%. Adding {remaining}% to Cash.")
        instruments_allocation['Cash'] = instruments_allocation.get('Cash', 0.0) + remaining
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=initial_capital, start_date=start_date)
    
    # Set allocation
    portfolio.set_allocation(**instruments_allocation)
    
    # Calculate returns
    portfolio.calculate_returns(end_date=end_date, cash_rate=cash_rate)
    
    return portfolio

def plot_portfolio_performance(portfolio, save_prefix):
    """
    Create plot for cumulative NAV.
    
    Args:
        portfolio: Portfolio object with calculated returns
        save_prefix: Prefix for saving plot files
    """
    if portfolio.nav_history.empty:
        print("No performance data available. Run calculate_returns() first.")
        return
    
    # Create figure for cumulative NAV (with log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio.nav_history.index, portfolio.nav_history['NAV'], 'b-', linewidth=2, label='Portfolio NAV')
    ax.set_title('Portfolio Cumulative NAV (Log Scale)')
    ax.set_ylabel('NAV ($)')
    ax.set_xlabel('Date')
    ax.set_yscale('log')  # Use logarithmic scale for y-axis
    ax.grid(True, alpha=0.3)
    
    # Add horizontal grid lines specifically for log scale
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, which='both', linestyle='-', alpha=0.2)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(PORTFOLIO_PLOT_PATH, dpi=300)
    plt.close(fig)
    
    print(f"Performance chart saved with prefix: {save_prefix}")

# Function to process all instruments and output a single CSV with NAVs and log returns
def process_all_instruments(data):
    """Process all instruments and collect NAVs and log returns in a single DataFrame"""
    # Initialize dictionary to collect results
    all_results = {}
    
    # Process equities
    print("Processing equities...")
    equity_results = price_equities(data)
    all_results.update(equity_results)
    
    # Process fixed income
    print("Processing fixed income instruments...")
    # 10Y Treasury
    treasury_results = price_10y_treasury_bond(data)
    all_results.update(treasury_results)
    
    # LQD ETF
    lqd_results = price_lqd_etf(data)
    all_results.update(lqd_results)
    
    # 10Y TIPS
    tips_results = price_10y_tips(data)
    all_results.update(tips_results)
    
    # 1Y EUR Zero Coupon Bond
    zcb_results = price_1y_eur_zcb(data)
    all_results.update(zcb_results)
    
    # High Yield Corporate Debt
    corp_debt_results = price_high_yield_corp_debt(data)
    all_results.update(corp_debt_results)
    
    # 5Y Green Bond
    green_bond_results = price_5y_green_bond(data)
    all_results.update(green_bond_results)
    
    # 30Y Revenue Bond
    revenue_bond_results = price_30y_revenue_bond(data)
    all_results.update(revenue_bond_results)
    
    # Process derivatives
    print("Processing derivatives...")
    # S&P 500 Futures
    sp500_futures_results = price_sp500_futures_1m(data)
    all_results.update(sp500_futures_results)
    
    # VIX Futures
    vix_futures_results = price_vix_futures(data)
    all_results.update(vix_futures_results)
    
    # Crude Oil Futures
    crude_oil_futures_results = price_crude_oil_futures(data)
    all_results.update(crude_oil_futures_results)
    
    # Gold Futures
    gold_futures_results = price_gold_futures(data)
    all_results.update(gold_futures_results)
    
    # Soybean Futures
    soybean_futures_results = price_soybean_futures(data)
    all_results.update(soybean_futures_results)
    
    # Costco ITM Call Option
    costco_call_results = price_costco_itm_call_option(data)
    all_results.update(costco_call_results)
    
    # EUR/USD ATM Call Option
    eurusd_call_results = price_eurusd_atm_call_option(data)
    all_results.update(eurusd_call_results)
    
    # Exxon Mobil ITM Put Option
    xom_put_results = price_xom_itm_put_option(data)
    all_results.update(xom_put_results)
    
    # USD/JPY ATM Put Option
    usdjpy_put_results = price_usdjpy_atm_put_option(data)
    all_results.update(usdjpy_put_results)
    
    # DAX 30-day Variance Swap
    dax_swap_results = price_variance_swap_dax(data)
    all_results.update(dax_swap_results)
    
    # Nikkei Asian Put Option
    nikkei_put_results = price_asian_put_option(data)
    all_results.update(nikkei_put_results)
    
    # Ford 5-year CDS
    ford_cds_results = price_ford_cds(data)
    all_results.update(ford_cds_results)
    
    # S&P 500 Knock-Out Call Option
    spx_knockout_results = price_spx_barrier_option(data)
    all_results.update(spx_knockout_results)
    
    # Process forex
    print("Processing forex...")
    # GBP/USD Forward
    gbpusd_forward_results = price_gbpusd_6m_forward(data)
    all_results.update(gbpusd_forward_results)
    
    # USD/INR Forward
    usdinr_forward_results = price_usdinr_3m_forward(data)
    all_results.update(usdinr_forward_results)
    
    # Convert results to DataFrame and ensure the index is properly set
    results_df = pd.DataFrame(all_results, index=data.index)
    
    # Ensure all NAV columns have corresponding log return columns
    print("\nChecking for missing log return columns...")
    nav_cols = [col for col in results_df.columns if col.startswith('NAV_')]
    
    for nav_col in nav_cols:
        instrument = nav_col.replace('NAV_', '')
        log_col = f"LogReturn_{instrument}"
        if log_col not in results_df.columns:
            print(f"Adding missing log return column for {instrument}...")
            # Calculate log returns directly from NAV values
            results_df[log_col] = np.log(results_df[nav_col] / results_df[nav_col].shift(1))
    
    # Output to CSV
    results_df.to_csv(COMBINED_RETURNS_PATH)
    
    # Now check what was actually written to the CSV
    try:
        # This needs to be done after saving to ensure we check what was actually saved
        import csv
        with open(COMBINED_RETURNS_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        # Print column counts and types
        nav_cols = [col for col in header if col.startswith('NAV_')]
        log_cols = [col for col in header if col.startswith('LogReturn_')]
        
        print(f"\nSummary of columns in combined_instrument_returns.csv:")
        print(f"Total columns: {len(header)}")
        print(f"NAV columns: {len(nav_cols)}")
        print(f"LogReturn columns: {len(log_cols)}")
        
        print("\nAll results written to combined_instrument_returns.csv")
    except Exception as e:
        print(f"Error checking saved file: {e}")
        print("\nResults written to combined_instrument_returns.csv")
    
    return results_df

def main():
    """Main function to run all pricing operations and portfolio creation"""
    print("Loading market data...")
    data = load_data()
    if data is None:
        print("Failed to load market data. Exiting.")
        return
    
    print(f"Loaded market data with {len(data)} rows")
    
    # Process all instruments and output combined results
    results = process_all_instruments(data)
    print("Pricing processing complete")
    
    # Create a diversified portfolio using the instruments from the combined CSV file
    custom_allocation = {
        'Apple': 2,
        'Lockheed_martin': 2.0,
        'Nvidia': 2,
        'Procter_gamble': 2.0,
        'Johnson_johnson': 2.0,
        'Toyota': 2.0,
        'Nestle': 2.0,
        'X_steel': 2.0,
        '10y_treasury': 5.0,
        'Lqd_etf': 4.0,
        '10y_tips': 5.0,
        '1y_eur_zcb': 5.0,
        'High_yield_corp_debt': 4.0,
        '5y_green_bond': 3.0,
        '30y_revenue_bond': 4.0,
        'Sp500_futures_1m': 4.0,
        'Vix_futures': 3.0,
        'Crude_oil_futures': 3.0,
        'Gold_futures': 4.0,
        'Soybean_futures': 3.0,
        'Costco_itm_call': 3.0,
        'Xom_itm_put': 3.0,
        'Eurusd_atm_call': 3.0,
        'Usdjpy_atm_put': 3.0,
        'Gbpusd_6m_forward': 4.0,
        'Usdinr_3m_forward': 4.0,
        'Ford_cds': 3.0, 
        'Dax_variance_swap': 3.0,
        'Nikkei_asian_put': 3.0,
        'Spx_knockout_call': 3.0,
        'Cash': 5.0
    }
    
    # Output directory
    output_dir = PORTFOLIO_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Create portfolio with custom allocation - use dates that we have full data for
    print("\nCreating and calculating portfolio...")
    portfolio = create_portfolio_with_custom_allocation(
        custom_allocation,
        initial_capital=10000000.0,  # Explicitly set to 10 million
        start_date='2005-01-03',  # Start date from our dataset
        end_date='2024-12-31'
    )
    
    # Plot performance and save to output directory
    plot_portfolio_performance(portfolio, PORTFOLIO_RESULTS_DIR)
    
    # Save results to output directory
    portfolio.save_results(output_dir)
    
    print("Portfolio processing complete.")

if __name__ == "__main__":
    main() 