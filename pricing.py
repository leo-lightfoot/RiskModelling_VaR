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
PORTFOLIO_RETURNS_HISTORY_PATH = 'portfolio_results/portfolio_returns_history.csv' #Portfolio Returns history
PORTFOLIO_PLOT_PATH = 'portfolio_results/portfolio_cumulative_nav_log.png' #Portfolio NAV Visualization

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
    
    # Set initial value using the first date
    first_date = prices.index[0]
    nav.loc[first_date] = start_value
    
    # Calculate NAV for each date
    for i in range(1, len(nav)):
        current_date = prices.index[i]
        previous_date = prices.index[i-1]
        
        if not np.isnan(daily_returns.loc[current_date]):
            nav.loc[current_date] = nav.loc[previous_date] * (1 + daily_returns.loc[current_date])
        else:
            nav.loc[current_date] = nav.loc[previous_date]
    
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

    data['daily_returns'] = data['bond_price'].pct_change(fill_method=None)
    data['log_returns'] = np.log(data['bond_price'] / data['bond_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        
        # Calculate roll yield (difference between old and new bond)
        roll_yield = (data.loc[roll_date, 'bond_price'] - data.loc[prev_date, 'bond_price']) / data.loc[prev_date, 'bond_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]

    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    
    return {'NAV_10y_treasury': nav, 'LogReturn_10y_treasury': log_returns}

def price_lqd_etf(data): # LQD Corporate Bond ETF

    lqd_column = 'lqd_corporate_bond_etf'
    data = clean_numeric_data(data, [lqd_column])
    nav, log_returns = calculate_nav_and_returns(data[lqd_column])
    log_returns_series = pd.Series(log_returns.values, index=data.index)
    return {'NAV_lqd_etf': nav, 'LogReturn_lqd_etf': log_returns_series}

def price_10y_tips(data): # 10Y TIPS

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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        # Keep same value on roll dates (no artificial jumps)
        data.loc[roll_date, 'daily_returns'] = 0
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not pd.isna(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]

    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    return {'NAV_10y_tips': nav, 'LogReturn_10y_tips': log_returns}

def price_1y_eur_zcb(data): # 1Y EUR Zero Coupon Bond

    INITIAL_USD = 100  # USD
    MATURITY = 1  # 1 year
    FREQUENCY = 1  # Annual payment (zero coupon)
    
    def calculate_zcb_price(yield_rate, years_to_maturity):
        return np.exp(-yield_rate * years_to_maturity)

    data = clean_numeric_data(data, ['1_year_euro_yield_curve', 'fx_eurusd_rate'])
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
        days_to_maturity = (current_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'zcb_price_eur'] - data.loc[prev_date, 'zcb_price_eur']) / data.loc[prev_date, 'zcb_price_eur']
        data.loc[roll_date, 'daily_returns_eur'] = roll_yield
    
    # Compute NAV in EUR
    nav_eur = pd.Series(index=data.index)
    first_date = data.index[0]
    initial_eur = INITIAL_USD / data.loc[first_date, 'fx_eurusd_rate']
    nav_eur.loc[first_date] = initial_eur
    
    for i in range(1, len(nav_eur)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns_eur']):
            nav_eur.loc[current_date] = nav_eur.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns_eur'])
        else:
            nav_eur.loc[current_date] = nav_eur.loc[prev_date]
    
    # Convert NAV to USD
    nav_usd = nav_eur * data['fx_eurusd_rate']
    
    # Calculate USD returns
    data['daily_returns_usd'] = nav_usd.pct_change()
    data['log_returns_usd'] = np.log(nav_usd / nav_usd.shift(1))
    
    return {'NAV_1y_eur_zcb': nav_usd, 'LogReturn_1y_eur_zcb': data['log_returns_usd']}

def price_high_yield_corp_debt(data): # High Yield Corporate Debt

    NOTIONAL = 100  # USD
    MATURITY = 5 
    COUPON_RATE = 0.065  # 6.5% typical for high yield corporate bonds
    FREQUENCY = 2  # Semiannual coupon payments
    
    data = clean_numeric_data(data, ['10y_treasury_yield', 'high_yield_credit spread'])
    data['10y_treasury_yield'] = data['10y_treasury_yield'] / 100
    data['high_yield_credit spread'] = data['high_yield_credit spread'] / 100
    data['high_yield_rate'] = data['10y_treasury_yield'] + data['high_yield_credit spread']
    
    # Initialize variables for bond pricing
    data['bond_price'] = np.nan
    data['days_to_maturity'] = 0
    data['roll_date'] = False
    
    # Calculate bond prices with monthly roll-over logic
    current_maturity = data.index[0] + pd.DateOffset(months=1)  # Monthly roll-over
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Check if we need to roll over
        if current_date >= current_maturity:
            data.loc[current_date, 'roll_date'] = True
            current_maturity = current_date + pd.DateOffset(months=1)

        full_maturity = current_date + pd.DateOffset(years=MATURITY)
        days_to_maturity = (full_maturity - current_date).days
        data.loc[current_date, 'days_to_maturity'] = days_to_maturity
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'bond_price'] - data.loc[prev_date, 'bond_price']) / data.loc[prev_date, 'bond_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield + daily_coupon_return
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_high_yield_corp_debt': nav, 'LogReturn_high_yield_corp_debt': data['log_returns']}

def price_5y_green_bond(data): # 5Y Green Bond

    NOTIONAL = 100  # USD
    MATURITY = 5  # 5 years
    COUPON_RATE = 0.025  # 2.5% annual, typically somewhat lower than conventional bonds
    FREQUENCY = 2  # Semiannual coupon payments
    GREEN_PREMIUM = 0.0010  # 10 bps "greenium" - premium for green bonds

    data = clean_numeric_data(data, ['5y_treasury_yield'])
    data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'bond_price'] - data.loc[prev_date, 'bond_price']) / data.loc[prev_date, 'bond_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield + daily_coupon_return
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_5y_green_bond': nav, 'LogReturn_5y_green_bond': data['log_returns']}

def price_30y_revenue_bond(data): # 30Y Revenue Bond

    NOTIONAL = 100  # USD
    MATURITY = 30  # 30 years
    COUPON_RATE = 0.035  # 3.5% annual
    FREQUENCY = 2  # Semiannual coupon payments
    CREDIT_SPREAD = 0.0050  # 50 bps spread over Treasury
    
    data = clean_numeric_data(data, ['30Y_treasury_yield'])
    data['30Y_treasury_yield'] = data['30Y_treasury_yield'] / 100
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'bond_price'] - data.loc[prev_date, 'bond_price']) / data.loc[prev_date, 'bond_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield + daily_coupon_return
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_30y_revenue_bond': nav, 'LogReturn_30y_revenue_bond': data['log_returns']}


# DERIVATIVES PRICING 

def price_sp500_futures_1m(data): # 1M S&P 500 Futures

    START_VALUE = 100

    # Function to get next expiry date
    def get_next_expiry(date):
        current_year = date.year
        current_month = date.month
        
        if current_month == 12:
            next_month = 1
            next_year = current_year + 1
        else:
            next_month = current_month + 1
            next_year = current_year
        
        return datetime(next_year, next_month, 15)
    
    data = clean_numeric_data(data, ['sp500_index', 'fed_funds_rate', 'SPX_Div_yield'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['SPX_Div_yield'] = data['SPX_Div_yield'] / 100
    
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
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price using actual dividend yield from the data
        if not np.isnan(data.loc[current_date, 'sp500_index']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']) and not np.isnan(data.loc[current_date, 'SPX_Div_yield']):
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'sp500_index'] * np.exp(
                (data.loc[current_date, 'fed_funds_rate'] - data.loc[current_date, 'SPX_Div_yield']) * time_to_maturity
            )
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'contract_price'] - data.loc[prev_date, 'contract_price']) / data.loc[prev_date, 'contract_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = START_VALUE
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_sp500_futures_1m': nav, 'LogReturn_sp500_futures_1m': data['log_returns']}

def price_vix_futures(data): # VIX Futures

    np.random.seed(42) # random seed for reproducibility
    START_VALUE = 100
    
    data = clean_numeric_data(data, ['vix_index_level'])
    vix_mean = data['vix_index_level'].rolling(window=30).mean()
    
    # Set parameters
    theta = 0.2           # Speed of mean reversion
    sigma = data['vix_index_level'].pct_change().std() * data['vix_index_level'].mean()
    contract_days = 21    # Approx 1 month (trading days)
    
    # Get the first value using date-based indexing
    first_date = data.index[0]
    initial_price = data.loc[first_date, 'vix_index_level']
    
    # Initialize simulation
    vix_futures_price = []
    current_price = initial_price
    
    for i in range(len(data)):
        current_date = data.index[i]
        
        if i < 30: # Not enough history for moving average
            vix_futures_price.append(np.nan)
            continue

        if (i - 30) % contract_days == 0:
            current_price = data.loc[current_date, 'vix_index_level']  # Assume it realizes to spot
            days_left = contract_days
        
        # Get long-term mean from moving average - need index location for vix_mean at current date
        long_term_mean = vix_mean.loc[current_date]
        
        # Mean-reverting update
        shock = sigma * np.random.normal()
        current_price += theta * (long_term_mean - current_price) + shock
        vix_futures_price.append(current_price)
    
    # Create futures price series
    data['contract_price'] = pd.Series(vix_futures_price, index=data.index)
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    data['roll_date'] = False
    
    # We need to use iloc here as we're setting values based on positional index
    for i in range(30, len(data), contract_days):
        if i < len(data):
            data.iloc[i, data.columns.get_loc('roll_date')] = True
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.loc[first_date] = START_VALUE
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_vix_futures': nav, 'LogReturn_vix_futures': data['log_returns']}

def price_crude_oil_futures(data): # 1M Crude Oil Futures

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
    
    data = clean_numeric_data(data, ['crude_oil_wti_spot', 'fed_funds_rate'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'contract_price'] - data.loc[prev_date, 'contract_price']) / data.loc[prev_date, 'contract_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = START_VALUE
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_crude_oil_futures': nav, 'LogReturn_crude_oil_futures': data['log_returns']}

def price_gold_futures(data): # 3M Gold Futures

    STORAGE_COST = 0.0005  # Monthly storage cost as fraction of price
    START_VALUE = 100
    CONTRACT_MONTHS = 3  # 3-month contract
    
    # Function to get next expiry date for 3-month contract
    def get_next_expiry(date):
        return date + pd.DateOffset(months=CONTRACT_MONTHS)

    data = clean_numeric_data(data, ['gold_spot_price', 'fed_funds_rate'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    
    # Initialize variables
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
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price
        if not np.isnan(data.loc[current_date, 'gold_spot_price']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            contango_factor = (1 + data.loc[current_date, 'fed_funds_rate'] + STORAGE_COST) ** time_to_maturity
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'gold_spot_price'] * contango_factor
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'contract_price'] - data.loc[prev_date, 'contract_price']) / data.loc[prev_date, 'contract_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = START_VALUE
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_gold_futures': nav, 'LogReturn_gold_futures': data['log_returns']}

def price_soybean_futures(data): # 6M Soybean Futures

    START_VALUE = 100
    CONTRACT_MONTHS = 6  # 6-month contract
    
    def get_next_expiry(date):
        return date + pd.DateOffset(months=CONTRACT_MONTHS)
    
    # Function to calculate seasonal storage cost based on month
    def seasonal_storage_cost(month):
        if month in [10, 11]:  # Harvest season
            return 0.0015  # Lower storage cost (0.15% per month)
        elif month in [7, 8, 9]:  # Pre-harvest
            return 0.0040  # Higher storage cost (0.40% per month)
        else:
            return 0.0025  # Standard storage cost (0.25% per month)
    
    data = clean_numeric_data(data, ['soybean_spot_usd', 'fed_funds_rate'])
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
        time_to_maturity = days_to_expiry / 365.0
        
        # Calculate futures price
        if not np.isnan(data.loc[current_date, 'soybean_spot_usd']) and not np.isnan(data.loc[current_date, 'fed_funds_rate']):
            contango_factor = (1 + data.loc[current_date, 'fed_funds_rate'] + data.loc[current_date, 'storage_cost']) ** time_to_maturity
            data.loc[current_date, 'contract_price'] = data.loc[current_date, 'soybean_spot_usd'] * contango_factor
    
    # Calculate returns with roll-over adjustment
    data['daily_returns'] = data['contract_price'].pct_change()
    data['log_returns'] = np.log(data['contract_price'] / data['contract_price'].shift(1))
    
    # Adjust returns on roll dates to account for roll yield
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        roll_yield = (data.loc[roll_date, 'contract_price'] - data.loc[prev_date, 'contract_price']) / data.loc[prev_date, 'contract_price']
        data.loc[roll_date, 'daily_returns'] = roll_yield
    
    # Compute NAV with roll-over adjustments
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = START_VALUE
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_soybean_futures': nav, 'LogReturn_soybean_futures': data['log_returns']}

def price_costco_itm_call_option(data): # 3M Costco ITM Call Option

    NOTIONAL = 100
    STRIKE_RATIO = 0.95  # In the money by 5%
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    TRANSACTION_COST = 0.0010  # 10 bps per transaction
    
    # Black-Scholes formula for call option pricing
    def black_scholes_call(S, K, T, r, sigma, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(0.0001, call)  # Ensure no zero values for log returns
    
    data = clean_numeric_data(data, ['costco_stock_price', 'COST_IVOL_3MMA', 'fed_funds_rate', 'COST_DIV_YIELD'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100
    data['COST_DIV_YIELD'] = data['COST_DIV_YIELD'] / 100
    data['COST_IVOL_3MMA'] = data['COST_IVOL_3MMA'] / 100
    
    # Initialize variables 
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        data.loc[roll_date, 'daily_returns'] = data.loc[roll_date, 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_costco_itm_call': nav, 'LogReturn_costco_itm_call': data['log_returns']}

def price_xom_itm_put_option(data): # 3M Exxon Mobil ITM Put Option

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
    
    data = clean_numeric_data(data, ['exxonmobil_stock_price', 'XOM_IVOL_3MMA', 'fed_funds_rate', 'XOM_DIV_YIELD'])
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        data.loc[roll_date, 'daily_returns'] = data.loc[roll_date, 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_xom_itm_put': nav, 'LogReturn_xom_itm_put': data['log_returns']}

def price_variance_swap_dax(data): # 30D DAX Variance Swap

    NOTIONAL = 100
    DAYS_TO_EXPIRY = 30  # 30-day variance swap
    VARIANCE_STRIKE_ADJUSTMENT = 0.98  # Variance strike is typically set below fair value
    
    data = clean_numeric_data(data, ['DAX_Call_ivol_30D', 'DAX_Put_ivol_30D'])
    data['DAX_Call_ivol_30D'] = data['DAX_Call_ivol_30D'] / 100
    data['DAX_Put_ivol_30D'] = data['DAX_Put_ivol_30D'] / 100
    data['implied_variance'] = (data['DAX_Call_ivol_30D']**2 + data['DAX_Put_ivol_30D']**2) / 2
    
    # Initialize variables
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
        time_to_maturity = days_to_expiry / 365.0
        
        # Set variance strike at the beginning of each swap period
        if data.loc[current_date, 'roll_date'] or i == 0:
            data.loc[current_date, 'variance_strike'] = data.loc[current_date, 'implied_variance'] * VARIANCE_STRIKE_ADJUSTMENT
        else:
            if i > 0:
                prev_date = data.index[i-1]
                data.loc[current_date, 'variance_strike'] = data.loc[prev_date, 'variance_strike']
        
        # Calculate present value of fixed leg (we pay fixed, so positive value means profit)
        if not np.isnan(data.loc[current_date, 'variance_strike']) and not np.isnan(data.loc[current_date, 'implied_variance']):
            # PV of fixed leg = (implied variance - variance strike) * notional * remaining time
            data.loc[current_date, 'pv_variance_leg'] = (data.loc[current_date, 'implied_variance'] - 
                                                       data.loc[current_date, 'variance_strike']) * time_to_maturity * NOTIONAL
    
    # Calculate returns based on changes in PV
    data['daily_pnl'] = data['pv_variance_leg'].diff()
    data['daily_returns'] = data['daily_pnl'] / NOTIONAL
    data['log_returns'] = data['daily_returns']
    
    # Set first day's return to 0
    first_date = data.index[0]
    data.loc[first_date, 'daily_returns'] = 0
    data.loc[first_date, 'log_returns'] = 0
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_variance_swap_dax': nav, 'LogReturn_variance_swap_dax': data['log_returns']}

def price_asian_put_option(data): # 3M Nikkei Asian Put Option

    NOTIONAL = 100
    STRIKE_RATIO = 1 #ATM
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    AVERAGING_FREQUENCY = 5  # Average price every 5 trading days
    TRANSACTION_COST = 0.0010  # 10 bps per transaction
    NUM_SIMULATIONS = 10000  # Number of simulations for Monte Carlo
    
    # Monte Carlo simulation for Asian put option pricing
    def price_asian_put_mc(S0, K, r, q, sigma, T, steps=63, paths=10000):

        np.random.seed(42)  # Random seed for reproducibility
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
        put_payoffs = np.maximum(K - avg_prices, 0)
        
        # Discount payoffs to present value
        price = np.exp(-r * T) * np.mean(put_payoffs)
        return max(0.0001, price)  # Ensure no zero values for log returns
    

    data = clean_numeric_data(data, ['Nikkei_spot', 'NKY_30D_ivol', 'NKY_Div_yield', 'Basic_Loan_Rate_JPY'])
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
        time_to_maturity = days_to_expiry / 365.0
        
        # Ensure we have valid days to expiry and stock price
        if days_to_expiry <= 0 or np.isnan(data.loc[current_date, 'Nikkei_spot']):
            continue
        
        # Calculate strike price 
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        data.loc[roll_date, 'daily_returns'] = data.loc[roll_date, 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    return {'NAV_nikkei_asian_put': nav, 'LogReturn_nikkei_asian_put': data['log_returns']}

def price_ford_cds(data): # 5Y Ford CDS

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
    
    data = clean_numeric_data(data, ['5_Y_ford_credit_spread', '5y_treasury_yield'])
    data['5_Y_ford_credit_spread'] = data['5_Y_ford_credit_spread'] / 100
    data['5y_treasury_yield'] = data['5y_treasury_yield'] / 100
    
    # Initialize variables for CDS valuation
    data['cds_value'] = np.nan
    data['hazard_rate'] = np.nan
    data['days_to_maturity'] = MATURITY * 365
    
    # Calculate CDS values
    for i in range(len(data)):
        current_date = data.index[i]
        
        # Calculate hazard rate from spread - approximation
        if not np.isnan(data.loc[current_date, '5_Y_ford_credit_spread']):
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
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_ford_cds': nav, 'LogReturn_ford_cds': data['log_returns']}

def price_spx_barrier_option(data): # 1M S&P 500 Knock-Out Call Option

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

        np.random.seed(42) # Set random seed for reproducibility
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
    

    data = clean_numeric_data(data, ['sp500_index', 'SPX_Div_yield', 'vix_index_level', 'fed_funds_rate'])
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        prev_date_idx = data.index.get_loc(roll_date) - 1
        prev_date = data.index[prev_date_idx]
        
        prev_price = data.loc[prev_date, 'option_price']
        curr_price = data.loc[roll_date, 'option_price']
        
        if prev_price > 0:
            data.loc[roll_date, 'daily_returns'] = (curr_price - prev_price) / prev_price - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if data.loc[current_date, 'roll_date']:
            # Reset NAV to NOTIONAL on every roll date to avoid astronomic growth
            nav.loc[current_date] = NOTIONAL
        else:
            ret = data.loc[current_date, 'daily_returns']
            new_val = nav.loc[prev_date] * (1 + ret) if not np.isnan(ret) else nav.loc[prev_date]
            nav.loc[current_date] = max(new_val, MIN_NAV)  # Apply minimum value to avoid extremely small numbers
    
    return {'NAV_spx_knockout_call': nav, 'LogReturn_spx_knockout_call': data['log_returns']}

# FOREX PRICING 

def price_gbpusd_6m_forward(data): # 6M GBP/USD Forward Contract

    NOTIONAL_USD = 100.0  # Notional amount in USD
    DAYS_IN_YEAR = 365
    DAYS_FORWARD = 182  # Approximately 6 months
    
    def get_next_expiry(date):
        return date + timedelta(days=DAYS_FORWARD)
    
    data = clean_numeric_data(data, ['fx_gbpusd_rate', 'fed_funds_rate', 'factor_GBP_sonia'])
    data_fx = data.copy()
    data_fx.rename(columns={
        'fx_gbpusd_rate': 'spot',
        'fed_funds_rate': 'usd_rate',
        'factor_GBP_sonia': 'gbp_rate'
    }, inplace=True) # renaming for clarity
    data_fx['usd_rate'] = data_fx['usd_rate'] / 100
    data_fx['gbp_rate'] = data_fx['gbp_rate'] / 100
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
            if current_contract is not None:
                data_fx.loc[current_date, 'roll_date'] = True
            
            spot = data_fx.loc[current_date, 'spot']
            r_gbp = data_fx.loc[current_date, 'gbp_rate']
            r_usd = data_fx.loc[current_date, 'usd_rate']
            T = DAYS_FORWARD / DAYS_IN_YEAR

            forward_rate = spot * (1 + r_usd * T) / (1 + r_gbp * T)
            
            current_contract = {
                'entry_date': current_date,
                'expiry_date': get_next_expiry(current_date),
                'forward_rate': forward_rate,
                'notional_usd': NOTIONAL_USD
            }
            
            data_fx.loc[current_date, 'active_contract'] = True
        
        data_fx.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
        data_fx.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
        
        # Calculate unrealized P&L (mark-to-market)
        # For a GBP/USD forward, buying GBP and selling USD: P&L = Notional_USD * (1/spot - 1/forward_rate)
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_USD * (1/spot - 1/forward_rate)
        data_fx.loc[current_date, 'contract_pnl'] = pnl
    
    # Calculate daily returns based on P&L changes
    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_USD
    if len(data_fx) > 0:
        data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav = pd.Series(index=data_fx.index)
    if len(nav) > 0:
        first_date = data_fx.index[0]
        nav.loc[first_date] = 100  # Start with 100
        
        for i in range(1, len(nav)):
            current_date = data_fx.index[i]
            prev_date = data_fx.index[i-1]
            daily_return = data_fx.loc[current_date, 'daily_return']
            nav.loc[current_date] = nav.loc[prev_date] * (1 + daily_return)
    
    log_returns = np.log(nav / nav.shift(1))
    
    return {'NAV_gbpusd_6m_forward': nav, 'LogReturn_gbpusd_6m_forward': log_returns}

def price_usdinr_3m_forward(data): # 3M USD/INR Forward Contract - Short Position

    NOTIONAL_USD = 100.0  # Notional amount in USD
    DAYS_IN_YEAR = 365
    DAYS_FORWARD = 91  # Approximately 3 months
    
    def get_next_expiry(date):
        return date + timedelta(days=DAYS_FORWARD)
    
    data = clean_numeric_data(data, ['USD_INR', 'fed_funds_rate', 'MIBOR'])
    data_fx = data.copy()
    data_fx.rename(columns={
        'USD_INR': 'spot',
        'fed_funds_rate': 'usd_rate',
        'MIBOR': 'inr_rate'
    }, inplace=True)
    
    data_fx['usd_rate'] = data_fx['usd_rate'] / 100
    data_fx['inr_rate'] = data_fx['inr_rate'] / 100
    data_fx = data_fx.dropna(subset=['spot', 'usd_rate', 'inr_rate'])
    
    # Initialize columns for forward analysis
    data_fx['forward_rate'] = np.nan
    data_fx['days_to_expiry'] = np.nan
    data_fx['roll_date'] = False
    data_fx['contract_pnl'] = np.nan  # P&L of each individual contract
    data_fx['active_contract'] = False  # Flag for the currently active contract
    
    # Create NAV series starting at 100
    nav_series = pd.Series(index=data_fx.index, data=np.nan)
    
    if len(data_fx) > 0:
        first_date = data_fx.index[0]
        nav_series.loc[first_date] = NOTIONAL_USD
    
    entry_dates = []
    forward_rates = []
    expiry_dates = []
    
    # Process each date sequentially
    current_contract = None
    for i in range(len(data_fx)):
        current_date = data_fx.index[i]
        
        if current_contract is None or current_date >= current_contract['expiry_date']:
            if current_contract is not None:
                data_fx.loc[current_date, 'roll_date'] = True
            
            spot = data_fx.loc[current_date, 'spot']
            r_inr = data_fx.loc[current_date, 'inr_rate']
            r_usd = data_fx.loc[current_date, 'usd_rate']
            T = DAYS_FORWARD / DAYS_IN_YEAR
            
            # Forward rate formula: Spot * (1 + r_base) / (1 + r_foreign): For USD/INR, USD is the base and INR is the foreign currency
            forward_rate = spot * (1 + r_usd * T) / (1 + r_inr * T)
            
            current_contract = {
                'entry_date': current_date,
                'expiry_date': current_date + timedelta(days=DAYS_FORWARD),
                'forward_rate': forward_rate,
                'notional_usd': NOTIONAL_USD
            }
            
            # Store dates and rates
            entry_dates.append(current_date)
            forward_rates.append(forward_rate)
            expiry_dates.append(current_contract['expiry_date'])
            data_fx.loc[current_date, 'active_contract'] = True
        
        # Update data for the current date
        data_fx.loc[current_date, 'forward_rate'] = current_contract['forward_rate']
        data_fx.loc[current_date, 'days_to_expiry'] = (current_contract['expiry_date'] - current_date).days
        
        # Calculate unrealized P&L (mark-to-market)
        # For a USD/INR forward, selling USD and buying INR: P&L = Notional_USD * (forward_rate/spot - 1)
        spot = data_fx.loc[current_date, 'spot']
        forward_rate = current_contract['forward_rate']
        pnl = NOTIONAL_USD * (forward_rate/spot - 1)  # Adjusted for USD/INR forward
        data_fx.loc[current_date, 'contract_pnl'] = pnl

    data_fx['daily_return'] = data_fx['contract_pnl'].diff() / NOTIONAL_USD
    
    if len(data_fx) > 0:
        data_fx.loc[data_fx.index[0], 'daily_return'] = 0  # First day has no return
    data_fx['daily_return'] = data_fx['daily_return'].fillna(0)
    
    # Compute cumulative NAV
    nav = pd.Series(index=data_fx.index)
    
    # Initialize and compute NAV if there are any rows
    if len(data_fx) > 0:
        first_date = data_fx.index[0]
        nav.loc[first_date] = NOTIONAL_USD  # Start with 100 USD
        
        for i in range(1, len(data_fx)):
            current_date = data_fx.index[i]
            prev_date = data_fx.index[i-1]
            daily_return = data_fx.loc[current_date, 'daily_return']
            nav.loc[current_date] = nav.loc[prev_date] * (1 + daily_return)
    
    # Compute log returns of NAV
    log_returns = np.log(nav / nav.shift(1))
    
    return {'NAV_usdinr_3m_forward': nav, 'LogReturn_usdinr_3m_forward': log_returns}

def price_eurusd_atm_call_option(data): # 1M EUR/USD ATM Call Option

    NOTIONAL = 100
    DAYS_TO_EXPIRY = 30  # Approximately 1 month
    TRANSACTION_COST = 0.0005  # 5 bps per transaction
    
    # Garman-Kohlhagen formula for FX call option pricing (extension of Black-Scholes for FX options)
    def garman_kohlhagen_call(S, K, T, r_d, r_f, sigma):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        return max(0.0001, call)  # Ensure no zero values for log returns
    
    data = clean_numeric_data(data, ['fx_eurusd_rate', 'EUR-USD_IVOL', 'fed_funds_rate', 'Euro_STR'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100  # USD interest rate
    data['Euro_STR'] = data['Euro_STR'] / 100  # EUR interest rate
    data['EUR-USD_IVOL'] = data['EUR-USD_IVOL'] / 100  # Implied volatility
    
    # Initialize variables
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        data.loc[roll_date, 'daily_returns'] = data.loc[roll_date, 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_eurusd_atm_call': nav, 'LogReturn_eurusd_atm_call': data['log_returns']}

def price_usdjpy_atm_put_option(data): # 3M USD/JPY ATM Put Option

    NOTIONAL = 100
    DAYS_TO_EXPIRY = 90  # Approximately 3 months
    TRANSACTION_COST = 0.0005  # 5 bps per transaction
    
    # Garman-Kohlhagen formula for FX put option pricing
    def garman_kohlhagen_put(S, K, T, r_d, r_f, sigma):
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        return max(1.0, put)  # Set lower bound to 1.0
    
    data = clean_numeric_data(data, ['fx_usdjpy_rate', 'USD-JPY_IVOL', 'fed_funds_rate', 'Basic_Loan_Rate_JPY'])
    data['fed_funds_rate'] = data['fed_funds_rate'] / 100  # USD interest rate
    data['Basic_Loan_Rate_JPY'] = data['Basic_Loan_Rate_JPY'] / 100  # JPY interest rate
    data['USD-JPY_IVOL'] = data['USD-JPY_IVOL'] / 100  # Implied volatility
    
    # Initialize variables
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
    roll_dates = data.index[data['roll_date'] == True][1:]  # Skip first date if it's a roll date
    for roll_date in roll_dates:
        data.loc[roll_date, 'daily_returns'] = data.loc[roll_date, 'daily_returns'] - TRANSACTION_COST
    
    # Compute NAV
    nav = pd.Series(index=data.index)
    first_date = data.index[0]
    nav.loc[first_date] = NOTIONAL
    
    for i in range(1, len(nav)):
        current_date = data.index[i]
        prev_date = data.index[i-1]
        
        if not np.isnan(data.loc[current_date, 'daily_returns']):
            nav.loc[current_date] = nav.loc[prev_date] * (1 + data.loc[current_date, 'daily_returns'])
        else:
            nav.loc[current_date] = nav.loc[prev_date]
    
    return {'NAV_usdjpy_atm_put': nav, 'LogReturn_usdjpy_atm_put': data['log_returns']}

###############################################################################
#                         PORTFOLIO CONSTRUCTION                              #
###############################################################################

class Portfolio:
    def __init__(self, initial_capital=10000000.0, start_date=None):
        initial_capital = initial_capital
        start_date = pd.to_datetime(start_date) if start_date else datetime.now()
        
        # Store instance variables
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.start_date = start_date
        self.instruments = {}
        self.nav_history = pd.DataFrame(columns=['Date', 'NAV'])
        self.returns_history = pd.DataFrame(columns=['Date', 'Return'])
        
        # Pre-defined allocation totalling to 1
        self.allocation = {
            'Apple': 0.02,
            'Lockheed_martin': 0.02,
            'Nvidia': 0.02,
            'Procter_gamble': 0.02,
            'Johnson_johnson': 0.02,
            'Toyota': 0.02,
            'Nestle': 0.02,
            'X_steel': 0.02,
            '10y_treasury': 0.05,
            'Lqd_etf': 0.04,
            '10y_tips': 0.05,
            '1y_eur_zcb': 0.05,
            'High_yield_corp_debt': 0.04,
            '5y_green_bond': 0.03,
            '30y_revenue_bond': 0.04,
            'Sp500_futures_1m': 0.04,
            'Vix_futures': 0.03,
            'Crude_oil_futures': 0.03,
            'Gold_futures': 0.04,
            'Soybean_futures': 0.03,
            'Costco_itm_call': 0.03,
            'Xom_itm_put': 0.03,
            'Eurusd_atm_call': 0.03,
            'Usdjpy_atm_put': 0.03,
            'Gbpusd_6m_forward': 0.04,
            'Usdinr_3m_forward': 0.04,
            'Ford_cds': 0.03,
            'Dax_variance_swap': 0.03,
            'Nikkei_asian_put': 0.03,
            'Spx_knockout_call': 0.03,
            'Cash': 0.05
        }
        
        # Calculate initial holdings and load pricing data
        self.holdings = {name: initial_capital * alloc for name, alloc in self.allocation.items()}
        self._load_pricing_data()
        
    def _load_pricing_data(self):
        combined_file = COMBINED_RETURNS_PATH
        
        try:
            print(f"Loading data from {combined_file}")
            combined_data = pd.read_csv(combined_file, index_col='date', parse_dates=True)
            self.trading_dates = combined_data.index.unique()
            print(f"Input file contains {len(self.trading_dates)} unique trading dates")
            
            # Parse NAV and return columns
            nav_cols = [col for col in combined_data.columns if col.startswith('NAV_')]
            return_cols = [col for col in combined_data.columns if col.startswith('LogReturn_')]
            
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
                    # Create a subset with just the NAV and return columns
                    instrument_df = combined_data[[nav_col, return_col]].copy()
                    instrument_df.columns = ['NAV', 'Return']
                    
                    # Add to instruments dictionary
                    self.instruments[instrument_name] = instrument_df
                    print(f"Loaded {instrument_name} data")
                else:
                    print(f"Warning: Could not find columns for {instrument_name}")
            
        except FileNotFoundError:
            print(f"Error: Combined instrument data file not found at {combined_file}")
        except Exception as e:
            print(f"Error loading combined instrument data: {str(e)}")
    
    def calculate_returns(self, end_date=None, cash_rate=0.025):
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
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
        
        # Get the trading dates
        sample_instrument = next(iter(self.instruments.values()))
        if sample_instrument is None or sample_instrument.empty:
            print("Error: No data available for return calculation.")
            return None
            
        trading_dates = sample_instrument.index.tolist()
        print(f"Using {len(trading_dates)} trading days from the dataset")
        
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
            
            nav_record = {'Date': date, 'NAV': total_nav, 'Cash_NAV': cash_nav}
            for name, nav in instrument_navs.items():
                nav_record[f'{name}_NAV'] = nav
            
            nav_records.append(nav_record)
            
            return_records.append({
                'Date': date,
                'Return': total_return
            })

            prev_total_nav = total_nav
            prev_date = date
        
        # Create history DataFrames
        self.nav_history = pd.DataFrame(nav_records).set_index('Date')
        self.returns_history = pd.DataFrame(return_records).set_index('Date')
        
        # Update current capital to latest NAV
        self.current_capital = total_nav
        
        print(f"\nPortfolio Performance ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final NAV: ${total_nav:,.2f}")
        print(f"Number of trading days: {len(self.returns_history)}")
        
        return self.nav_history
    
    def save_results(self, folder_path=PORTFOLIO_RESULTS_DIR):            
                
        # Save NAV history
        nav_path = PORTFOLIO_NAV_HISTORY_PATH
        try:
            self.nav_history.to_csv(nav_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {nav_path}. File may be in use.")
        
        # Save returns history
        returns_path = PORTFOLIO_RETURNS_HISTORY_PATH
        try:
            self.returns_history.to_csv(returns_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {returns_path}. File may be in use.")
        
        print(f"Portfolio results saved to {folder_path}")

def plot_portfolio_performance(portfolio, save_prefix):

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
    all_results = {}
    
    # Process equities
    equity_results = price_equities(data)
    all_results.update(equity_results)

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

    # GBP/USD Forward
    gbpusd_forward_results = price_gbpusd_6m_forward(data)
    all_results.update(gbpusd_forward_results)
    
    # USD/INR Forward
    usdinr_forward_results = price_usdinr_3m_forward(data)
    all_results.update(usdinr_forward_results)
    
    # Convert results to DataFrame and ensure the index is properly set
    results_df = pd.DataFrame(all_results, index=data.index)
    nav_cols = [col for col in results_df.columns if col.startswith('NAV_')]
    
    for nav_col in nav_cols:
        instrument = nav_col.replace('NAV_', '')
        log_col = f"LogReturn_{instrument}"
        if log_col not in results_df.columns:
            # Calculate log returns directly from NAV values
            results_df[log_col] = np.log(results_df[nav_col] / results_df[nav_col].shift(1))
    
    # Output to CSV
    results_df.to_csv(COMBINED_RETURNS_PATH)

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

    print("Loading market data...")
    data = load_data()
    if data is None:
        print("Failed to load market data. Exiting.")
        return
    
    print(f"Loaded market data with {len(data)} rows")
    
    # Process all instruments and output combined results
    results = process_all_instruments(data)
    print("Pricing processing complete")
    
    # Output directory
    output_dir = PORTFOLIO_RESULTS_DIR
    
    # Create portfolio - use dates that we have full data for
    print("\nCreating and calculating portfolio...")
    portfolio = Portfolio( initial_capital=10000000.0)
    
    # Calculate returns through end of 2024
    portfolio.calculate_returns(end_date='2024-12-31')
    
    # Plot performance and save to output directory
    plot_portfolio_performance(portfolio, PORTFOLIO_RESULTS_DIR)
    
    # Save results to output directory
    portfolio.save_results(output_dir)
    
    print("Portfolio processing complete.")

if __name__ == "__main__":
    main() 