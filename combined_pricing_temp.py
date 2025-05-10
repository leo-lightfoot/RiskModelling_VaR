import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm  # Required for option pricing

###############################################################################
#                         DATA LOADING AND CLEANING                           #
###############################################################################

# Load the data
def load_data():
    """Load market data from the CSV file"""
    try:
        data = pd.read_csv(r'data_restructured.csv')
        
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

###############################################################################
#                                EQUITY PRICING                               #
###############################################################################

def price_equities(data):
    """Price equity securities and return NAV and log returns"""
    # Identify the equity columns
    equity_columns = [col for col in data.columns if col.startswith('eq_')]
    
    # Clean the equity data
    data = clean_numeric_data(data, equity_columns)
    
    # Results dictionary to store NAVs and log returns
    results = {}
    
    # Process each equity
    for col in equity_columns:
        # Calculate NAV and log returns
        equity_name = col.replace('eq_', '')
        nav, log_returns = calculate_nav_and_returns(data[col])
        
        # Store results
        results[f"NAV_{equity_name}"] = nav
        results[f"LogReturn_{equity_name}"] = log_returns
    
    return results

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

def price_10y_treasury_bond(data):
    """Price 10-year Treasury bond and return NAV and log returns"""
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not np.isnan(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a more standardized naming for log returns
    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    
    return {'NAV_10y_treasury': nav, 'LogReturn_10y_treasury': log_returns}

def price_lqd_etf(data):
    """Price LQD Corporate Bond ETF and return NAV and log returns"""
    # Identify the LQD ETF column
    lqd_column = 'lqd_corporate_bond_etf'
    
    # Clean the LQD ETF data
    data = clean_numeric_data(data, [lqd_column])
    
    # Calculate NAV and log returns
    nav, log_returns = calculate_nav_and_returns(data[lqd_column])
    
    # Create a more standardized log_returns series
    log_returns_series = pd.Series(log_returns.values, index=data.index)
    
    return {'NAV_lqd_etf': nav, 'LogReturn_lqd_etf': log_returns_series}

def price_10y_tips(data):
    """Price 10-year TIPS and return NAV and log returns"""
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
    nav = pd.Series(index=data.index)
    nav.iloc[0] = NOTIONAL
    for i in range(1, len(nav)):
        if not pd.isna(data['daily_returns'].iloc[i]):
            nav.iloc[i] = nav.iloc[i-1] * (1 + data['daily_returns'].iloc[i])
        else:
            nav.iloc[i] = nav.iloc[i-1]
    
    # Create a named log_returns series
    log_returns = pd.Series(data['log_returns'].values, index=data.index)
    
    return {'NAV_10y_tips': nav, 'LogReturn_10y_tips': log_returns}

def price_1y_eur_zcb(data):
    """Price 1-year EUR Zero Coupon Bond and return NAV and log returns"""
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
    
    # Adjust returns on roll dates to account for roll yield
    for i in range(1, len(data)):
        if data['roll_date'].iloc[i]:
            # Calculate roll yield (difference between old and new bond)
            roll_yield = (data['zcb_price_eur'].iloc[i] - data['zcb_price_eur'].iloc[i-1]) / data['zcb_price_eur'].iloc[i-1]
            data.loc[data.index[i], 'daily_returns_eur'] = roll_yield
    
    # Compute NAV in EUR
    nav_eur = pd.Series(index=data.index)
    nav_eur.iloc[0] = NOTIONAL
    for i in range(1, len(nav_eur)):
        if not np.isnan(data['daily_returns_eur'].iloc[i]):
            nav_eur.iloc[i] = nav_eur.iloc[i-1] * (1 + data['daily_returns_eur'].iloc[i])
        else:
            nav_eur.iloc[i] = nav_eur.iloc[i-1]
    
    # Convert NAV to USD
    nav_usd = pd.Series(index=data.index)
    nav_usd = nav_eur * data['fx_eurusd_rate']
    
    # Calculate USD returns
    data['daily_returns_usd'] = nav_usd.pct_change(fill_method=None)
    data['log_returns_usd'] = np.log(nav_usd / nav_usd.shift(1))
    
    # Create a log_returns series for consistency
    log_returns = pd.Series(data['log_returns_usd'].values, index=data.index)
    
    return {'NAV_1y_eur_zcb': nav_usd, 'LogReturn_1y_eur_zcb': log_returns}

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
        return max(0.0001, put)  # Ensure no zero values for log returns
    
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
    NUM_SIMULATIONS = 1000  # Number of simulations for Monte Carlo
    
    # Monte Carlo simulation for Asian put option pricing
    def price_asian_put_mc(S0, K, r, q, sigma, T, steps=63, paths=1000):
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
            price = black_scholes_call(spot, K, T, r, sigma, q)
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
    results_df.to_csv('combined_instrument_returns.csv')
    
    # Now check what was actually written to the CSV
    try:
        # This needs to be done after saving to ensure we check what was actually saved
        import csv
        with open('combined_instrument_returns.csv', 'r') as f:
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
    """Main function to run all pricing operations"""
    print("Loading market data...")
    data = load_data()
    if data is None:
        print("Failed to load market data. Exiting.")
        return
    
    print(f"Loaded market data with {len(data)} rows")
    
    # Process all instruments and output combined results
    results = process_all_instruments(data)
    print("Processing complete")

if __name__ == "__main__":
    main() 