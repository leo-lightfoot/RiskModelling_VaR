import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm, multivariate_normal, multivariate_t, binom, chi2, rankdata, logistic, laplace
from sklearn.covariance import EmpiricalCovariance
from datetime import datetime, timedelta

DATA_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/data_restructured.csv"
PORTFOLIO_RETURNS_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_returns_history.csv"
PORTFOLIO_NAV_URL = r"https://raw.githubusercontent.com/leo-lightfoot/RiskModelling_VaR/refs/heads/main/portfolio_results/portfolio_nav_history.csv"

def load_data(url):
    """Simple function to load data with parse_dates"""
    return pd.read_csv(url, parse_dates=['date' if 'data_restructured' in url else 'Date'])

# -----------------------------------------
#  SIMPLIFIED PRICING FUNCTIONS FOR MC VaR
# -----------------------------------------

class SimplifiedPricing:
    """ Provides simplified pricing functions for Monte Carlo VaR simulation. """
    
    def __init__(self):
        # Constants used by multiple pricing functions
        self.DAYS_IN_YEAR = 365
    
    def calculate_bond_price(self, yield_rate, coupon_rate, years_to_maturity, frequency=2, notional=100):
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
    
    def price_10y_treasury_bond(self, yield_rate, days_to_maturity=None):
        """ Price a 10-year Treasury bond """
        NOTIONAL = 100
        COUPON_RATE = 0.02  # 2% annual
        FREQUENCY = 2  # Semiannual payments
        
        if days_to_maturity is None:
            years_to_maturity = 10
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        return self.calculate_bond_price(
            yield_rate=yield_rate,
            coupon_rate=COUPON_RATE,
            years_to_maturity=years_to_maturity,
            frequency=FREQUENCY,
            notional=NOTIONAL
        )
    
    def price_tips(self, real_yield, inflation_factor, days_to_maturity=None):
        """ Price a Treasury Inflation-Protected Security (TIPS) """
        NOTIONAL = 100
        COUPON_RATE = 0.0125  # 1.25% annual
        FREQUENCY = 2  # Semiannual payments
        
        if days_to_maturity is None:
            years_to_maturity = 10
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        if years_to_maturity <= 0:
            return NOTIONAL * inflation_factor
        
        # Adjusted principal for inflation
        adj_principal = NOTIONAL * inflation_factor
        
        # Calculate payments
        periods = int(years_to_maturity * FREQUENCY)
        period_yield = real_yield / FREQUENCY
        period_coupon = COUPON_RATE / FREQUENCY
        coupon_payment = adj_principal * period_coupon
        
        # Present value of coupon payments
        if abs(period_yield) < 1e-10:  # Handle zero yield case
            coupon_pv = coupon_payment * periods
        else:
            coupon_pv = coupon_payment * (1 - (1 + period_yield)**(-periods)) / period_yield
        
        # Present value of principal
        principal_pv = adj_principal / (1 + period_yield)**periods
        
        return coupon_pv + principal_pv
    
    def price_corporate_bond(self, treasury_yield, credit_spread, days_to_maturity=None, coupon_rate=0.04, maturity_years=5):
        """ Price a corporate bond based on Treasury yield plus credit spread """
        NOTIONAL = 100
        FREQUENCY = 2  # Semiannual payments
        
        # Calculate effective yield
        effective_yield = treasury_yield + credit_spread
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        return self.calculate_bond_price(
            yield_rate=effective_yield,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            frequency=FREQUENCY,
            notional=NOTIONAL
        )
    
    def price_green_bond(self, treasury_yield, days_to_maturity=None, coupon_rate=0.025, greenium=-0.002, maturity_years=5, frequency=1):
        """ Price a green bond, accounting for greenium (green premium)  """
        NOTIONAL = 100
        
        # Calculate effective yield (Treasury + greenium)
        effective_yield = treasury_yield + greenium
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        return self.calculate_bond_price(
            yield_rate=effective_yield,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
            notional=NOTIONAL
        )
    
    def price_revenue_bond(self, treasury_yield, days_to_maturity=None, coupon_rate=0.04, spread=0.0075, maturity_years=30, frequency=2):
        """ Price a revenue bond with spread over treasury  """
        NOTIONAL = 100
        
        # Calculate effective yield (Treasury + spread)
        effective_yield = treasury_yield + spread
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        return self.calculate_bond_price(
            yield_rate=effective_yield,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
            notional=NOTIONAL
        )
    
    def price_zero_coupon_bond(self, yield_rate, days_to_maturity=None, maturity_years=1):
        """  Price a zero-coupon bond   """
        NOTIONAL = 100
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        # Using continuous compounding for simplicity
        return NOTIONAL * np.exp(-yield_rate * years_to_maturity)
    
    def price_high_yield_corp_debt(self, treasury_yield, credit_spread, days_to_maturity=None, 
                                  coupon_rate=0.065, maturity_years=5, frequency=2):
        """  Price high yield corporate debt instrument  """
        NOTIONAL = 100
        
        # Calculate the high yield rate (treasury + high yield spread)
        high_yield_rate = treasury_yield + credit_spread
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        return self.calculate_bond_price(
            yield_rate=high_yield_rate,
            coupon_rate=coupon_rate,
            years_to_maturity=years_to_maturity,
            frequency=frequency,
            notional=NOTIONAL
        )
        
    def price_equity_futures(self, spot_price, risk_free_rate, dividend_yield, days_to_expiry):
        """ Price an equity futures contract """
        time_to_maturity = days_to_expiry / self.DAYS_IN_YEAR
        
        # F = S * e^((r-q)*T)
        futures_price = spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)
        
        return futures_price
    
    def price_vix_futures(self, vix_index_level, vix_history=None, days_to_expiry=21, mean_reversion_speed=0.2, vol_of_vol=None):
        """ Price VIX futures using a mean-reverting model  """
        time_to_maturity = days_to_expiry / self.DAYS_IN_YEAR
        
        # Calculate the 30-day moving average if history is provided
        if vix_history is not None and len(vix_history) >= 30:
            vix_mean = vix_history[-30:].mean()
            
            # Calculate volatility if not provided
            if vol_of_vol is None:
                # Calculate as percentage change std * mean (as in vix_front_month.py)
                pct_changes = np.diff(vix_history[-31:]) / vix_history[-31:-1]
                vol_of_vol = np.std(pct_changes) * vix_history[-30:].mean()
        else:
            # Without history, use spot as mean and default volatility
            vix_mean = vix_index_level
            if vol_of_vol is None:
                vol_of_vol = 0.3  # Default value
        
        # Simple mean-reverting model for VIX futures
        # F = S * e^(-θ*T) + M * (1 - e^(-θ*T))
        # where θ is the mean reversion speed, M is the long-term mean (30-day average)
        vix_futures_1m = vix_index_level * np.exp(-mean_reversion_speed * time_to_maturity) + \
                          vix_mean * (1 - np.exp(-mean_reversion_speed * time_to_maturity))
        
        return vix_futures_1m
    
    def price_commodity_futures(self, spot_price, risk_free_rate, storage_cost, convenience_yield, days_to_expiry):
        """ Price a commodity futures contract """
        time_to_maturity = days_to_expiry / self.DAYS_IN_YEAR
        
        # F = S * e^((r + u - y)*T) where u = storage cost, y = convenience yield
        futures_price = spot_price * np.exp((risk_free_rate + storage_cost - convenience_yield) * time_to_maturity)
        
        return futures_price
    
    def price_gold_futures(self, spot_price, risk_free_rate, storage_cost=0.005, days_to_expiry=90):
        """ Price gold futures contract """
        # Gold typically has minimal convenience yield, so we set it to 0
        return self.price_commodity_futures(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            storage_cost=storage_cost,
            convenience_yield=0,
            days_to_expiry=days_to_expiry
        )
    
    def price_crude_oil_futures(self, spot_price, risk_free_rate, days_to_expiry=30,
                              storage_cost=0.02/12, convenience_yield=0.01/12):
        """ Price crude oil futures contract """
        return self.price_commodity_futures(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            storage_cost=storage_cost,
            convenience_yield=convenience_yield,
            days_to_expiry=days_to_expiry
        )
    
    def price_soybean_futures(self, spot_price, risk_free_rate, days_to_expiry, 
                            month=None, base_storage_cost=0.04, seasonal_amplitude=0.02):
        """ Price soybean futures with seasonal storage costs """
        if month is None:
            month = datetime.now().month
        
        phase_shift = 10
        seasonal_storage_cost = base_storage_cost + seasonal_amplitude * np.cos(2 * np.pi * (month - phase_shift) / 12)
        
        convenience_yield = 0.02
        
        return self.price_commodity_futures(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            storage_cost=seasonal_storage_cost,
            convenience_yield=convenience_yield,
            days_to_expiry=days_to_expiry
        )
    
    def black_scholes_call(self, spot, strike, days_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """Price a European call option using Black-Scholes-Merton model  """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        call_price = spot * np.exp(-dividend_yield * T) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(self, spot, strike, days_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """ Price a European put option using Black-Scholes-Merton model  """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        put_price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * T) * norm.cdf(-d1)
        
        return put_price
    
    def garman_kohlhagen_call(self, spot, strike, days_to_expiry, domestic_rate, foreign_rate, volatility):
        """ Price a European FX call option using Garman-Kohlhagen model  """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        call_price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        
        return max(0.0001, call_price)
    
    def garman_kohlhagen_put(self, spot, strike, days_to_expiry, domestic_rate, foreign_rate, volatility):
        """Price a European FX put option using Garman-Kohlhagen model  """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        put_price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
        
        return put_price
     
    def price_fx_forward(self, spot_rate, domestic_rate, foreign_rate, days_to_expiry, notional=100.0):
        """ Price an FX forward contract """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        # Forward rate formula: Spot * (1 + r_foreign) / (1 + r_base)
        forward_rate = spot_rate * (1 + domestic_rate * T) / (1 + foreign_rate * T)
        
        # Calculate unrealized P&L (mark-to-market)
        # P&L = Notional * (1/spot - 1/forward_rate)
        pnl = notional * (1/spot_rate - 1/forward_rate)
        
        return {
            'forward_rate': forward_rate,
            'contract_pnl': pnl,
            'days_to_expiry': days_to_expiry
        }
       
    def price_cds(self, credit_spread, risk_free_rate, current_date=None, previous_date=None, 
                maturity_years=5, recovery_rate=0.4, payments_per_year=4, roll_frequency_days=365, 
                notional=100, transaction_cost=0.002, previous_value=None):
        """ Price a Credit Default Swap (CDS) with rolling strategy logic  """
        from datetime import timedelta
        
        result = {}
        
        # Check for valid input
        if np.isnan(credit_spread) or np.isnan(risk_free_rate):
            result['cds_value'] = np.nan
            result['hazard_rate'] = np.nan
            result['roll_date'] = False
            result['daily_return'] = np.nan
            result['log_return'] = np.nan
            result['nav'] = notional if previous_date is None else np.nan
            return result
            
        # Compute hazard rate from spread
        hazard_rate = credit_spread / (1 - recovery_rate)
        result['hazard_rate'] = hazard_rate
        
        # Compute CDS value using the internal pricing function
        dt = 1 / payments_per_year
        n = int(maturity_years * payments_per_year)
        discount_factors = np.exp(-risk_free_rate * dt * np.arange(1, n + 1))
        survival_probs = np.exp(-hazard_rate * dt * np.arange(1, n + 1))
        
        premium_leg = credit_spread * np.sum(discount_factors * survival_probs) * notional * dt
        protection_leg = (1 - recovery_rate) * np.sum(discount_factors * np.diff(np.insert(survival_probs, 0, 1))) * notional
        
        cds_value = premium_leg - protection_leg
        result['cds_value'] = cds_value
        
        # Roll date logic
        result['roll_date'] = False
        
        if current_date is not None and previous_date is not None:
            # Check if the current date is a roll date based on roll frequency
            next_roll_date = previous_date + timedelta(days=roll_frequency_days)
            if current_date >= next_roll_date:
                result['roll_date'] = True
                
            # Calculate returns if we have the previous value
            if previous_value is not None and not np.isnan(previous_value) and previous_value > 0:
                gross_return = (cds_value - previous_value) / previous_value
                
                # Apply transaction cost on roll dates
                if result['roll_date']:
                    gross_return -= transaction_cost
                    
                result['daily_return'] = gross_return
                result['log_return'] = np.log(cds_value / previous_value)
                
                # Calculate NAV (assume previous_nav passed in if needed)
                result['nav'] = notional * (1 + gross_return)
            else:
                result['daily_return'] = 0
                result['log_return'] = 0
                result['nav'] = notional
        else:
            # For the first date, no return calculation
            result['daily_return'] = 0
            result['log_return'] = 0
            result['nav'] = notional
            
        return result
    
    def price_variance_swap(self, call_ivol, put_ivol, current_date=None, previous_date=None, maturity_days=30, 
                          notional=100, annual_basis=365.0, transaction_cost=0.002):
        """ Price a variance swap with rollover logic  """
        from datetime import timedelta
        
        result = {}
        
        # Check for valid input
        if np.isnan(call_ivol) or np.isnan(put_ivol):
            result['fixed_variance'] = np.nan
            result['fixed_vol'] = np.nan
            result['daily_return'] = 0.0
            result['roll_date'] = False
            result['days_to_expiry'] = 0
            result['nav'] = notional if previous_date is None else np.nan
            return result
        
        # Calculate fixed leg (variance strike) as average squared vol
        avg_ivol = 0.5 * (call_ivol + put_ivol)
        fixed_var = avg_ivol ** 2
        
        result['fixed_variance'] = fixed_var
        result['fixed_vol'] = avg_ivol
        
        # Roll date logic
        result['roll_date'] = False
        result['days_to_expiry'] = maturity_days
        
        if current_date is not None:
            if previous_date is not None:

                current_expiry = previous_date + timedelta(days=maturity_days)
                if current_date >= current_expiry:
                    result['roll_date'] = True
                    # Reset expiry from current date
                    current_expiry = current_date + timedelta(days=maturity_days)
            else:
                # If this is the first date, set expiry
                current_expiry = current_date + timedelta(days=maturity_days)
                
            # Calculate days to expiry
            days_to_expiry = (current_expiry - current_date).days
            result['days_to_expiry'] = days_to_expiry
        
        # Calculate daily return from fixed variance (matching original code)
        daily_return = fixed_var / annual_basis
        
        # Apply transaction cost on roll dates
        if result['roll_date']:
            daily_return -= transaction_cost
            
        result['daily_return'] = daily_return
        result['log_return'] = np.log(1 + daily_return)
        
        # Calculate NAV if previous NAV provided
        if previous_date is not None:
            result['nav'] = notional * (1 + daily_return) 
        else:
            result['nav'] = notional
            
        return result
    
    
    def price_asian_option(self, spot, strike=None, risk_free_rate=0, dividend_yield=0, volatility=0, 
                          current_date=None, previous_date=None, option_type='put',
                          days_in_option=63, paths=2000, notional=100, seed=42):
        """ Price an Asian option with rollover logic using Monte Carlo simulation  """
        from datetime import timedelta
        import numpy as np
        
        result = {}
        
        if current_date is None:
            current_date = datetime.today()

        if np.isnan(spot) or np.isnan(volatility) or np.isnan(risk_free_rate) or np.isnan(dividend_yield):
            result['option_price'] = np.nan
            result['days_to_maturity'] = 0
            result['roll_date'] = False
            result['daily_return'] = np.nan
            result['log_return'] = np.nan
            result['nav'] = notional
            return result
            
        if strike is None:
            strike = spot
            
        result['strike'] = strike
        
        result['roll_date'] = False
        days_to_maturity = days_in_option
        
        if current_date is not None:
            if previous_date is not None:
                if hasattr(previous_date, 'current_end'):
                    previous_end = previous_date.current_end
                else:
                    previous_end = previous_date + timedelta(days=days_in_option - 1)
                    
                if current_date > previous_end:
                    result['roll_date'] = True
                    current_end = current_date + timedelta(days=days_in_option - 1)
                else:
                    current_end = previous_end
                
                days_to_maturity = (current_end - current_date).days + 1
            else:
                result['roll_date'] = True
                days_to_maturity = days_in_option
                
        result['days_to_maturity'] = days_to_maturity
        
        if seed is not None:
            np.random.seed(seed)
        
        T = days_to_maturity / 252.0
        
        steps = min(days_to_maturity, days_in_option)
        
        if steps <= 0 or T <= 0:
            if option_type.lower() == 'put':
                result['option_price'] = max(strike - spot, 0)
            else:
                result['option_price'] = max(spot - strike, 0)
        else:
            dt = T / steps
            drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
            diffusion = volatility * np.sqrt(dt)
            
            Z = np.random.randn(paths, steps)
            price_paths = spot * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
            
            average_price = np.mean(price_paths, axis=1)
            
            if option_type.lower() == 'put':
                payoff = np.maximum(strike - average_price, 0)
            else:
                payoff = np.maximum(average_price - strike, 0)
            
            option_price = np.exp(-risk_free_rate * T) * np.mean(payoff)
            result['option_price'] = option_price
        
        if previous_date is not None and 'previous_price' in previous_date and not np.isnan(previous_date.previous_price):
            previous_price = previous_date.previous_price
            
            if previous_price > 0:
                daily_return = (result['option_price'] - previous_price) / previous_price
                result['daily_return'] = daily_return
                result['log_return'] = np.log(result['option_price'] / previous_price)
                
                result['nav'] = notional * (1 + daily_return)
            else:
                result['daily_return'] = 0
                result['log_return'] = 0
                result['nav'] = notional
        else:
            result['daily_return'] = 0
            result['log_return'] = 0
            result['nav'] = notional
            
        current_end = current_date + timedelta(days=days_in_option - 1)
        previous_price = result['option_price']
            
        return result
    
    def price_barrier_option(self, spot, strike=None, risk_free_rate=0, dividend_yield=0, volatility=0, 
                           current_date=None, previous_date=None, barrier_multiplier=1.1,
                           maturity_days=30, option_type='call', barrier_type='knockout',
                           notional=100, transaction_cost=0.003, annual_basis=365.0):
        """ Price a barrier option with rollover logic  """
        from datetime import timedelta
        
        result = {}
        
        if np.isnan(spot) or np.isnan(volatility) or np.isnan(risk_free_rate) or np.isnan(dividend_yield):
            result['option_price'] = np.nan
            result['strike_price'] = np.nan
            result['barrier_level'] = np.nan
            result['days_to_expiry'] = 0
            result['roll_date'] = False
            result['knocked_out'] = False
            result['daily_return'] = np.nan
            result['log_return'] = np.nan
            result['nav'] = notional
            return result
            
        if strike is None:
            strike = spot
            
        result['strike_price'] = strike
        
        barrier_level = barrier_multiplier * strike
        result['barrier_level'] = barrier_level
        
        result['knocked_out'] = False
        if barrier_type.lower() == 'knockout':
            if (option_type.lower() == 'call' and spot >= barrier_level) or \
               (option_type.lower() == 'put' and spot <= barrier_level):
                result['knocked_out'] = True
        
        result['roll_date'] = False
        if current_date is not None and previous_date is not None:
            days_since_previous = (current_date - previous_date).days
            if days_since_previous >= maturity_days:
                result['roll_date'] = True
        
        days_to_expiry = maturity_days
        if current_date is not None and previous_date is not None:
            days_to_expiry = max(0, maturity_days - (current_date - previous_date).days)
        result['days_to_expiry'] = days_to_expiry
        
        if result['knocked_out']:
            result['option_price'] = 0.0
        else:
            T = days_to_expiry / annual_basis
            if option_type.lower() == 'call':
                result['option_price'] = self.black_scholes_call(
                    spot=spot,
                    strike=strike,
                    days_to_expiry=days_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield
                )
            else:
                result['option_price'] = self.black_scholes_put(
                    spot=spot,
                    strike=strike,
                    days_to_expiry=days_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=dividend_yield
                )
        
        if previous_date is not None:
            if result['roll_date']:
                result['daily_return'] = -transaction_cost
            else:
                result['daily_return'] = (result['option_price'] - result.get('previous_price', result['option_price'])) / result.get('previous_price', result['option_price'])
            
            result['log_return'] = np.log(1 + result['daily_return'])
        else:
            result['daily_return'] = 0.0
            result['log_return'] = 0.0
        
        result['previous_price'] = result['option_price']
        
        if previous_date is None:
            result['nav'] = notional
        else:
            result['nav'] = result.get('previous_nav', notional) * (1 + result['daily_return'])
        
        return result

    def price_usdinr_3m_forward(self, spot_rate, domestic_rate, foreign_rate, days_to_expiry, notional=10000.0):
        """
        Price a 3-month USD/INR forward contract (short position)
    
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        forward_rate = spot_rate * (1 + foreign_rate * T) / (1 + domestic_rate * T)
        
        pnl = notional * (forward_rate - spot_rate) / spot_rate
        
        return {
            'forward_rate': forward_rate,
            'contract_pnl': pnl,
            'days_to_expiry': days_to_expiry
        }

# Transforming Data
def empirical_cdf_transform(X):
    ranks = np.apply_along_axis(rankdata, 0, X)
    return (ranks - 0.5) / X.shape[0]

# Identifying best fit to the distribution
def best_fit_distribution(x):
    candidates = {
        't': lambda: t.fit(x),
        'logistic': lambda: logistic.fit(x),
        'laplace': lambda: laplace.fit(x)
    }
    best_score = np.inf
    best_dist = None
    best_params = None
    for name, fit_fn in candidates.items():
        try:
            params = fit_fn()
            dist = eval(name)
            logpdf = dist.logpdf(x, *params)
            score = -np.sum(logpdf)
            if score < best_score:
                best_score = score
                best_dist = name
                best_params = params
        except Exception as e:
            continue
    return best_dist, best_params

# Fitting marginals
def fit_marginals(returns_clean):
    marginals = {}
    u_data = pd.DataFrame(index=returns_clean.index, columns=returns_clean.columns)

    for col in returns_clean.columns:
        x = returns_clean[col].dropna()
        dist_name, params = best_fit_distribution(x)
        marginals[col] = (dist_name, params)
        dist = eval(dist_name)
        u_data[col] = dist.cdf(x, *params)

    u_data = u_data.astype(float).clip(1e-6, 1 - 1e-6)
    return u_data, marginals

# Regularize Correlation Matrix
def regularize_corr_matrix(corr_matrix, lambda_=1e-3):
    identity = np.eye(corr_matrix.shape[0])
    return (1 - lambda_) * corr_matrix + lambda_ * identity

# Copula likelihood and AIC
def copula_log_likelihood(copula_type, u_data, corr_matrix, df_copula=None):
    norm_data = norm.ppf(u_data)
    d = norm_data.shape[1]
    
    if copula_type == "gaussian":
        ll = multivariate_normal(mean=np.zeros(d), cov=corr_matrix).logpdf(norm_data).sum()
        k = d * (d - 1) / 2
    
    elif copula_type == "t":
        ll = multivariate_t(loc=np.zeros(d), shape=corr_matrix, df=df_copula).logpdf(norm_data).sum()
        k = d * (d - 1) / 2 + 1
    
    else:
        raise ValueError("Invalid copula type")
    
    aic = -2 * ll + 2 * k
    return ll, aic

# Gaussian copula simulation
def simulate_gaussian_copula(u_data, n_sim, ridge=1e-6):
    norm_data = norm.ppf(u_data)
    emp_cov = np.cov(norm_data.T)

    emp_cov += ridge * np.eye(emp_cov.shape[0])

    z = np.random.multivariate_normal(mean=np.zeros(emp_cov.shape[0]), cov=emp_cov, size=n_sim)
    u_sim = norm.cdf(z)
    return u_sim

# Student's t copula simulation
def simulate_student_copula(u_data, n_sim, df_copula=6, ridge=1e-6):
    u_data = empirical_cdf_transform(u_data)
    u_data = np.clip(u_data, 1e-6, 1 - 1e-6)
    norm_data = norm.ppf(u_data)

    emp_cov = np.cov(norm_data.T)
    emp_cov += ridge * np.eye(emp_cov.shape[0])

    L = np.linalg.cholesky(emp_cov)
    z = np.random.randn(n_sim, emp_cov.shape[0]) @ L.T
    g = np.random.gamma(df_copula / 2.0, 2.0 / df_copula, size=n_sim)
    t_samples = z / np.sqrt(g)[:, None]
    u_sim = t.cdf(t_samples, df_copula)

    return u_sim

# Transform to returns
def u_to_returns(u_sim, marginals):
    sim_returns = np.zeros_like(u_sim)
    for i, (col, (dist_name, params)) in enumerate(marginals.items()):
        dist = eval(dist_name)
        sim_returns[:, i] = dist.ppf(u_sim[:, i], *params)
    return pd.DataFrame(sim_returns, columns=marginals.keys())

def returns_to_levels(sim_returns_df, current_levels, 
                      log_return_factors=None, diff_factors=None, 
                      clip_bounds=(1e-4, 1e4), verbose=False):
    
    levels = pd.DataFrame(index=sim_returns_df.index, columns=sim_returns_df.columns)
    log_return_factors = log_return_factors or []
    diff_factors = diff_factors or []

    for col in sim_returns_df.columns:
        if col in log_return_factors:
            levels[col] = np.exp(sim_returns_df[col]) * current_levels[col]
        elif col in diff_factors:
            levels[col] = sim_returns_df[col] + current_levels[col]
        else:
            if verbose:
                print(f"⚠️ Warning: Column '{col}' not classified as log or diff.")
            levels[col] = sim_returns_df[col] + current_levels.get(col, 0)

    levels = levels.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    return levels

# Define the portfolio pricing function
def price_portfolio(factor_dict):
    spx = factor_dict["sp500_index"]
    gold_price = factor_dict['gold_spot_price']
    crude_oil = factor_dict['crude_oil_wti_spot']
    vix = factor_dict['vix_index_level']
    exxon_stock = factor_dict['exxonmobil_stock_price']
    costco_stock = factor_dict['costco_stock_price']
    apple_stock = factor_dict['eq_apple_price']
    lockheed_martin_stock = factor_dict['eq_lockheed_martin_price']
    nvidia_stock = factor_dict['eq_nvidia_price']
    PnG_stock = factor_dict['eq_procter_gamble_price']
    JnJ_stock = factor_dict['eq_johnson_johnson_price']
    toyota_stock = factor_dict['eq_toyota_price']
    nestle_stock = factor_dict['eq_nestle_price']
    steel_price = factor_dict['eq_x_steel_price']
    lqd_corporate_etf = factor_dict['lqd_corporate_bond_etf']
    ten_yr_trsry_yld = factor_dict['10y_treasury_yield']
    eurusd_rate = factor_dict['fx_eurusd_rate']
    usdjpy_rate = factor_dict['fx_usdjpy_rate']
    gbpusd_rate = factor_dict['fx_gbpusd_rate']
    fed_funds_rate = factor_dict['fed_funds_rate']
    soybean_spot = factor_dict['soybean_spot_usd']
    gbp_sonia = factor_dict['factor_GBP_sonia']
    high_yield_credit_spread = factor_dict['high_yield_credit spread']
    one_yr_eur_yield = factor_dict['1_year_euro_yield_curve']
    corporate_yield = factor_dict['factor_corporate_yield']
    eurusd_ivol = factor_dict['EUR-USD_IVOL']
    usdjpy_ivol = factor_dict['USD-JPY_IVOL']
    AA_corp_bond_yield = factor_dict['AA_Corp_bond_yield']
    thirty_yr_trsry_yield = factor_dict['30Y_treasury_yield']
    basic_loan_rate_jpy = factor_dict['Basic_Loan_Rate_JPY']
    mibor = factor_dict['MIBOR']
    eur_str = factor_dict['Euro_STR']
    cpi = factor_dict['CPI']
    cost_ivol_3mma = factor_dict['COST_IVOL_3MMA']
    xom_ivol_3mma = factor_dict['XOM_IVOL_3MMA']
    cost_div_yield = factor_dict['COST_DIV_YIELD']
    xom_div_yield = factor_dict['XOM_DIV_YIELD']
    usdinr = factor_dict['USD_INR']
    real_10y_yield = factor_dict['Real_10Y_yield']
    five_yr_trsry_yield = factor_dict['5y_treasury_yield']
    dax_spot = factor_dict['DAX_spot']
    dax_call_ivol_30d = factor_dict['DAX_Call_ivol_30D']
    dax_put_ivol_30d = factor_dict['DAX_Put_ivol_30D']
    five_yr_ford_credit_spread = factor_dict['5_Y_ford_credit_spread']
    nikkei_spot = factor_dict['Nikkei_spot']
    nky_30d_ivol = factor_dict['NKY_30D_ivol']
    nky_div_yield = factor_dict['NKY_Div_yield']
    spx_div_yield = factor_dict['SPX_Div_yield']
    
    pricing_model = SimplifiedPricing()

    equity_value = 10000000 * 0.02 * (apple_stock/df1['eq_apple_price'].iloc[0] +
                                      lockheed_martin_stock/df1['eq_lockheed_martin_price'].iloc[0] +
                                      nvidia_stock/df1['eq_nvidia_price'].iloc[0] +
                                      PnG_stock/df1['eq_procter_gamble_price'].iloc[0] +
                                      JnJ_stock/df1['eq_johnson_johnson_price'].iloc[0] +
                                      toyota_stock/df1['eq_toyota_price'].iloc[0] +
                                      nestle_stock/df1['eq_nestle_price'].iloc[0] +
                                      steel_price/df1['eq_x_steel_price'].iloc[0]
                                      )
    
    derivatives_value = 10000000 * (0.03 * pricing_model.price_crude_oil_futures(crude_oil,fed_funds_rate)/crude_oil +
                                    0.04 * pricing_model.price_equity_futures(spx,fed_funds_rate, 0.018, 30)/spx +
                                    0.04 * pricing_model.price_gold_futures(gold_price, fed_funds_rate)/gold_price +
                                    0.03 * pricing_model.price_commodity_futures(soybean_spot,fed_funds_rate, 0.04, 0.02, 182)/soybean_spot +
                                    0.04 * pricing_model.price_fx_forward(gbpusd_rate, fed_funds_rate, gbp_sonia, 182)['forward_rate']/gbpusd_rate +
                                    0.04 * pricing_model.price_usdinr_3m_forward(usdinr, fed_funds_rate, mibor, 90)['forward_rate']/usdinr +
                                    0.03 * pricing_model.price_vix_futures(vix, None, 30)/vix +
                                    0.03 * pricing_model.garman_kohlhagen_call(eurusd_rate, eurusd_rate, 30, fed_funds_rate, eur_str, eurusd_ivol)/eurusd_rate +
                                    0.03 * pricing_model.garman_kohlhagen_put(usdjpy_rate, usdjpy_rate, 90, fed_funds_rate, basic_loan_rate_jpy, usdjpy_ivol)/usdjpy_rate +
                                    0.03 * pricing_model.black_scholes_call(costco_stock, 0.9 * costco_stock, 90, fed_funds_rate, cost_ivol_3mma, cost_div_yield)/costco_stock +
                                    0.03 * pricing_model.black_scholes_put(exxon_stock, 1.1 * exxon_stock, 90, fed_funds_rate,xom_ivol_3mma, xom_div_yield)/exxon_stock +
                                    0.03 * pricing_model.price_cds(five_yr_ford_credit_spread, five_yr_trsry_yield)['cds_value']/100 +
                                    0.03 * pricing_model.price_variance_swap(dax_call_ivol_30d, dax_put_ivol_30d)['nav']/100 +
                                    0.03 * pricing_model.price_barrier_option(spx, None, fed_funds_rate, spx_div_yield, vix)['nav']/100 +
                                    0.03 * pricing_model.price_asian_option(nikkei_spot, None, basic_loan_rate_jpy, nky_div_yield, nky_30d_ivol)['nav']/100
                                    )

    fixed_income_value = 100000 * (0.05 * pricing_model.price_zero_coupon_bond(one_yr_eur_yield,365,1) +
                                   0.05 * pricing_model.price_10y_treasury_bond(ten_yr_trsry_yld) +
                                   0.04 * pricing_model.price_high_yield_corp_debt(ten_yr_trsry_yld,high_yield_credit_spread) +
                                   100 * 0.04 * lqd_corporate_etf/df1['lqd_corporate_bond_etf'].iloc[0] +
                                   0.05 * pricing_model.price_tips(real_10y_yield,cpi) +
                                   0.04 * pricing_model.price_revenue_bond(thirty_yr_trsry_yield) +
                                   0.03 * pricing_model.price_green_bond(five_yr_trsry_yield)
                                   )
    
    cash_value = 10000000 * 0.05

    return equity_value + fixed_income_value + derivatives_value + cash_value

# Revalue under each simulation
def revalue(sim_levels_df):
    values = []
    for i in range(len(sim_levels_df)):
        scenario = sim_levels_df.iloc[i].to_dict()
        values.append(price_portfolio(scenario))
    return np.array(values)

# ------------------------------------------------------
#      MONTE CARLO SIM VAR COMPUTATION - FULL WINDOW
# ------------------------------------------------------ 
# Load Data
try:
    df_response = load_data(DATA_URL)
    if df_response is not None:
        df = df_response.copy()
        # Set date as index
        df = df.set_index('date')
        df1 = df.copy()
        factors = df.columns.tolist()
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
    else:
        raise ValueError("Failed to load data from GitHub")
    
    if len(df) == 0:
        raise ValueError("No data available in dataframe. Please check data file.")
    
    df_portfolio_returns_response = load_data(PORTFOLIO_RETURNS_URL)
    if df_portfolio_returns_response is not None:
        df_portfolio_returns = df_portfolio_returns_response.copy()
        df_portfolio_returns_250d = df_portfolio_returns.iloc[-250:]
    else:
        raise ValueError("Failed to load portfolio returns data from GitHub")
except Exception as e:
    print(f"Error loading or processing data: {e}")
    raise

log_return_factors = [
    'eq_apple_price', 'eq_lockheed_martin_price', 'eq_nvidia_price',
    'eq_procter_gamble_price', 'eq_johnson_johnson_price', 'eq_toyota_price',
    'eq_nestle_price', 'eq_x_steel_price', 'sp500_index',
    'costco_stock_price', 'exxonmobil_stock_price',
    'lqd_corporate_bond_etf', 'gold_spot_price', 'crude_oil_wti_spot',
    'soybean_spot_usd', 'DAX_spot', 'Nikkei_spot',
    'fx_eurusd_rate', 'fx_usdjpy_rate', 'fx_gbpusd_rate', 'USD_INR',
    'vix_index_level'
]

diff_factors = [
    '10y_treasury_yield', 'fed_funds_rate', 'factor_GBP_sonia',
    'factor_corporate_yield', 'high_yield_credit spread',
    '1_year_euro_yield_curve', 'Real_10Y_yield', '5y_treasury_yield',
    'AA_Corp_bond_yield', '30Y_treasury_yield', 'Basic_Loan_Rate_JPY',
    'MIBOR', 'Euro_STR', 'CPI',
    'EUR-USD_IVOL', 'USD-JPY_IVOL', 'COST_IVOL_3MMA', 'XOM_IVOL_3MMA',
    'COST_DIV_YIELD', 'XOM_DIV_YIELD',
    'DAX_Call_ivol_30D', 'DAX_Put_ivol_30D', '5_Y_ford_credit_spread',
    'NKY_30D_ivol', 'NKY_Div_yield', 'SPX_Div_yield'
]

df[diff_factors] = df[diff_factors] / 100.0

returns_log = np.log(df[log_return_factors] / df[log_return_factors].shift(1))
returns_diff = df[diff_factors].diff()

returns = pd.concat([returns_log, returns_diff], axis=1).dropna()

returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

assert np.isfinite(returns_clean.values).all(), "Still contains non-finite values"

u_data, marginals = fit_marginals(returns_clean)

missing_log_factors = [f for f in log_return_factors if f not in returns_clean.columns]
missing_diff_factors = [f for f in diff_factors if f not in returns_clean.columns]

if missing_log_factors or missing_diff_factors:
    print("Warning: Some factors are missing in the returns data:")
    if missing_log_factors:
        print(f"  Missing log return factors: {missing_log_factors}")
    if missing_diff_factors:
        print(f"  Missing diff factors: {missing_diff_factors}")

u_data = u_data.astype(float)
u_data = u_data.clip(1e-6, 1 - 1e-6)
assert np.all((u_data >= 0) & (u_data <= 1)), "u_data must contain values in [0,1]"

norm_data = norm.ppf(u_data)
cov_matrix = np.cov(norm_data.T)

std_devs = np.sqrt(np.diag(cov_matrix))
corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

np.fill_diagonal(corr_matrix, 1.0)

ll_g, aic_g = copula_log_likelihood("gaussian", u_data, corr_matrix)
ll_t, aic_t = copula_log_likelihood("t", u_data, corr_matrix, df_copula=4)

print(f"Gaussian Copula: Log-likelihood = {ll_g:.2f}, AIC = {aic_g:.2f}")
print(f"Student's t Copula: Log-likelihood = {ll_t:.2f}, AIC = {aic_t:.2f}")

chosen_copula = "t" if aic_t < aic_g else "gaussian"
print(f"\n✅ Selected Copula: {chosen_copula.upper()}")

current_levels = df.iloc[-1].to_dict()

confidence_level = 0.99
n_simulations = 10000

np.random.seed(42)

if chosen_copula == 't':
    try:
        u_sim = simulate_student_copula(u_data, n_simulations, 12)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error in Student's t copula simulation: {e}")
        print("Falling back to Gaussian copula...")
        u_sim = simulate_gaussian_copula(u_data, n_simulations)
elif chosen_copula == 'gaussian':
    try:
        u_sim = simulate_gaussian_copula(u_data, n_simulations)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error in Gaussian copula simulation: {e}")
        print("Reducing ridge parameter and retrying...")
        u_sim = simulate_gaussian_copula(u_data, n_simulations, ridge=1e-4)

sim_returns = u_to_returns(u_sim, marginals)

for col in diff_factors:
    if col in sim_returns.columns:
        sim_returns[col] = sim_returns[col].clip(lower=-0.01, upper=0.01)

sim_levels = returns_to_levels(sim_returns, current_levels, 
                             log_return_factors=log_return_factors,
                             diff_factors=diff_factors,
                             clip_bounds=(1e-4, 1e4),
                             verbose=False)

values = revalue(sim_levels)
V0 = price_portfolio(current_levels)
pnl = (values - V0) / V0

pnl = pnl[np.isfinite(pnl)]
VaR = -np.percentile(pnl, (1 - confidence_level) * 100)

plt.hist(pnl, bins=50, color="steelblue", alpha=0.7)
plt.axvline(-VaR, color="red", linestyle="--", label=f"VaR @ {int(confidence_level*100)}% = {VaR:.4%}")
plt.title(f"Portfolio PnL Distribution ({chosen_copula.capitalize()} Copula, {n_simulations} Sims)")
plt.xlabel("PnL")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

print(f"\n✅ Final VaR @ {int(confidence_level*100)}% = {VaR:.4%} using {n_simulations} simulations.")
print(f"\n✅ Final VaR @ {int(confidence_level*100)}% = ${VaR * 10000000} using {n_simulations} simulations.")

bootstrap_iterations = 1000
ci_percentile = 95

np.random.seed(42)

bootstrap_vars = []

for _ in range(bootstrap_iterations):
    resample = np.random.choice(pnl, size=len(pnl), replace=True)
    var_boot = -np.percentile(resample, (1 - confidence_level) * 100)
    bootstrap_vars.append(var_boot)

bootstrap_vars = np.array(bootstrap_vars)

lower_bound = np.percentile(bootstrap_vars, (100 - ci_percentile) / 2)
upper_bound = np.percentile(bootstrap_vars, 100 - (100 - ci_percentile) / 2)
mean_var = np.mean(bootstrap_vars)

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_vars, bins=50, color="lightsteelblue", edgecolor="black", alpha=0.8)
plt.axvline(lower_bound, color="red", linestyle="--", label=f"Lower {ci_percentile}% CI = {lower_bound:.5f}")
plt.axvline(upper_bound, color="red", linestyle="--", label=f"Upper {ci_percentile}% CI = {upper_bound:.5f}")
plt.axvline(mean_var, color="blue", linestyle="-", label=f"Mean Bootstrap VaR = {mean_var:.5f}")
plt.title(f"Bootstrap Distribution of 99% VaR ({bootstrap_iterations} resamples)")
plt.xlabel("VaR Estimate")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()

print(f"\n✅ Bootstrap {ci_percentile}% CI for 99% VaR:")
print(f"    Lower Bound: {lower_bound:.5f}")
print(f"    Upper Bound: {upper_bound:.5f}")
print(f"    Mean VaR:    {mean_var:.5f}")

notional = 10000000
n_days = len(df_portfolio_returns)
expected_exceptions = int((1 - confidence_level) * n_days)
actual_exceptions = (df_portfolio_returns['Return'] * notional < -VaR * notional).sum()

x = np.arange(0, 2 * expected_exceptions + 20)
pmf = binom.pmf(x, n_days, 1 - confidence_level)

plt.figure(figsize=(10, 5))
plt.plot(x, pmf, label='Expected Binomial Distribution')
plt.axvline(expected_exceptions, color='green', linestyle='--', label=f'Expected: {expected_exceptions}')
plt.axvline(actual_exceptions, color='red', linestyle='-', label=f'Actual: {actual_exceptions}')
plt.title(f'Binomial Test of 99% 1-Day VaR Violations\n(Total Days: {n_days})')
plt.xlabel('Number of Exceptions')
plt.ylabel('Probability Mass')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

N = 5014
x = 22
p = 0.01
phat = x / N

LR_uc = -2 * (np.log(((1 - p)**(N - x)) * (p**x)) - np.log(((1 - phat)**(N - x)) * (phat**x)))
p_value = 1 - chi2.cdf(LR_uc, df=1)

print(f"Kupiec LR: {LR_uc:.2f}, p-value: {p_value:.4f}")

df_250d = df.iloc[-250:].copy()

df_250d = df_250d.apply(pd.to_numeric, errors='coerce')
df_250d = df_250d.dropna()

df_250d[diff_factors] = df_250d[diff_factors] / 100.0

returns_log_250d = np.log(df_250d[log_return_factors] / df_250d[log_return_factors].shift(1))
returns_diff_250d = df_250d[diff_factors].diff()

returns_250d = pd.concat([returns_log_250d, returns_diff_250d], axis=1).dropna()

returns_clean_250d = returns_250d.replace([np.inf, -np.inf], np.nan).dropna()

assert np.isfinite(returns_clean_250d.values).all(), "Still contains non-finite values"

u_data_250d, marginals_250d = fit_marginals(returns_clean_250d)

missing_log_factors_250d = [f for f in log_return_factors if f not in returns_clean_250d.columns]
missing_diff_factors_250d = [f for f in diff_factors if f not in returns_clean_250d.columns]

if missing_log_factors_250d or missing_diff_factors_250d:
    print("Warning: Some factors are missing in the 250-day returns data:")
    if missing_log_factors_250d:
        print(f"  Missing log return factors: {missing_log_factors_250d}")
    if missing_diff_factors_250d:
        print(f"  Missing diff factors: {missing_diff_factors_250d}")

u_data_250d = u_data_250d.astype(float)
u_data_250d = u_data_250d.clip(1e-6, 1 - 1e-6)
assert np.all((u_data_250d >= 0) & (u_data_250d <= 1)), "u_data_250d must contain values in [0,1]"

norm_data_250d = norm.ppf(u_data_250d)
cov_matrix_250d = np.cov(norm_data_250d.T)

std_devs_250d = np.sqrt(np.diag(cov_matrix_250d))
corr_matrix_250d = cov_matrix_250d / np.outer(std_devs_250d, std_devs_250d)
corr_matrix_250d = regularize_corr_matrix(corr_matrix_250d)

np.fill_diagonal(corr_matrix_250d, 1.0)

ll_g_250d, aic_g_250d = copula_log_likelihood("gaussian", u_data_250d, corr_matrix_250d)
ll_t_250d, aic_t_250d = copula_log_likelihood("t", u_data_250d, corr_matrix_250d, df_copula=4)

print(f"\nGaussian Copula: Log-likelihood = {ll_g_250d:.2f}, AIC = {aic_g_250d:.2f}")
print(f"Student's t Copula: Log-likelihood = {ll_t_250d:.2f}, AIC = {aic_t_250d:.2f}")

chosen_copula_250d = "t" if aic_t_250d < aic_g_250d else "gaussian"
print(f"\n✅ Selected Copula: {chosen_copula_250d.upper()}")

current_levels_250d = df_250d.iloc[-1].to_dict()

n_simulations_250d = 3000

np.random.seed(42)

if chosen_copula_250d == 't':
    try:
        u_sim_250d = simulate_student_copula(u_data_250d, n_simulations_250d, 12)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error in Student's t copula simulation (250d window): {e}")
        print("Falling back to Gaussian copula...")
        u_sim_250d = simulate_gaussian_copula(u_data_250d, n_simulations_250d)
elif chosen_copula_250d == 'gaussian':
    try:
        u_sim_250d = simulate_gaussian_copula(u_data_250d, n_simulations_250d)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error in Gaussian copula simulation (250d window): {e}")
        print("Reducing ridge parameter and retrying...")
        u_sim_250d = simulate_gaussian_copula(u_data_250d, n_simulations_250d, ridge=1e-4)

sim_returns_250d = u_to_returns(u_sim_250d, marginals_250d)

for col in diff_factors:
    if col in sim_returns_250d.columns:
        sim_returns_250d[col] = sim_returns_250d[col].clip(lower=-0.01, upper=0.01)

sim_levels_250d = returns_to_levels(sim_returns_250d, current_levels_250d, 
                                  log_return_factors=log_return_factors,
                                  diff_factors=diff_factors,
                                  clip_bounds=(1e-4, 1e4),
                                  verbose=False)

values_250d = revalue(sim_levels_250d)
V0_250d = price_portfolio(current_levels_250d)
pnl_250d = (values_250d - V0_250d) / V0_250d

pnl_250d = pnl_250d[np.isfinite(pnl_250d)]
VaR_250d = -np.percentile(pnl_250d, (1 - confidence_level) * 100)

print(f"\n✅ Final 1 Day VaR @ {int(confidence_level*100)}% = {VaR_250d:.4%} using {n_simulations_250d} simulations.")

def main():
    global df, df1
    
    np.random.seed(42)
    
    try:
        df_response = load_data(DATA_URL)
        if df_response is not None:
            df = df_response.copy()
            # Set date as index
            df = df.set_index('date')
            df1 = df.copy()
            factors = df.columns.tolist()
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna()
        else:
            raise ValueError("Failed to load data from GitHub")
        
        if len(df) == 0:
            raise ValueError("No data available in dataframe. Please check data file.")
        
        df_portfolio_returns_response = load_data(PORTFOLIO_RETURNS_URL)
        if df_portfolio_returns_response is not None:
            df_portfolio_returns = df_portfolio_returns_response.copy()
            df_portfolio_returns_250d = df_portfolio_returns.iloc[-250:]
        else:
            raise ValueError("Failed to load portfolio returns data from GitHub")
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        raise

    print("\n✅ Analysis completed successfully.")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
