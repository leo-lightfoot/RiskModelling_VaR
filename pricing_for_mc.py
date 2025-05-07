import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

class SimplifiedPricing:
    """
    Provides simplified pricing functions for Monte Carlo VaR simulation.
    """
    
    def __init__(self):
        # Constants used by multiple pricing functions
        self.DAYS_IN_YEAR = 365
    
    #----------------------------------
    # Fixed Income Pricing Functions
    #----------------------------------
    
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
        """
        Price a 10-year Treasury bond
        
        """
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
        """
        Price a Treasury Inflation-Protected Security (TIPS)
        
        """
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
        """
        Price a corporate bond based on Treasury yield plus credit spread
        
        """
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
        """
        Price a green bond, accounting for greenium (green premium)
        
        """
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
        """
        Price a revenue bond with spread over treasury
        
        """
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
        """
        Price a zero-coupon bond
        
        """
        NOTIONAL = 100
        
        if days_to_maturity is None:
            years_to_maturity = maturity_years
        else:
            years_to_maturity = days_to_maturity / 365.0
        
        # For zero-coupon bond, P = F / (1 + y)^t for discrete compounding
        # or P = F * exp(-y * t) for continuous compounding
        # Using continuous compounding for simplicity
        return NOTIONAL * np.exp(-yield_rate * years_to_maturity)
    
    def price_high_yield_corp_debt(self, treasury_yield, credit_spread, days_to_maturity=None, 
                                  coupon_rate=0.065, maturity_years=5, frequency=2):
        """
        Price high yield corporate debt instrument
        
        """
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
    
    #----------------------------------
    # Derivatives Pricing Functions
    #----------------------------------
    
    def price_equity_futures(self, spot_price, risk_free_rate, dividend_yield, days_to_expiry):
        """
        Price an equity futures contract
        
        """
        time_to_maturity = days_to_expiry / self.DAYS_IN_YEAR
        
        # F = S * e^((r-q)*T)
        futures_price = spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)
        
        return futures_price
    
    def price_vix_futures(self, vix_index_level, vix_history=None, days_to_expiry=21, mean_reversion_speed=0.2, vol_of_vol=None):
        """
        Price VIX futures using a mean-reverting model
        
        Parameters:
        -----------
        vix_index_level : float
            Current VIX index level
        vix_history : pandas.Series
            Historical VIX values to calculate 30-day moving average (expected column name: 'vix_index_level')
        days_to_expiry : int
            Days until the futures contract expires (default 21 trading days ≈ 1 month)
        mean_reversion_speed : float
            Speed of mean reversion (theta)
        vol_of_vol : float
            Volatility parameter for the model. If None, calculated from vix_history
            
        Returns:
        --------
        float
            Estimated VIX futures price (column name in output dataframe: 'vix_futures_1m')
        """
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
        """
        Price a commodity futures contract
        
        """
        time_to_maturity = days_to_expiry / self.DAYS_IN_YEAR
        
        # F = S * e^((r + u - y)*T) where u = storage cost, y = convenience yield
        futures_price = spot_price * np.exp((risk_free_rate + storage_cost - convenience_yield) * time_to_maturity)
        
        return futures_price
    
    def price_gold_futures(self, spot_price, risk_free_rate, storage_cost=0.005, days_to_expiry=90):
        """
        Price gold futures contract
        
        """
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
        """
        Price crude oil futures contract
        
        """
        return self.price_commodity_futures(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            storage_cost=storage_cost,
            convenience_yield=convenience_yield,
            days_to_expiry=days_to_expiry
        )
    
    def price_soybean_futures(self, spot_price, risk_free_rate, days_to_expiry, 
                            month=None, base_storage_cost=0.04, seasonal_amplitude=0.02):
        """
        Price soybean futures with seasonal storage costs
        
        """
        # If month not provided, get current month
        if month is None:
            month = datetime.now().month
        
        # Calculate seasonal storage cost with peak in October (month 10)
        phase_shift = 10
        seasonal_storage_cost = base_storage_cost + seasonal_amplitude * np.cos(2 * np.pi * (month - phase_shift) / 12)
        
        # Constant convenience yield (could also be made seasonal if desired)
        convenience_yield = 0.02
        
        return self.price_commodity_futures(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            storage_cost=seasonal_storage_cost,
            convenience_yield=convenience_yield,
            days_to_expiry=days_to_expiry
        )
    
    def black_scholes_call(self, spot, strike, days_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Price a European call option using Black-Scholes-Merton model
        
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        call_price = spot * np.exp(-dividend_yield * T) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(self, spot, strike, days_to_expiry, risk_free_rate, volatility, dividend_yield=0):
        """
        Price a European put option using Black-Scholes-Merton model
        
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        put_price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * T) * norm.cdf(-d1)
        
        return put_price
    
    def garman_kohlhagen_call(self, spot, strike, days_to_expiry, domestic_rate, foreign_rate, volatility):
        """
        Price a European FX call option using Garman-Kohlhagen model
        
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        call_price = spot * np.exp(-foreign_rate * T) * norm.cdf(d1) - strike * np.exp(-domestic_rate * T) * norm.cdf(d2)
        
        return call_price
    
    def garman_kohlhagen_put(self, spot, strike, days_to_expiry, domestic_rate, foreign_rate, volatility):
        """
        Price a European FX put option using Garman-Kohlhagen model
        
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        if T <= 0 or volatility <= 0 or spot <= 0:
            return 0.0
        
        d1 = (np.log(spot / strike) + (domestic_rate - foreign_rate + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)
        
        put_price = strike * np.exp(-domestic_rate * T) * norm.cdf(-d2) - spot * np.exp(-foreign_rate * T) * norm.cdf(-d1)
        
        return put_price
    
    #----------------------------------
    # Forex Forward Pricing
    #----------------------------------
    
    def price_fx_forward(self, spot_rate, domestic_rate, foreign_rate, days_to_expiry):
        """
        Price an FX forward contract
        
        """
        T = days_to_expiry / self.DAYS_IN_YEAR
        
        # F = S * (1 + r_d)^T / (1 + r_f)^T for discrete compounding
        # or F = S * e^((r_d - r_f) * T) for continuous compounding
        # Using the continuous version for simplicity
        forward_rate = spot_rate * np.exp((domestic_rate - foreign_rate) * T)
        
        return forward_rate

    #----------------------------------
    # Credit Default Swap Pricing
    #----------------------------------
    
    def price_cds(self, credit_spread, risk_free_rate, current_date=None, previous_date=None, 
                maturity_years=5, recovery_rate=0.4, payments_per_year=4, roll_frequency_days=365, 
                notional=100, transaction_cost=0.002, previous_value=None):
        """
        Price a Credit Default Swap (CDS) with rolling strategy logic
        
        """
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
    
    #----------------------------------
    # Variance Swap Pricing
    #----------------------------------
    
    def price_variance_swap(self, call_ivol, put_ivol, current_date=None, previous_date=None, maturity_days=30, 
                          notional=100, annual_basis=365.0, transaction_cost=0.002):
        """
        Price a variance swap with rollover logic
        
        """
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
                # If we have a previous date, check if this is a roll date
                # based on the original contract expiry
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
    
    #----------------------------------
    # Asian Option Pricing
    #----------------------------------
    
    def price_asian_option(self, spot, strike=None, risk_free_rate=0, dividend_yield=0, volatility=0, 
                          current_date=None, previous_date=None, option_type='put',
                          days_in_option=63, paths=2000, notional=100, seed=42):
        """
        Price an Asian option with rollover logic using Monte Carlo simulation
        
        """
        from datetime import timedelta
        import numpy as np
        
        result = {}
        
        # Check for valid inputs
        if np.isnan(spot) or np.isnan(volatility) or np.isnan(risk_free_rate) or np.isnan(dividend_yield):
            result['option_price'] = np.nan
            result['days_to_maturity'] = 0
            result['roll_date'] = False
            result['daily_return'] = np.nan
            result['log_return'] = np.nan
            result['nav'] = notional
            return result
            
        # If strike not provided, use ATM
        if strike is None:
            strike = spot
            
        result['strike'] = strike
        
        # Roll date logic
        result['roll_date'] = False
        days_to_maturity = days_in_option
        
        if current_date is not None:
            if previous_date is not None:
                # Determine if we need to roll based on original contract expiry
                # In the original code, a roll happens at the start of a new period
                # which starts the day after the previous option expires
                
                # Calculate when the previous option would end (if any)
                if hasattr(previous_date, 'current_end'):
                    previous_end = previous_date.current_end
                else:
                    previous_end = previous_date + timedelta(days=days_in_option - 1)
                    
                # If current date is after or on the expiry, it's a roll date
                if current_date > previous_end:
                    result['roll_date'] = True
                    # Set new expiry
                    current_end = current_date + timedelta(days=days_in_option - 1)
                else:
                    # Continue with existing contract
                    current_end = previous_end
                
                # Calculate days remaining
                days_to_maturity = (current_end - current_date).days + 1  # +1 because we include current day
            else:
                # First date, start new contract
                result['roll_date'] = True
                days_to_maturity = days_in_option
                
        result['days_to_maturity'] = days_to_maturity
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate time parameters
        T = days_to_maturity / 252.0  # Assuming 252 trading days per year
        
        # Use remaining days as steps, capped at original days_in_option
        steps = min(days_to_maturity, days_in_option)
        
        if steps <= 0 or T <= 0:
            # Option at or past expiry
            if option_type.lower() == 'put':
                result['option_price'] = max(strike - spot, 0)
            else:
                result['option_price'] = max(spot - strike, 0)
        else:
            # Price using Monte Carlo
            dt = T / steps
            drift = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
            diffusion = volatility * np.sqrt(dt)
            
            # Simulate price paths
            Z = np.random.randn(paths, steps)
            price_paths = spot * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
            
            # Calculate arithmetic average price for each path
            average_price = np.mean(price_paths, axis=1)
            
            # Calculate payoff
            if option_type.lower() == 'put':
                payoff = np.maximum(strike - average_price, 0)
            else:
                payoff = np.maximum(average_price - strike, 0)
            
            # Calculate option price as discounted expected payoff
            option_price = np.exp(-risk_free_rate * T) * np.mean(payoff)
            result['option_price'] = option_price
        
        # Calculate returns
        if previous_date is not None and 'previous_price' in previous_date and not np.isnan(previous_date.previous_price):
            previous_price = previous_date.previous_price
            
            if previous_price > 0:
                daily_return = (result['option_price'] - previous_price) / previous_price
                result['daily_return'] = daily_return
                result['log_return'] = np.log(result['option_price'] / previous_price)
                
                # Update NAV
                result['nav'] = notional * (1 + daily_return)
            else:
                result['daily_return'] = 0
                result['log_return'] = 0
                result['nav'] = notional
        else:
            # For the first date, no return calculation
            result['daily_return'] = 0
            result['log_return'] = 0
            result['nav'] = notional
            
        # Store current date and price for next valuation
        current_date.current_end = current_date + timedelta(days=days_in_option - 1)
        current_date.previous_price = result['option_price']
            
        return result
    
    #----------------------------------
    # Barrier Option Pricing
    #----------------------------------
    
    def price_barrier_option(self, spot, strike=None, risk_free_rate=0, dividend_yield=0, volatility=0, 
                           current_date=None, previous_date=None, barrier_multiplier=1.1,
                           maturity_days=30, option_type='call', barrier_type='knockout',
                           notional=100, transaction_cost=0.003, annual_basis=365.0):
        """
        Price a barrier option with rollover logic
        
        """
        from datetime import timedelta
        
        result = {}
        
        # Check for valid inputs
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
            
        # If strike not provided, use ATM
        if strike is None:
            strike = spot
            
        result['strike_price'] = strike
        
        # Calculate barrier level
        barrier_level = barrier_multiplier * strike
        result['barrier_level'] = barrier_level
        
        # Check for knock-out or knock-in conditions
        result['knocked_out'] = False
        if barrier_type.lower() == 'knockout':
            if (option_type.lower() == 'call' and spot >= barrier_level) or \
               (option_type.lower() == 'put' and spot <= barrier_level):
                result['knocked_out'] = True
        
        # Roll date logic
        result['roll_date'] = False
        days_to_expiry = maturity_days
        current_expiry = None
        
        if current_date is not None:
            if previous_date is not None:
                # In the original code, roll happens when current_date >= current_expiry
                if hasattr(previous_date, 'current_expiry'):
                    current_expiry = previous_date.current_expiry
                else:
                    current_expiry = previous_date + timedelta(days=maturity_days)
                
                if current_date >= current_expiry:
                    result['roll_date'] = True
                    # Reset for new contract
                    current_expiry = current_date + timedelta(days=maturity_days)
                    # With a new contract, reset strike and barrier based on current spot
                    if strike is None:  # Only if ATM was used
                        strike = spot
                        result['strike_price'] = strike
                        barrier_level = barrier_multiplier * strike
                        result['barrier_level'] = barrier_level
            else:
                # First date, set expiry
                current_expiry = current_date + timedelta(days=maturity_days)
                result['roll_date'] = True  # First date is effectively a roll date
                
            # Calculate days to expiry
            if current_expiry is not None:
                days_to_expiry = (current_expiry - current_date).days
                
        result['days_to_expiry'] = days_to_expiry
        
        # Calculate time to maturity in years
        T = days_to_expiry / annual_basis
        
        # Price the option based on whether it's knocked out/in
        if result['knocked_out']:
            # For knock-out option
            result['option_price'] = 0.0
        elif T <= 0 or volatility <= 0 or spot <= 0:
            # Option expired or invalid parameters
            result['option_price'] = 0.0
        else:
            # Price using Black-Scholes with barrier adjustment
            # First calculate standard Black-Scholes price
            d1 = (np.log(spot / strike) + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
            d2 = d1 - volatility * np.sqrt(T)
            
            if option_type.lower() == 'call':
                bs_price = spot * np.exp(-dividend_yield * T) * norm.cdf(d1) - strike * np.exp(-risk_free_rate * T) * norm.cdf(d2)
                
                # Apply barrier adjustment
                if barrier_type.lower() == 'knockout':
                    # Simple adjustment for up-and-out call
                    barrier_discount = (barrier_level - spot) / barrier_level
                    result['option_price'] = bs_price * barrier_discount
                else:  # 'knockin'
                    # For knock-in, price = vanilla - knockout
                    barrier_discount = 1 - (barrier_level - spot) / barrier_level
                    result['option_price'] = bs_price * barrier_discount
            else:  # 'put'
                bs_price = strike * np.exp(-risk_free_rate * T) * norm.cdf(-d2) - spot * np.exp(-dividend_yield * T) * norm.cdf(-d1)
                
                # Apply barrier adjustment
                if barrier_type.lower() == 'knockout':
                    # Simple adjustment for down-and-out put
                    barrier_discount = (spot - barrier_level) / spot
                    result['option_price'] = bs_price * barrier_discount
                else:  # 'knockin'
                    # For knock-in, price = vanilla - knockout
                    barrier_discount = 1 - (spot - barrier_level) / spot
                    result['option_price'] = bs_price * barrier_discount
        
        # Calculate returns
        if previous_date is not None and hasattr(previous_date, 'previous_price'):
            previous_price = previous_date.previous_price
            
            if previous_price > 0:
                # Calculate return
                daily_return = (result['option_price'] - previous_price) / previous_price
                
                # Apply transaction cost on roll dates
                if result['roll_date']:
                    daily_return -= transaction_cost
                    
                result['daily_return'] = daily_return
                result['log_return'] = np.log((result['option_price'] + 1e-10) / (previous_price + 1e-10))  # Avoid log(0)
                
                # Calculate NAV
                result['nav'] = notional * (1 + daily_return)
            else:
                result['daily_return'] = 0
                result['log_return'] = 0
                result['nav'] = notional
        else:
            # For the first date, no return calculation
            result['daily_return'] = 0
            result['log_return'] = 0
            result['nav'] = notional
            
        # Store info for next valuation
        if current_date is not None:
            current_date.current_expiry = current_expiry
            current_date.previous_price = result['option_price']
            
        return result


