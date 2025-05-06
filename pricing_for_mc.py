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


