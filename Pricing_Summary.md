# Portfolio Instruments Summary

This document provides a summary of all instruments in the portfolio, including their weights, assumptions, risk factors, positions, and pricing methodologies.

## Portfolio Allocation

| Instrument | Weight | Position |
|------------|--------|----------|
| Apple | 2.0% | Long |
| Lockheed Martin | 2.0% | Long |
| Nvidia | 2.0% | Long |
| Procter & Gamble | 2.0% | Long |
| Johnson & Johnson | 2.0% | Long |
| Toyota | 2.0% | Long |
| Nestle | 2.0% | Long |
| X Steel | 2.0% | Long |
| 10Y Treasury Bond | 5.0% | Long |
| LQD Corporate Bond ETF | 4.0% | Long |
| 10Y TIPS | 5.0% | Long |
| 1Y EUR Zero Coupon Bond | 5.0% | Long |
| High Yield Corporate Debt | 4.0% | Long |
| 5Y Green Bond | 3.0% | Long |
| 30Y Revenue Bond | 4.0% | Long |
| S&P 500 Futures (1M) | 4.0% | Long |
| VIX Futures | 3.0% | Long |
| Crude Oil Futures | 3.0% | Long |
| Gold Futures (3M) | 4.0% | Long |
| Soybean Futures (6M) | 3.0% | Long |
| Costco ITM Call Option (3M) | 3.0% | Long |
| Exxon Mobil ITM Put Option (3M) | 3.0% | Long |
| EUR/USD ATM Call Option (1M) | 3.0% | Long |
| USD/JPY ATM Put Option (3M) | 3.0% | Long |
| GBP/USD Forward (6M) | 4.0% | Long |
| USD/INR Forward (3M) | 4.0% | Short |
| Ford CDS (5Y) | 3.0% | Long |
| DAX Variance Swap (30D) | 3.0% | Long |
| Nikkei Asian Put Option (3M) | 3.0% | Long |
| S&P 500 Knock-Out Call Option (1M) | 3.0% | Long |
| Cash | 5.0% | Long |

## Instrument Details

### Equities
- **Pricing Method**: NAV and log returns calculation based on price data
- **Risk Factors**: Equity prices
- **Assumptions**: Directly using market prices

### 10Y Treasury Bond
- **Pricing Method**: Bond pricing with roll-over logic
- **Risk Factors**: 10-year Treasury yield
- **Assumptions**: 
  - 2% annual coupon rate
  - Semiannual payments
  - 10-year maturity with roll-over every 10 years

### LQD Corporate Bond ETF
- **Pricing Method**: NAV and log returns calculation based on ETF price
- **Risk Factors**: LQD ETF price
- **Assumptions**: Direct market prices of the ETF

### 10Y TIPS
- **Pricing Method**: TIPS pricing with inflation adjustment
- **Risk Factors**: Real 10Y yield, CPI (inflation)
- **Assumptions**: 
  - 1.25% annual coupon rate
  - Monthly inflation from CPI data
  - Semiannual payments
  - Roll-over every 10 years

### 1Y EUR Zero Coupon Bond
- **Pricing Method**: Zero-coupon bond pricing in EUR with USD conversion
- **Risk Factors**: 1-year Euro yield curve, EUR/USD exchange rate
- **Assumptions**:
  - Zero coupon (single payment at maturity)
  - 1-year maturity with annual roll-over
  - USD/EUR exchange rate conversion for final NAV

### High Yield Corporate Debt
- **Pricing Method**: Bond pricing with credit spread
- **Risk Factors**: 10Y Treasury yield, high yield credit spread
- **Assumptions**:
  - 6.5% coupon rate
  - 5-year maturity with monthly roll-over
  - Semiannual coupon payments
  - Credit spread added to base Treasury yield

### 5Y Green Bond
- **Pricing Method**: Bond pricing with "greenium" (green bond premium)
- **Risk Factors**: 5Y Treasury yield
- **Assumptions**:
  - 2.5% coupon rate
  - 5-year maturity with quarterly roll-over
  - 10 bps "greenium" (lower yield than conventional bonds)
  - Semiannual coupon payments

### 30Y Revenue Bond
- **Pricing Method**: Bond pricing with credit spread
- **Risk Factors**: 30Y Treasury yield
- **Assumptions**:
  - 3.5% coupon rate
  - 30-year maturity with semiannual roll-over
  - 50 bps credit spread over Treasury
  - Semiannual coupon payments

### S&P 500 Futures (1M)
- **Pricing Method**: Futures pricing with roll-over
- **Risk Factors**: S&P 500 index, Fed funds rate, dividend yield
- **Assumptions**:
  - Monthly expiry and roll-over
  - Cost of carry model based on risk-free rate and dividend yield

### VIX Futures
- **Pricing Method**: Mean-reverting simulation with roll-over
- **Risk Factors**: VIX index level
- **Assumptions**:
  - 21-day contract duration (monthly)
  - Mean reversion to rolling average
  - Stochastic process with volatility proportional to VIX level

### Crude Oil Futures
- **Pricing Method**: Futures pricing with roll-over and storage costs
- **Risk Factors**: Crude oil WTI spot price, Fed funds rate
- **Assumptions**:
  - Monthly expiry (20th of each month)
  - 0.5% monthly storage cost
  - Contango model based on interest rates and storage costs

### Gold Futures (3M)
- **Pricing Method**: Futures pricing with roll-over and storage costs
- **Risk Factors**: Gold spot price, Fed funds rate
- **Assumptions**:
  - 3-month contract
  - 0.05% monthly storage cost
  - Contango model based on interest rates and storage costs

### Soybean Futures (6M)
- **Pricing Method**: Futures pricing with seasonal storage costs
- **Risk Factors**: Soybean spot price, Fed funds rate
- **Assumptions**:
  - 6-month contract
  - Seasonal storage costs varying by month:
    - Harvest season (Oct-Nov): 0.15% monthly
    - Pre-harvest (Jul-Sep): 0.40% monthly
    - Other months: 0.25% monthly

### Costco ITM Call Option (3M)
- **Pricing Method**: Black-Scholes option pricing
- **Risk Factors**: Costco stock price, implied volatility, interest rate, dividend yield
- **Assumptions**:
  - 3-month maturity
  - Strike price 5% in-the-money
  - 10 bps transaction cost on roll-over
  - American-style exercise

### Exxon Mobil ITM Put Option (3M)
- **Pricing Method**: Black-Scholes option pricing
- **Risk Factors**: Exxon Mobil stock price, implied volatility, interest rate, dividend yield
- **Assumptions**:
  - 3-month maturity
  - Strike price 5% in-the-money
  - 10 bps transaction cost on roll-over
  - American-style exercise

### EUR/USD ATM Call Option (1M)
- **Pricing Method**: Garman-Kohlhagen model (FX option pricing)
- **Risk Factors**: EUR/USD exchange rate, EUR-USD implied volatility, Fed funds rate, Euro STR rate
- **Assumptions**:
  - 1-month maturity
  - At-the-money strike
  - 5 bps transaction cost on roll-over
  - European-style exercise

### USD/JPY ATM Put Option (3M)
- **Pricing Method**: Garman-Kohlhagen model (FX option pricing)
- **Risk Factors**: USD/JPY exchange rate, USD-JPY implied volatility, Fed funds rate, JPY interest rate
- **Assumptions**:
  - 3-month maturity
  - At-the-money strike
  - 5 bps transaction cost on roll-over
  - European-style exercise

### GBP/USD Forward (6M)
- **Pricing Method**: FX forward pricing based on interest rate parity
- **Risk Factors**: GBP/USD exchange rate, Fed funds rate, GBP SONIA rate
- **Assumptions**:
  - 6-month contract
  - Long GBP position (buy GBP, sell USD)
  - Interest rate parity for forward rate calculation

### USD/INR Forward (3M)
- **Pricing Method**: FX forward pricing based on interest rate parity
- **Risk Factors**: USD/INR exchange rate, Fed funds rate, MIBOR (India) rate
- **Assumptions**:
  - 3-month contract
  - Short USD position (sell USD, buy INR)
  - Interest rate parity for forward rate calculation

### Ford CDS (5Y)
- **Pricing Method**: CDS valuation with hazard rate model
- **Risk Factors**: Ford 5Y credit spread, 5Y Treasury yield
- **Assumptions**:
  - 5-year maturity
  - 2% annual coupon (200 bps)
  - 40% recovery rate
  - 5 bps transaction cost

### DAX Variance Swap (30D)
- **Pricing Method**: Variance swap pricing based on implied volatility
- **Risk Factors**: DAX call/put implied volatility
- **Assumptions**:
  - 30-day contract duration
  - Variance strike set at 98% of fair value
  - Pay fixed, receive realized variance

### Nikkei Asian Put Option (3M)
- **Pricing Method**: Monte Carlo simulation for Asian option pricing
- **Risk Factors**: Nikkei spot index, implied volatility, JPY interest rate, dividend yield
- **Assumptions**:
  - 3-month maturity
  - At-the-money strike
  - Averaging price every 5 trading days
  - 10 bps transaction cost on roll-over
  - 10,000 simulations for pricing

### S&P 500 Knock-Out Call Option (1M)
- **Pricing Method**: Monte Carlo simulation for barrier option pricing
- **Risk Factors**: S&P 500 index, VIX, Fed funds rate, dividend yield
- **Assumptions**:
  - 1-month maturity
  - At-the-money strike
  - 110% knock-out barrier level
  - 30 bps transaction cost on roll-over
  - Option expires worthless if barrier is breached 