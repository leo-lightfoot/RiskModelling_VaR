import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker and date range
ticker = "INR=X"
start_date = "2005-01-01"
end_date = "2024-12-31"

# Download data from Yahoo Finance
usd_inr = yf.download(ticker, start=start_date, end=end_date)

# Check and clean data
usd_inr = usd_inr['Close'].dropna()
usd_inr.name = 'USD/INR'

# Save to CSV (optional)
usd_inr.to_csv(r"C:\Users\abdul\Desktop\OLD_RAW_DATA\usd_inr_2005_2024.csv")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(usd_inr, label='USD/INR Exchange Rate')
plt.title('USD/INR Exchange Rate (2005–2024)')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
