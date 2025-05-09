import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob
import sys

# Incorporate the Portfolio class directly instead of importing
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
        combined_file = os.path.join(os.getcwd(), 'combined_instrument_returns.csv')
        
        # If the file doesn't exist in the current directory, try the parent directory
        if not os.path.exists(combined_file):
            parent_dir = os.path.dirname(os.getcwd())
            combined_file = os.path.join(parent_dir, 'combined_instrument_returns.csv')
        
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
            folder_path = os.path.join(os.getcwd(), 'portfolio_results')
            
        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)
        
        # Save NAV history with retry mechanism
        nav_path = os.path.join(folder_path, 'portfolio_nav_history.csv')
        try:
            self.nav_history.to_csv(nav_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {nav_path}. File may be in use.")
        
        # Save returns history with retry mechanism
        returns_path = os.path.join(folder_path, 'portfolio_returns_history.csv')
        try:
            self.returns_history.to_csv(returns_path)
        except PermissionError:
            print(f"Warning: Permission error when saving {returns_path}. File may be in use.")
        
        print(f"Portfolio results saved to {folder_path}")

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
    Create separate plots for cumulative NAV and individual position NAVs.
    
    Args:
        portfolio: Portfolio object with calculated returns
        save_prefix: Prefix for saving plot files
    """
    if portfolio.nav_history.empty:
        print("No performance data available. Run calculate_returns() first.")
        return
    
    # Create figure for cumulative NAV (with log scale)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(portfolio.nav_history.index, portfolio.nav_history['NAV'], 'b-', linewidth=2, label='Portfolio NAV')
    ax1.set_title('Portfolio Cumulative NAV (Log Scale)')
    ax1.set_ylabel('NAV ($)')
    ax1.set_xlabel('Date')
    ax1.set_yscale('log')  # Use logarithmic scale for y-axis
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal grid lines specifically for log scale
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(True, which='both', linestyle='-', alpha=0.2)
    ax1.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cumulative_nav_log.png', dpi=300)
    plt.close(fig1)
    
    # Create improved individual asset chart with better y-axis limits
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    
    # Define minimum threshold for values to display (eliminate extreme negatives)
    min_threshold = 100.0  # $100 minimum value to filter out extreme dips
    
    # Get all instrument columns in portfolio NAV history
    instrument_navs = [col for col in portfolio.nav_history.columns if col.endswith('_NAV') and col != 'Cash_NAV']
    instrument_names = [col.replace('_NAV', '') for col in instrument_navs]
    
    print(f"Found {len(instrument_names)} instruments for plotting: {instrument_names}")
    
    # Group instruments by type based on name patterns
    instrument_groups = {}
    
    # Map instruments to groups based on name pattern recognition
    for name in instrument_names:
        # Classify based on patterns in names
        if any(equity_term in name.lower() for equity_term in ['apple', 'lockheed', 'nvidia', 'procter', 'johnson', 'toyota', 'nestle', 'x_steel']):
            group = 'Equity'
        elif any(fi_term in name.lower() for fi_term in ['bond', 'tips', 'zcb', 'lqd', 'green', 'yield']):
            group = 'Fixed_Income'
        elif any(deriv_term in name.lower() for deriv_term in ['futures', 'call', 'put', 'cds', 'swap', 'barrier']):
            group = 'Derivatives'
        elif any(forex_term in name.lower() for forex_term in ['usd', 'gbp', 'eur', 'jpy', 'inr', 'forward']):
            group = 'Forex'
        elif name.lower() == 'cash':
            group = 'Cash'
        else:
            group = 'Other'
            
        if group not in instrument_groups:
            instrument_groups[group] = []
        instrument_groups[group].append(name)
    
    print(f"Instrument groups: {instrument_groups}")
    
    # Define colors and styles for each group
    group_styles = {
        'Equity': {'color': 'blue', 'alpha': 0.8, 'linestyle': '-'},
        'Fixed_Income': {'color': 'green', 'alpha': 0.8, 'linestyle': '--'},
        'Derivatives': {'color': 'red', 'alpha': 0.8, 'linestyle': '-.'},
        'Forex': {'color': 'purple', 'alpha': 0.8, 'linestyle': ':'},
        'Cash': {'color': 'black', 'alpha': 0.8, 'linestyle': '-'},
        'Other': {'color': 'gray', 'alpha': 0.8, 'linestyle': '-'}
    }
    
    # Prepare a list to collect the asset data for plotting
    asset_plots = []
    
    # First pass: collect data points
    for group_name, instruments in instrument_groups.items():
        for i, name in enumerate(instruments):
            nav_col = f'{name}_NAV'
            if nav_col in portfolio.nav_history.columns:
                # Filter to only show values above the threshold
                filtered_data = portfolio.nav_history[portfolio.nav_history[nav_col] > min_threshold].copy()
                if not filtered_data.empty:
                    style = group_styles[group_name]
                    # Vary the color slightly within each group
                    color = style['color']
                    if len(instruments) > 1:
                        # Adjust color brightness for differentiation within group
                        brightness = 0.5 + (i / (len(instruments) * 2))
                        if color == 'blue':
                            color = (0, 0, brightness)
                        elif color == 'green':
                            color = (0, brightness, 0)
                        elif color == 'red':
                            color = (brightness, 0, 0)
                        elif color == 'purple':
                            color = (brightness, 0, brightness)
                    
                    asset_plots.append({
                        'name': name,
                        'group': group_name,
                        'data': filtered_data[nav_col],
                        'style': style,
                        'color': color,
                        'final_value': filtered_data[nav_col].iloc[-1] if len(filtered_data) > 0 else 0
                    })
                else:
                    print(f"Warning: No values above threshold for {name}")
    
    print(f"Found {len(asset_plots)} assets with data above threshold")
    
    # Sort assets by final value to plot highest values last (on top) for better visibility
    asset_plots.sort(key=lambda x: x['final_value'])
    
    # Second pass: plot the data
    for asset in asset_plots:
        ax2.plot(asset['data'].index, asset['data'], 
                 linestyle=asset['style']['linestyle'], 
                 color=asset['color'], 
                 alpha=asset['style']['alpha'],
                 label=f"{asset['name']} ({asset['group']})")
    
    # Set y-axis parameters
    ax2.set_yscale('log')
    ax2.set_ylim(min_threshold, None)  # Set minimum y value to our threshold
    
    ax2.set_title('Individual Position NAVs (Log Scale)', fontsize=14)
    ax2.set_ylabel('NAV ($)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # Improve the grid for better readability
    ax2.grid(True, which='both', linestyle='-', alpha=0.2)
    
    # Only create legend if we have assets to show
    if asset_plots:
        # Create a more readable legend with columns
        leg = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                        ncol=4, fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
        
        # Make the plot a bit taller to accommodate the legend below
        plt.subplots_adjust(bottom=0.25)
    else:
        print("Warning: No assets to display in individual NAVs plot")
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_individual_navs_log.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
        
    print(f"Performance charts saved with prefix: {save_prefix}")

def main():
    # Create a diversified portfolio using the instruments from the combined CSV file
    custom_allocation = {
        'Apple': 2.0,
        'Lockheed_martin': 2.0,
        'Nvidia': 2.0,
        'Procter_gamble': 2.0,
        'Johnson_johnson': 2.0,
        'Toyota': 2.0,
        'Nestle': 2.0,
        'X_steel': 2.0,
        '10y_treasury': 4.0,
        'Lqd_etf': 4.0,
        '10y_tips': 4.0,
        '1y_eur_zcb': 4.0,
        'High_yield_corp_debt': 4.0,
        '5y_green_bond': 4.0,
        '30y_revenue_bond': 4.0,
        'Sp500_futures_1m': 4.0,
        'Vix_futures': 2.0,
        'Crude_oil_futures': 2.0,
        'Gold_futures': 4.0,
        'Soybean_futures': 2.0,
        'Costco_itm_call': 4.0,
        'Xom_itm_put': 4.0,
        'Eurusd_atm_call': 4.0,
        'Usdjpy_atm_put': 4.0,
        'Gbpusd_6m_forward': 3.0,
        'Usdinr_3m_forward': 3.0,
        'Ford_cds': 3.0, 
        'Dax_variance_swap': 3.0,
        'Nikkei_asian_put': 3.0,
        'Spx_knockout_call': 3.0,
        'Cash': 8.0
    }
    
    # Output directory
    output_dir = 'portfolio_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create portfolio with custom allocation - use dates that we have full data for
    portfolio = create_portfolio_with_custom_allocation(
        custom_allocation,
        initial_capital=10000000.0,  # Explicitly set to 10 million
        start_date='2005-01-03',  # Start date from our dataset
        end_date='2024-12-31'
    )
    
    # Plot performance and save to output directory
    plot_portfolio_performance(portfolio, os.path.join(output_dir, 'portfolio'))
    
    # Save results to output directory
    portfolio.save_results(output_dir)

if __name__ == "__main__":
    main()
