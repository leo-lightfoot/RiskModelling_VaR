import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import glob

# Incorporate the Portfolio class directly instead of importing
class Portfolio:
    def __init__(self, initial_capital=1000000.0, start_date=None):
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
        
    def _load_instrument_data(self, instrument_name, file_path):
        """Load instrument data from specified CSV file
        
        Args:
            instrument_name: Name to use for the instrument in the portfolio
            file_path: Full path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.asfreq('D', method='ffill')
            
            # Find NAV and Log Return columns
            nav_col = next((col for col in df.columns if 'NAV' in col), None)
            return_col = next((col for col in df.columns if 'Return' in col or 'return' in col), None)
            
            if nav_col and return_col:
                # Create a subset with just the NAV and return columns
                instrument_df = df[[nav_col, return_col]].copy()
                instrument_df.columns = ['NAV', 'Return']
                self.instruments[instrument_name] = instrument_df
                print(f"Loaded {instrument_name} data with {len(instrument_df)} observations")
                return True
            else:
                print(f"Error: Could not find NAV and Return columns in {file_path}")
                return False
        except FileNotFoundError:
            print(f"Error: {instrument_name} pricing data not found at {file_path}")
            return False
        except Exception as e:
            print(f"Error loading {instrument_name} data: {str(e)}")
            return False
        
    def _load_pricing_data(self):
        """Load instrument pricing data from asset directories"""
        # Get parent directory (workspace root)
        parent_dir = os.path.dirname(os.getcwd())
        
        # Dictionary mapping asset categories to their directory paths
        asset_dirs = {
            'Equity': os.path.join(parent_dir, 'Equity'),
            'Derivatives': os.path.join(parent_dir, 'Derivatives'),
            'Fixed_Income': os.path.join(parent_dir, 'Fixed_Income'),
            'Forex': os.path.join(parent_dir, 'Forex'),
            'Money_Market': os.path.join(parent_dir, 'Money_Market')
        }
        
        print(f"Current working directory: {os.getcwd()}")
        print(f"Parent directory: {parent_dir}")
        print(f"Asset directories: {asset_dirs}")
        
        # Load equity data
        if os.path.exists(asset_dirs['Equity']):
            print(f"Found Equity directory: {asset_dirs['Equity']}")
            equity_csv = os.path.join(asset_dirs['Equity'], 'equity_data.csv')
            if os.path.exists(equity_csv):
                print(f"Found equity data CSV: {equity_csv}")
                df = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
                # Print available columns for debugging
                print(f"Available equity columns: {[col for col in df.columns if col.startswith('NAV_')]}")
                # Extract each equity instrument (assumed to be in pairs of NAV_name and Log_Return_name)
                nav_cols = [col for col in df.columns if col.startswith('NAV_')]
                for nav_col in nav_cols:
                    instrument_name = nav_col.replace('NAV_', '').replace('_price', '').title()
                    return_col = nav_col.replace('NAV_', 'Log_Return_')
                    if return_col in df.columns:
                        # Create a subset with just the NAV and return columns for this instrument
                        instrument_df = df[[nav_col, return_col]].copy()
                        instrument_df.columns = ['NAV', 'Return']
                        self.instruments[instrument_name] = instrument_df
                        print(f"Loaded {instrument_name} data from Equity with {len(instrument_df)} observations")
            else:
                print(f"Equity data CSV not found: {equity_csv}")
        else:
            print(f"Equity directory not found: {asset_dirs['Equity']}")
        
        # Load derivatives data
        if os.path.exists(asset_dirs['Derivatives']):
            print(f"Found Derivatives directory: {asset_dirs['Derivatives']}")
            # Find all subdirectories in Derivatives
            derivative_dirs = [d for d in os.listdir(asset_dirs['Derivatives']) if os.path.isdir(os.path.join(asset_dirs['Derivatives'], d))]
            print(f"Found derivative directories: {derivative_dirs}")
            for derivative_dir in derivative_dirs:
                full_dir_path = os.path.join(asset_dirs['Derivatives'], derivative_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                print(f"Found CSV files in {derivative_dir}: {csv_files}")
                for csv_file in csv_files:
                    # Map directory names to instrument names in our allocation
                    instrument_mapping = {
                        'VIX_FrontMonth_futures': 'VIX_Futures',
                        'USD-JPY_ATM_Put_Option': 'USDJPY_ATM_Put',
                        'Exxon_ITM_Put_option': 'XOM_ITM_Put',
                        'EUR-USD_ATM_Call_Option': 'EURUSD_ATM_Call',
                        'Costco_ITM_Call_option': 'Costco_ITM_Call',
                        '6M_Soybean_Futures': 'Soybean_Futures_6M',
                        '3M_Gold_Futures': 'Gold_Futures_3M',
                        '1M_S&P_Futures': 'SP500_Futures_1M',
                        '1M_Crude_Oil_Futures': 'Crude_Oil_Futures_1M'
                    }
                    # Get the instrument name from mapping or use directory name as fallback
                    instrument_name = instrument_mapping.get(derivative_dir, derivative_dir.replace('_', ' '))
                    self._load_instrument_data(instrument_name, csv_file)
        else:
            print(f"Derivatives directory not found: {asset_dirs['Derivatives']}")

        # Load fixed income data
        if os.path.exists(asset_dirs['Fixed_Income']):
            print(f"Found Fixed_Income directory: {asset_dirs['Fixed_Income']}")
            # Find all subdirectories in Fixed_Income
            fixed_income_dirs = [d for d in os.listdir(asset_dirs['Fixed_Income']) if os.path.isdir(os.path.join(asset_dirs['Fixed_Income'], d))]
            print(f"Found fixed income directories: {fixed_income_dirs}")
            for fi_dir in fixed_income_dirs:
                full_dir_path = os.path.join(asset_dirs['Fixed_Income'], fi_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                print(f"Found CSV files in {fi_dir}: {csv_files}")
                for csv_file in csv_files:
                    # Map directory names to instrument names in our allocation
                    instrument_mapping = {
                        'Revenue_Bond': '30y_Revenue_Bond',
                        '5year_corp_green_bond': '5y_Green_Bond',
                        'LQD_ETF': 'LQD_ETF',
                        'High_Yield_CorpDebt': 'High_Yield_Corp_Debt',
                        '10_year_TIPS': '10y_TIPS',
                        '10_year_bond': '10y_Bond',
                        '1_year_EUR_bond': '1y_EUR_ZCB'
                    }
                    # Get the instrument name from mapping or use directory name as fallback
                    instrument_name = instrument_mapping.get(fi_dir, fi_dir.replace('_', ' '))
                    self._load_instrument_data(instrument_name, csv_file)
        else:
            print(f"Fixed_Income directory not found: {asset_dirs['Fixed_Income']}")
        
        # Load Forex data
        if os.path.exists(asset_dirs['Forex']):
            print(f"Found Forex directory: {asset_dirs['Forex']}")
            # Find all subdirectories in Forex
            forex_dirs = [d for d in os.listdir(asset_dirs['Forex']) if os.path.isdir(os.path.join(asset_dirs['Forex'], d))]
            print(f"Found forex directories: {forex_dirs}")
            for forex_dir in forex_dirs:
                full_dir_path = os.path.join(asset_dirs['Forex'], forex_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                print(f"Found CSV files in {forex_dir}: {csv_files}")
                for csv_file in csv_files:
                    # Map directory names to instrument names in our allocation
                    instrument_mapping = {
                        'INR_USD_3M_Forward_Short': 'USDINR_Forward',
                        'GBP_USD_6M_Forward': 'GBPUSD_Forward'
                    }
                    # Get the instrument name from mapping or use directory name as fallback
                    instrument_name = instrument_mapping.get(forex_dir, forex_dir.replace('_', ' '))
                    self._load_instrument_data(instrument_name, csv_file)
        else:
            print(f"Forex directory not found: {asset_dirs['Forex']}")
        
        # Print loaded instruments for debugging
        print(f"Loaded {len(self.instruments)} instruments: {list(self.instruments.keys())}")
        
        # Align datasets to common date range
        self._align_datasets()
    
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
            
        # Create a common daily date range
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex all datasets to the same daily range
        for name, data in self.instruments.items():
            if data is not None and not data.empty:
                self.instruments[name] = data.reindex(daily_range).fillna(method='ffill')
        
        print(f"Datasets aligned with {len(daily_range)} daily observations from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

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
            self._load_instrument_data(name, data_path)
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
        
        # Create a complete daily date range
        daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily cash return (compounded daily)
        daily_cash_return = (1 + cash_rate) ** (1/252) - 1
        
        # Store initial allocation amounts
        initial_allocations = {name: self.holdings.get(name, 0) for name in self.instruments.keys()}
        cash_allocation = self.holdings.get('Cash', 0)
        
        # Calculate initial prices and shares for each instrument
        instrument_shares = {}
        for name, amount in initial_allocations.items():
            if name in self.instruments and self.instruments[name] is not None:
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
        
        # Initialize tracking values
        instrument_navs = {}
        cash_nav = cash_allocation
        
        nav_records = []
        return_records = []
        
        prev_total_nav = sum(initial_allocations.values()) + cash_allocation
        
        for date in daily_range:
            # Calculate current NAV for each instrument based on shares and current price
            for name, shares in instrument_shares.items():
                if name in self.instruments and self.instruments[name] is not None and date in self.instruments[name].index:
                    if 'NAV' in self.instruments[name].columns and not pd.isna(self.instruments[name].loc[date, 'NAV']):
                        current_price = self.instruments[name].loc[date, 'NAV']
                        instrument_navs[name] = shares * current_price
                    else:
                        # If NAV not available for this date, use previous NAV
                        instrument_navs[name] = instrument_navs.get(name, initial_allocations.get(name, 0))
                else:
                    # If instrument data not available for this date, use previous NAV
                    instrument_navs[name] = instrument_navs.get(name, initial_allocations.get(name, 0))
            
            # Update cash NAV
            cash_nav = cash_nav * (1 + daily_cash_return)
            
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
        
        # Create history DataFrames
        self.nav_history = pd.DataFrame(nav_records).set_index('Date')
        self.returns_history = pd.DataFrame(return_records).set_index('Date')
        
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
                                      Example: {'VIX FrontMonth futures': 30.0, 'Apple': 25.0}
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
    ax1.plot(portfolio.nav_history.index, portfolio.nav_history['NAV'], 'b-', linewidth=2)
    ax1.set_title('Portfolio Cumulative NAV (Log Scale)')
    ax1.set_ylabel('NAV ($)')
    ax1.set_xlabel('Date')
    ax1.set_yscale('log')  # Use logarithmic scale for y-axis
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal grid lines specifically for log scale
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(True, which='both', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cumulative_nav_log.png', dpi=300)
    plt.close(fig1)
    
    # Create improved individual asset chart with better y-axis limits
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    
    # Group instruments by type
    instrument_groups = {
        'Equity': ['Apple', 'Lockheed_Martin', 'Nvidia', 'Procter_Gamble', 
                   'Johnson_Johnson', 'Toyota', 'Nestle', 'X_Steel'],
        'Fixed_Income': ['10y_Bond', '10y_TIPS', '1y_EUR_ZCB', '5y_Green_Bond', 
                         'High_Yield_Corp_Debt', 'LQD_ETF', '30y_Revenue_Bond'],
        'Derivatives': ['SP500_Futures_1M', 'VIX_Futures', 'Crude_Oil_Futures_1M', 
                       'Gold_Futures_3M', 'Soybean_Futures_6M', 'Costco_ITM_Call', 
                       'EURUSD_ATM_Call', 'XOM_ITM_Put', 'USDJPY_ATM_Put'],
        'Forex': ['GBPUSD_Forward', 'USDINR_Forward'],
        'Cash': ['Cash']
    }
    
    # Define colors and styles for each group
    group_styles = {
        'Equity': {'color': 'blue', 'alpha': 0.8, 'linestyle': '-'},
        'Fixed_Income': {'color': 'green', 'alpha': 0.8, 'linestyle': '--'},
        'Derivatives': {'color': 'red', 'alpha': 0.8, 'linestyle': '-.'},
        'Forex': {'color': 'purple', 'alpha': 0.8, 'linestyle': ':'},
        'Cash': {'color': 'black', 'alpha': 0.8, 'linestyle': '-'}
    }
    
    # Define minimum threshold for values to display (eliminate extreme negatives)
    min_threshold = 100.0  # $100 minimum value to filter out extreme dips
    
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
    
    # Create a more readable legend with columns
    leg = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=4, fontsize=9, frameon=True, facecolor='white', edgecolor='gray')
    
    # Make the plot a bit taller to accommodate the legend below
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_individual_navs_log.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Also create linear scale chart for the cumulative NAV
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(portfolio.nav_history.index, portfolio.nav_history['NAV'], 'b-', linewidth=2)
    ax4.set_title('Portfolio Cumulative NAV (Linear Scale)')
    ax4.set_ylabel('NAV ($)')
    ax4.set_xlabel('Date')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cumulative_nav.png', dpi=300)
    plt.close(fig4)
    
    print(f"Performance charts saved with prefix: {save_prefix}")

def main():
    # Example allocation with a diversified portfolio
    custom_allocation = {
        'Apple': 3.0,
        'Lockheed_Martin': 3.0,
        'Nvidia': 3.0,
        'Procter_Gamble': 3.0,
        'Johnson_Johnson': 3.0,
        'Toyota': 3.0,
        'Nestle': 3.0,
        'X_Steel': 3.0,
        '10y_Bond': 4.0,
        'LQD_ETF': 4.0,
        '10y_TIPS': 4.0,
        '1y_EUR_ZCB': 4.0,
        'High_Yield_Corp_Debt': 4.0,
        '5y_Green_Bond': 4.0,
        '30y_Revenue_Bond': 4.0,
        'SP500_Futures_1M': 4.0,
        'VIX_Futures': 4.0,
        'Crude_Oil_Futures_1M': 4.0,
        'Gold_Futures_3M': 4.0,
        'Soybean_Futures_6M': 4.0,
        'Costco_ITM_Call': 4.0,
        'EURUSD_ATM_Call': 4.0,
        'XOM_ITM_Put': 4.0,
        'USDJPY_ATM_Put': 4.0,
        'GBPUSD_Forward': 4.0,
        'USDINR_Forward': 4.0,
        'Cash': 4.0
    }
    
    # Output directory
    output_dir = 'portfolio_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create portfolio with custom allocation - use dates that we have full data for
    portfolio = create_portfolio_with_custom_allocation(
        custom_allocation,
        start_date='2005-01-01',  # Start after we have data for all assets
        end_date='2024-12-31'
    )
    
    # Plot performance and save to output directory
    plot_portfolio_performance(portfolio, os.path.join(output_dir, 'portfolio'))
    
    # Save results to output directory
    portfolio.save_results(output_dir)

if __name__ == "__main__":
    main()
