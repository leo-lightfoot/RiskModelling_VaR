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
        # Dictionary mapping asset categories to their directory paths
        asset_dirs = {
            'Equity': os.path.join(os.getcwd(), 'Equity'),
            'Derivatives': os.path.join(os.getcwd(), 'Derivatives'),
            'Fixed_Income': os.path.join(os.getcwd(), 'Fixed_Income'),
            'Forex': os.path.join(os.getcwd(), 'Forex'),
            'Money_Market': os.path.join(os.getcwd(), 'Money_Market')
        }
        
        # Load equity data
        if os.path.exists(asset_dirs['Equity']):
            equity_csv = os.path.join(asset_dirs['Equity'], 'equity_data.csv')
            if os.path.exists(equity_csv):
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
        
        # Load derivatives data
        if os.path.exists(asset_dirs['Derivatives']):
            # Find all subdirectories in Derivatives
            derivative_dirs = [d for d in os.listdir(asset_dirs['Derivatives']) if os.path.isdir(os.path.join(asset_dirs['Derivatives'], d))]
            for derivative_dir in derivative_dirs:
                full_dir_path = os.path.join(asset_dirs['Derivatives'], derivative_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                for csv_file in csv_files:
                    # Extract name from directory name
                    instrument_name = derivative_dir.replace('_', ' ')
                    self._load_instrument_data(instrument_name, csv_file)

        # Load fixed income data
        if os.path.exists(asset_dirs['Fixed_Income']):
            # Find all subdirectories in Fixed_Income
            fixed_income_dirs = [d for d in os.listdir(asset_dirs['Fixed_Income']) if os.path.isdir(os.path.join(asset_dirs['Fixed_Income'], d))]
            for fi_dir in fixed_income_dirs:
                full_dir_path = os.path.join(asset_dirs['Fixed_Income'], fi_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                for csv_file in csv_files:
                    # Extract name from directory name
                    instrument_name = fi_dir.replace('_', ' ')
                    self._load_instrument_data(instrument_name, csv_file)
        
        # Load Forex data
        if os.path.exists(asset_dirs['Forex']):
            # Find all subdirectories in Forex
            forex_dirs = [d for d in os.listdir(asset_dirs['Forex']) if os.path.isdir(os.path.join(asset_dirs['Forex'], d))]
            for forex_dir in forex_dirs:
                full_dir_path = os.path.join(asset_dirs['Forex'], forex_dir)
                # Find CSV files in this directory
                csv_files = glob(os.path.join(full_dir_path, '*_data.csv')) or glob(os.path.join(full_dir_path, '*.csv'))
                for csv_file in csv_files:
                    # Extract name from directory name
                    instrument_name = forex_dir.replace('_', ' ')
                    self._load_instrument_data(instrument_name, csv_file)
        
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
        
        # Save NAV history
        nav_path = os.path.join(folder_path, 'portfolio_nav_history.csv')
        self.nav_history.to_csv(nav_path)
        
        # Save returns history
        returns_path = os.path.join(folder_path, 'portfolio_returns_history.csv')
        self.returns_history.to_csv(returns_path)
        
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
    
    # Create figure for individual positions (with log scale)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Plot individual position NAVs
    for name in list(portfolio.instruments.keys()) + ['Cash']:
        nav_col = f'{name}_NAV'
        if nav_col in portfolio.nav_history.columns:
            # Only plot positions that have positive values
            if (portfolio.nav_history[nav_col] > 0).all():
                ax2.plot(portfolio.nav_history.index, portfolio.nav_history[nav_col], '-', label=f'{name}')
    
    ax2.set_title('Individual Position NAVs (Log Scale)')
    ax2.set_ylabel('NAV ($)')
    ax2.set_xlabel('Date')
    ax2.set_yscale('log')  # Use logarithmic scale for y-axis
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, which='both', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_individual_navs_log.png', dpi=300)
    plt.close(fig2)
    
    # Create figure for allocation over time
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    allocation_data = {}
    for name in list(portfolio.instruments.keys()) + ['Cash']:
        nav_col = f'{name}_NAV'
        if nav_col in portfolio.nav_history.columns:
            # Calculate percentage allocation (absolute value for plotting purposes)
            allocation_data[name] = portfolio.nav_history[nav_col].abs() / portfolio.nav_history['NAV'].abs() * 100
    
    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data)
        
        # Normalize the allocation percentages to sum to 100%
        row_sums = allocation_df.sum(axis=1)
        for col in allocation_df.columns:
            allocation_df[col] = allocation_df[col] / row_sums * 100
            
        # Plot as lines instead of stacked area to handle negative values
        for col in allocation_df.columns:
            ax3.plot(allocation_df.index, allocation_df[col], '-', label=col)
            
        ax3.set_ylabel('Allocation (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('Portfolio Allocation Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_allocation.png', dpi=300)
    plt.close(fig3)
    
    # Also create standard linear scale charts for comparison
    
    # Linear scale cumulative NAV
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
        'Apple': 4.5,
        'Lockheed_Martin': 4.5,
        'Nvidia': 3.0,
        'Procter_Gamble': 3.0,
        'Johnson_Johnson': 3.0,
        'Toyota': 3.0,
        'Nestle': 3.0,
        'X_Steel': 3.0,
        '1M Crude Oil Futures': 4.0,
        '3M Gold Futures': 4.0,
        '6M Soybean Futures': 4.0,
        'VIX FrontMonth futures': 4.0,
        '1M S&P Futures': 4.0,
        'GBP USD 6M Forward': 5.0,
        'LQD ETF': 5.0,
        '10 year bond': 10.0,
        '1 year EUR bond': 10.0, 
        'High Yield CorpDebt': 10.0,
        'Cash': 13.0  
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
