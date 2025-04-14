import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from typing import List, Dict

class Backtester:
    """
    Simulates a trading strategy across multiple tickers simultaneously.
    Handles portfolio management (single cash pool, multiple positions) 
    and performance tracking.
    Driven externally by processing data date by date.
    """

    def __init__(self, tickers: List[str], initial_capital: float = 10000.0, 
                 transaction_cost: float = 0.001, allocation_per_trade: float = 0.10):
        """
        Initializes the Multi-Ticker Backtester.

        Args:
            tickers (List[str]): List of ticker symbols to trade.
            initial_capital (float): Starting cash for the backtest.
            transaction_cost (float): Cost per transaction (e.g., 0.001 for 0.1%).
            allocation_per_trade (float): Max fraction of portfolio value to allocate 
                                          to a single new buy trade.
        """
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.allocation_per_trade = allocation_per_trade
        self.console = Console()

        # Portfolio state variables
        self.cash = initial_capital
        self.positions: Dict[str, float] = {ticker: 0.0 for ticker in tickers} # Ticker -> Num Shares
        self.portfolio_value = initial_capital
        self.history = [] # List to store portfolio state at each step

    def _calculate_portfolio_value(self, current_prices: Dict[str, float]):
        """Calculates the current total portfolio value."""
        holdings_value = sum(
            self.positions[ticker] * current_prices.get(ticker, 0) 
            for ticker in self.tickers if self.positions[ticker] > 0
        )
        return self.cash + holdings_value

    def process_date(self, date, prices: Dict[str, float], signals: Dict[str, int]):
        """
        Processes trades and updates portfolio for a single date across all tickers.

        Args:
            date: The current date (e.g., Timestamp or string).
            prices (Dict[str, float]): Dictionary of {ticker: close_price} for the date.
            signals (Dict[str, int]): Dictionary of {ticker: signal (0/1/2)} for the date.
        """
        current_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        trades_today = [] # Track actions for logging

        # 1. Calculate portfolio value *before* potential trades for allocation decisions
        self.portfolio_value = self._calculate_portfolio_value(prices)
        available_cash_for_trades = self.cash
        
        # 2. Process SELL signals first to free up cash
        for ticker in self.tickers:
            signal = signals.get(ticker)
            current_price = prices.get(ticker)

            if signal == 0 and self.positions.get(ticker, 0) > 0 and current_price is not None:
                shares_to_sell = self.positions[ticker]
                sell_value = shares_to_sell * current_price
                cost = sell_value * self.transaction_cost
                self.cash += (sell_value - cost)
                self.positions[ticker] = 0
                trades_today.append({
                    "Ticker": ticker, 
                    "Action": "SELL", 
                    "Shares": -shares_to_sell, 
                    "Price": current_price, 
                    "Cost": cost
                })
                available_cash_for_trades += (sell_value - cost) # Update available cash *after* sell

        # 3. Process BUY signals based on available cash and allocation
        potential_buys = {ticker: prices[ticker] for ticker, signal in signals.items() 
                          if signal == 2 and self.positions.get(ticker, 0) == 0 and prices.get(ticker) is not None}
        
        # Simple allocation: Allocate evenly among potential buys, up to limit per trade
        num_potential_buys = len(potential_buys)
        cash_per_buy = available_cash_for_trades / num_potential_buys if num_potential_buys > 0 else 0
        
        for ticker, current_price in potential_buys.items():
            # Determine capital to allocate: min(cash_available_for_this_buy, max_allocation_value)
            max_allocation_value = self.portfolio_value * self.allocation_per_trade
            capital_to_allocate = min(cash_per_buy, max_allocation_value)
            
            shares_to_buy = (capital_to_allocate / current_price) * (1 - self.transaction_cost) # Adjust for cost
            buy_value = shares_to_buy * current_price
            cost = buy_value * self.transaction_cost
            total_cost = buy_value + cost

            if shares_to_buy > 1e-6 and self.cash >= total_cost: # Check if affordable and non-zero
                self.positions[ticker] += shares_to_buy
                self.cash -= total_cost
                trades_today.append({
                    "Ticker": ticker, 
                    "Action": "BUY", 
                    "Shares": shares_to_buy, 
                    "Price": current_price, 
                    "Cost": cost
                })
            # Else: Cannot afford or allocation is too small

        # 4. Recalculate final portfolio value for the day
        self.portfolio_value = self._calculate_portfolio_value(prices)

        # 5. Log the state for this step
        log_entry = {
            "Date": current_date_str,
            "Cash": self.cash,
            "Holdings Value": self.portfolio_value - self.cash,
            "Portfolio Value": self.portfolio_value,
            "Positions": self.positions.copy(), # Store snapshot
            "Trades": trades_today
        }
        self.history.append(log_entry)

        # 6. Print live feedback (adapt rich table for multi-ticker)
        self._log_daily_status(current_date_str, trades_today)

    def _log_daily_status(self, date_str: str, trades: List[Dict]):
        """Logs the portfolio status and trades for the current day using Rich."""
        
        summary_table = Table(title=f"Portfolio Status - {date_str}", show_header=True, header_style="bold blue")
        summary_table.add_column("Metric", style="dim", width=15)
        summary_table.add_column("Value", justify="right")
        summary_table.add_row("Cash", f"${self.cash:,.2f}")
        summary_table.add_row("Holdings Value", f"${self.portfolio_value - self.cash:,.2f}")
        summary_table.add_row("Total Value", f"[bold cyan]${self.portfolio_value:,.2f}[/bold cyan]")
        self.console.print(summary_table)

        if self.positions and any(v > 1e-6 for v in self.positions.values()):
            position_table = Table(title="Current Positions", show_header=True, header_style="bold magenta")
            position_table.add_column("Ticker")
            position_table.add_column("Shares", justify="right")
            for ticker, size in self.positions.items():
                if size > 1e-6:
                    position_table.add_row(ticker, f"{size:,.4f}")
            self.console.print(position_table)

        if trades:
            trade_table = Table(title="Trades Executed", show_header=True, header_style="bold yellow")
            trade_table.add_column("Ticker")
            trade_table.add_column("Action")
            trade_table.add_column("Shares", justify="right")
            trade_table.add_column("Price", justify="right")
            trade_table.add_column("Cost", justify="right")
            for trade in trades:
                action_color = "green" if trade['Action'] == "BUY" else "red"
                trade_table.add_row(
                    trade['Ticker'],
                    f"[{action_color}]{trade['Action']}[/{action_color}]",
                    f"{trade['Shares']:,.4f}",
                    f"${trade['Price']:,.2f}",
                    f"${trade['Cost']:,.2f}"
                )
            self.console.print(trade_table)
        
        self.console.print("---") # Separator


    def get_history(self) -> pd.DataFrame:
         """Returns the backtest history as a pandas DataFrame."""
         if not self.history:
             return pd.DataFrame()
         # Convert list of dicts to DataFrame, handle positions dict
         history_df = pd.DataFrame(self.history)
         history_df['Date'] = pd.to_datetime(history_df['Date'])
         history_df = history_df.set_index('Date')
         # TODO: Consider how to best represent positions and trades in the history df
         return history_df

    def display_summary(self):
        """
        Displays a summary of the backtest performance.
        """
        if not self.history:
            self.console.print("[yellow]No history recorded for the backtest.[/yellow]")
            return

        history_df = self.get_history()
        if history_df.empty:
             self.console.print("[yellow]History is empty, cannot generate summary.[/yellow]")
             return
             
        final_value = history_df['Portfolio Value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Max Drawdown Calculation
        peak = history_df['Portfolio Value'].cummax()
        drawdown = (history_df['Portfolio Value'] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Count trades (simple count of trade entries)
        num_trades = sum(len(d.get('Trades', [])) for d in self.history)
        total_trade_costs = sum(trade['Cost'] for d in self.history for trade in d.get('Trades', []))
        
        summary_table = Table(title="Multi-Ticker Backtest Summary", show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="dim", width=20)
        summary_table.add_column("Value", justify="right")

        summary_table.add_row("Tickers", ", ".join(self.tickers))
        summary_table.add_row("Initial Capital", f"${self.initial_capital:,.2f}")
        summary_table.add_row("Final Portfolio Value", f"${final_value:,.2f}")
        summary_table.add_row("Total Return", f"{total_return:.2f}%")
        summary_table.add_row("Max Drawdown", f"{max_drawdown:.2f}%")
        summary_table.add_row("Total Trades Executed", str(num_trades))
        summary_table.add_row("Total Trade Costs", f"${total_trade_costs:,.2f}")

        self.console.print(summary_table)
        
        # TODO: Add plotting via utils.py later (e.g., portfolio value over time)

# Note: Removed __main__ example as this class is now driven externally. 