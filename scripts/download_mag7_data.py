"""
Script to download and save MAG7 stock data for portfolio optimization.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def download_mag7_data():
    """Download MAG7 stock data and save to parquet file."""
    # Define MAG7 stocks
    mag7_tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    # Define date range (20 years of data)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Fetch data
    raw_data = yf.download(mag7_tickers, start=start_date, end=end_date)
    
    # Check the structure of the downloaded data
    print("Raw data structure:")
    print(f"Columns: {raw_data.columns}")
    print(f"Shape: {raw_data.shape}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save raw data to parquet (preserving all fields and multiindex)
    raw_filepath = os.path.join(data_dir, 'mag7_data_raw.parquet')
    raw_data.to_parquet(raw_filepath)
    print(f"Raw data saved to {raw_filepath}")
    
    # Also save adjusted close data to CSV for backward compatibility
    if isinstance(raw_data.columns, pd.MultiIndex):
        # MultiIndex columns - select 'Adj Close' column
        if 'Adj Close' in raw_data.columns.levels[0]:
            data = raw_data['Adj Close']
        else:
            # Fallback to 'Close' if 'Adj Close' is not available
            data = raw_data['Close']
    else:
        # Single level columns - assume it's already the adjusted close prices
        data = raw_data
        
    print(f"Adjusted close data shape: {data.shape}")
    
    # Save adjusted close data to CSV
    adj_close_filepath = os.path.join(data_dir, 'mag7_data.csv')
    data.to_csv(adj_close_filepath)
    print(f"Adjusted close data saved to {adj_close_filepath}")
    
    return raw_data, data

if __name__ == "__main__":
    download_mag7_data()