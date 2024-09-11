import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define NIFTY50 stocks tickers
nifty50_tickers = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'LTIM': 'LTIM.NS',           # Updated from LT to LTIM
    'LT': 'LT.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'AXISBANK': 'AXISBANK.NS',
    'MARUTI': 'MARUTI.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'WIPRO': 'WIPRO.NS',
    'HCLTECH': 'HCLTECH.NS',
    'ADANIENT': 'ADANIENT.NS',    # Updated from ADANIGREEN to ADANIENT
    'NTPC': 'NTPC.NS',
    'TITAN': 'TITAN.NS',
    'POWERGRID': 'POWERGRID.NS',
    'ONGC': 'ONGC.NS',
    'M&M': 'M&M.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',  # Added BAJAJ-AUTO
    'DIVISLAB': 'DIVISLAB.NS',
    'JSWSTEEL': 'JSWSTEEL.NS',
    'TECHM': 'TECHM.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'DRREDDY': 'DRREDDY.NS',
    'COALINDIA': 'COALINDIA.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'BPCL': 'BPCL.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'SBILIFE': 'SBILIFE.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'APOLLOHOSP': 'APOLLOHOSP.NS',
    'GRASIM': 'GRASIM.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'CIPLA': 'CIPLA.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    'SHRIRAMFIN': 'SHRIRAMFIN.NS',  # Added SHRIRAMFIN
    'HINDALCO': 'HINDALCO.NS'      # Added HINDALCO
}

# Define NIFTY50 index ticker
index_tickers = {
    'NIFTY50': '^NSEI',
}

def fetch_data(tickers, start_date, end_date):
    data = []
    
    # Fetch index data
    for index_name, ticker in index_tickers.items():
        print(f"Fetching data for {index_name} ({ticker})")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df = df.rename(columns={
                'Open': f'{index_name}_Open',
                'High': f'{index_name}_High',
                'Low': f'{index_name}_Low',
                'Close': f'{index_name}_Close',
            })
            df[f'{index_name}_PreClose'] = df[f'{index_name}_Close'].shift(1)  # Add PreClose column
            df = df.drop(columns=['Adj Close', 'Volume'])
            data.append(df)
        except Exception as e:
            print(f"Could not fetch data for {index_name} ({ticker}): {e}")
    
    # Fetch data for individual stocks in NIFTY50
    for stock_name, ticker in tickers.items():
        print(f"Fetching data for {stock_name} ({ticker})")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df = df.rename(columns={
                'Open': f'{stock_name}_Open',
                'High': f'{stock_name}_High',
                'Low': f'{stock_name}_Low',
                'Close': f'{stock_name}_Close',
                'Volume': f'{stock_name}_Volume'
            })
            df[f'{stock_name}_PreClose'] = df[f'{stock_name}_Close'].shift(1)  # Add PreClose column
            df = df.drop(columns=['Adj Close'])
            data.append(df)
        except Exception as e:
            print(f"Could not fetch data for {stock_name} ({ticker}): {e}")
    
    # Combine all DataFrames
    all_data = pd.concat(data, axis=1, join='outer')
    return all_data

def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")

def main():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')
    
    # Fetch data for indices and individual NIFTY50 stocks
    all_data = fetch_data(nifty50_tickers, start_date, end_date)
    
    # Save combined data to CSV
    save_to_csv(all_data, 'combined_nifty50_and_index_data.csv')

if __name__ == '__main__':
    main()
