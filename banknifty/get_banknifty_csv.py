import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define bank tickers and indices tickers
bank_tickers = {
    'AUBANK': 'AUBANK.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'IDFCFIRSTB': 'IDFCFIRSTB.NS',
    'AXISBANK': 'AXISBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'PNB': 'PNB.NS',
    'FEDERALBNK': 'FEDERALBNK.NS',
    'BANKBARODA': 'BANKBARODA.NS',
    'BANDHANBNK': 'BANDHANBNK.NS',
    'SBIN': 'SBIN.NS'
}

index_tickers = {
    'BANKNIFTY': '^NSEBANK',
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
    
    # Fetch data for individual banks
    for bank_name, ticker in bank_tickers.items():
        print(f"Fetching data for {bank_name} ({ticker})")
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            df = df.rename(columns={
                'Open': f'{bank_name}_Open',
                'High': f'{bank_name}_High',
                'Low': f'{bank_name}_Low',
                'Close': f'{bank_name}_Close',
                'Volume': f'{bank_name}_Volume'
            })
            df[f'{bank_name}_PreClose'] = df[f'{bank_name}_Close'].shift(1)  # Add PreClose column
            df = df.drop(columns=['Adj Close'])
            data.append(df)
        except Exception as e:
            print(f"Could not fetch data for {bank_name} ({ticker}): {e}")
    
    # Combine all DataFrames
    all_data = pd.concat(data, axis=1, join='outer')
    return all_data

def save_to_csv(data, filename):
    data.to_csv(filename)
    print(f"Data saved to {filename}")

def main():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2000)).strftime('%Y-%m-%d')
    
    # Fetch data for indices and individual banks
    all_data = fetch_data(bank_tickers, start_date, end_date)
    
    # Save combined data to CSV
    save_to_csv(all_data, 'combined_bank_and_index_data.csv')

if __name__ == '__main__':
    main()
