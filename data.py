"""
CoinGecko Data Analysis Script
"""
# Import the requests library for making HTTP requests
import requests
# Import pandas library for data manipulation and analysis
import pandas as pd
# Import datetime and timedelta classes from the datetime module
from datetime import datetime, timedelta


def get_price_data(api_key, coin_id, days):
    """
    Retrieves historical price data from the CoinGecko API.

    Args:
        api_key (str): Your CoinGecko API key.
        coin_id (str): The ID of the cryptocurrency to retrieve data for.
        days (int): The number of days of historical data to retrieve.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical price data.
    """
    # Get the current date
    today = datetime.now()
    # Calculate the start date by subtracting 'days' from today
    start_date = (today - timedelta(days=days)).strftime('%d-%m-%Y')
    # Format the end date (today) as a string
    end_date = today.strftime('%d-%m-%Y')

    # Construct the API URL with the required parameters
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_date}&to={end_date}"

    # Send a GET request to the API with the API key in the headers
    response = requests.get(url, headers={"Accept": "application/json", "X-CoinGecko-API-Key": api_key})

    # Parse the JSON response
    data = response.json()
    # Extract the 'prices' data from the response
    prices = data['prices']
    # Create a DataFrame from the prices data with 'time' and 'price' columns
    df = pd.DataFrame(prices, columns=['time', 'price'])

    # Convert the 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    # Convert the 'price' column to numeric type
    df['price'] = pd.to_numeric(df['price'])

    # Set the 'time' column as the index of the DataFrame
    df.set_index('time', inplace=True)

    # Return the processed DataFrame
    return df


def calculate_returns(prices):
    """
    Calculates daily returns from a DataFrame of prices.

    Args:
        prices (pandas.DataFrame): A DataFrame of prices.

    Returns:
        pandas.DataFrame: A DataFrame of daily returns.
    """
    # Calculate the percentage change of prices
    returns = prices.pct_change(1)
    # Set the first row of returns to 0 (since there's no previous day to calculate return)
    returns.iloc[0] = 0

    # Return the DataFrame of daily returns
    return returns