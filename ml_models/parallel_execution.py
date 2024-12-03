import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def generate_sales_data(start_date='2022-01-01', periods=730):
    """
    Generate synthetic sales data with realistic patterns for demand forecasting.
    
    Parameters:
    start_date (str): Start date for the dataset
    periods (int): Number of days to generate data for
    
    Returns:
    pandas.DataFrame: DataFrame containing daily sales data with various features
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Initialize DataFrame
    df = pd.DataFrame(index=dates)
    
    # Add basic time features
    df['date'] = df.index
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Generate base demand with trend
    np.random.seed(42)
    trend = np.linspace(100, 150, len(df))
    
    # Add seasonality
    seasonal_pattern = 20 * np.sin(2 * np.pi * df.index.dayofyear / 365)
    
    # Weekly pattern
    weekly_pattern = -15 * df['is_weekend'] + 10 * (df['day_of_week'] == 4)  # Higher on Fridays, lower on weekends
    
    # Generate quantity sold with various patterns
    df['quantity'] = trend + seasonal_pattern + weekly_pattern
    
    # Add noise
    df['quantity'] += np.random.normal(0, 10, len(df))
    df['quantity'] = df['quantity'].clip(lower=0).round()
    
    # Add price with some variation
    base_price = 100
    df['price'] = base_price + np.random.normal(0, 5, len(df))
    df['price'] = df['price'].clip(lower=base_price*0.8).round(2)
    
    # Add promotional periods
    df['promotion'] = 0
    # Regular promotions every 2 weeks
    promo_mask = (df.index.isocalendar().week % 2 == 0)
    df.loc[promo_mask, 'promotion'] = 1
    
    # Add special events/holidays
    df['special_event'] = 0
    # Mark some major shopping days
    holidays = [
        '2022-11-25', '2022-12-25', '2023-11-25', '2023-12-25'  # Black Friday and Christmas
    ]
    df.loc[holidays, 'special_event'] = 1
    
    # Add weather features (synthetic)
    df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df.index.dayofyear / 365) + np.random.normal(0, 2, len(df))
    df['is_rainy'] = np.random.binomial(1, 0.3, len(df))
    
    # Add competitor price (slightly higher than our price)
    df['competitor_price'] = df['price'] * (1 + np.random.uniform(0.1, 0.2, len(df)))
    df['competitor_price'] = df['competitor_price'].round(2)
    
    # Add inventory levels
    df['inventory'] = 1000 + np.random.normal(0, 100, len(df))
    df['inventory'] = df['inventory'].clip(lower=0).round()
    
    # Add marketing spend
    df['marketing_spend'] = 1000 + 500 * df['promotion'] + np.random.normal(0, 100, len(df))
    df['marketing_spend'] = df['marketing_spend'].clip(lower=0).round()
    
    # Calculate total revenue
    df['revenue'] = df['quantity'] * df['price']
    
    return df

def run_algorithm(algorithm, X_train, X_test, y_train, y_test):
    model = algorithm()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def parallel_run():
    df = generate_sales_data()
    X = df.drop(columns=['quantity'])  # Assuming 'quantity' is the target
    y = df['quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    algorithms = [LinearRegression, RandomForestRegressor, SVR]
    results = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_algorithm, algo, X_train, X_test, y_train, y_test): algo.__name__ for algo in algorithms}
        for future in futures:
            algo_name = futures[future]
            mse = future.result()
            results[algo_name] = mse

    return results

# Example usage
# results = parallel_run()
# print(results)
