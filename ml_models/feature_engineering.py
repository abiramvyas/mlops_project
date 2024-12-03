import pandas as pd

def load_data(file_path):
    # Load data from a CSV file or any other source
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example feature engineering steps
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop columns that are not needed
    data = data.drop(['date'], axis=1)
    
    # Handle missing values
    data = data.fillna(data.mean())
    
    return data

def feature_engineering(file_path):
    processed_data = preprocess_data(data)
    return processed_data
