import streamlit as st
import pandas as pd
import numpy as np
from config.config_manager import load_config
from io import BytesIO
from google.cloud import pubsub_v1
import json

def main():
    # Load configuration
    config = load_config()

    st.title("MLOps Engine")
    st.write("Welcome to the MLOps Engine. Please provide your modeling inputs below.")

    # Load and display synthetic data
    df = generate_synthetic_data()
    st.write("Synthetic Data:")
    st.dataframe(df)

    # Feature selection
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select Features", df.columns.tolist())

    # Algorithm choice
    st.sidebar.header("Algorithm Choice")
    algorithms = st.sidebar.multiselect("Select Algorithms", ["XGBoost", "Linear Regression", "Lasso", "Random Forest", "Extra Trees Regressor"])

    # Evaluation metric
    st.sidebar.header("Evaluation Metric")
    metric = st.sidebar.selectbox("Select Evaluation Metric", ["RMSE", "MAE", "R2"])

    # Minimum data points filter
    min_data_points = st.sidebar.number_input("Minimum Data Points", min_value=1, value=50)

    if st.button("Submit"):
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(config['gcp']['project_id'], config['pubsub']['topic_name'])
        
        results = []
        for algorithm in algorithms:
            # Filter products based on minimum data points
            modelable_products = df.groupby('product').filter(lambda x: len(x) >= min_data_points)
            for product in modelable_products['product'].unique():
                product_data = modelable_products[modelable_products['product'] == product]
                X = product_data[features]
                y = product_data['quantity']  # Assuming 'revenue' is the target variable

                # Prepare data to send to Pub/Sub
                input_data = {
                    "features": features,
                    "X": X.to_dict(orient='records'),
                    "y": y.tolist(),
                    "algorithm": algorithm,
                    "metric": metric
                }
                data = json.dumps(input_data).encode("utf-8")
                future = publisher.publish(topic_path, data)
                future.result()  # Wait for the publish call to complete
                st.write(f"Inputs for {algorithm} on {product} submitted successfully to Pub/Sub!")
        
        if results:
            processed_data = process_output_data(results, metric)
            st.write(f"Processed Data: {processed_data}")  # Debugging statement
            if not processed_data.empty:
                st.write("Processed Data:")
                st.dataframe(processed_data)
            else:
                st.write("No processed data available. Please check the input parameters and try again.")

            # Generate downloadable links for predictions, feature importances, and coefficients
            for index, product_data in enumerate(results):
                if isinstance(product_data, dict):
                    predictions_df = pd.DataFrame(product_data.get('predictions', []))
                    feature_importance_df = pd.DataFrame(product_data.get('feature_importance', []))
                    
                    # Debugging statements to check the data
                    st.write(f"Predictions DataFrame for {index}:")
                    st.dataframe(predictions_df)
                    st.write(f"Feature Importance DataFrame for {index}:")
                    st.dataframe(feature_importance_df)
                    
                    st.write(f"Downloadable files:")
                    st.download_button(
                        label=f"Download Predictions",
                        data=convert_df_to_excel(predictions_df),
                        file_name=f'predictions_{product}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f"download_predictions_{index}"
                    )
                    st.download_button(
                        label=f"Download Feature Importances",
                        data=convert_df_to_excel(feature_importance_df),
                        file_name=f'feature_importance_{product}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f"download_feature_importance_{index}"
                    )

def generate_synthetic_data():
    # Generate synthetic data from 2020 to the current date in 2024
    dates = pd.date_range(start='2020-01-01', end=pd.Timestamp.now(), freq='D')
    df = pd.DataFrame(index=dates)
    df['date'] = df.index
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quantity'] = 100 + 20 * np.sin(2 * np.pi * df.index.dayofyear / 365) + np.random.normal(0, 10, len(df))
    df['price'] = 100 + np.random.normal(0, 5, len(df))
    df['promotion'] = (df.index.isocalendar().week % 2 == 0).astype(int)
    df['special_event'] = 0
    holidays = ['2020-11-25', '2020-12-25', '2021-11-25', '2021-12-25', '2022-11-25', '2022-12-25', '2023-11-25', '2023-12-25']
    df.loc[holidays, 'special_event'] = 1
    df['temperature'] = 20 + 10 * np.sin(2 * np.pi * df.index.dayofyear / 365) + np.random.normal(0, 2, len(df))
    df['is_rainy'] = np.random.binomial(1, 0.3, len(df))
    df['competitor_price'] = df['price'] * (1 + np.random.uniform(0.1, 0.2, len(df)))
    df['inventory'] = 1000 + np.random.normal(0, 100, len(df))
    df['marketing_spend'] = 1000 + 500 * df['promotion'] + np.random.normal(0, 100, len(df))
    df['revenue'] = df['quantity'] * df['price']
    df['product'] = np.random.choice(['Product A', 'Product B', 'Product C'], size=len(df))
    return df

def process_output_data(output_data, metric):
    # Process the output data to select the best algorithm for each product
    results = []
    for product_predictions in output_data:
        if isinstance(product_predictions, dict):
            # Directly append the product_predictions to results
            results.append(product_predictions)
    return pd.DataFrame(results)

def convert_df_to_excel(data):
    output = BytesIO()
    if isinstance(data, dict):
        data = [data]  # Convert to list of dicts if it's a single dict
    df = pd.DataFrame(data)
    df.to_excel(output, index=False)
    processed_data = output.getvalue()
    return processed_data

if __name__ == "__main__":
    main()
