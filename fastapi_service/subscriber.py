from google.cloud import pubsub_v1
import json
import requests
from mlops_project.ml_models.parallel_execution import generate_sales_data
from google.cloud import bigquery
from config.config_manager import load_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def callback(message):
    logging.info(f"Received message: {message.data}")
    data = json.loads(message.data)
    print(data)
    process_data(data)
    message.ack()

def process_data(data):
    logging.info("Starting data processing.")
    try:
        df = generate_sales_data()
        logging.info("Generated DataFrame:")
        logging.info(df.head())
        config = load_config()
        cloud_run_url = f"https://{config['cloud_run']['service_name']}-{config['cloud_run']['region']}.a.run.app/process"
        response = requests.post(cloud_run_url, json=data)
        if response.status_code == 200:
            logging.info("Data processed successfully in Cloud Run.")
            output_data = response.json()
            push_to_bigquery(output_data)
        else:
            logging.error(f"Failed to process data with Cloud Run. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")

def push_to_bigquery(output_data):
    logging.info("Pushing data to BigQuery.")
    try:
        client = bigquery.Client()
        dataset_id = 'mlops_project'
        table_id = 'prediction_results'
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)

        # Convert output_data to a list of dictionaries if it's not already
        if isinstance(output_data, dict):
            output_data = [output_data]

        errors = client.insert_rows_json(table, output_data)
        if errors == []:
            logging.info("New rows have been added to BigQuery.")
        else:
            logging.error(f"Encountered errors while inserting rows: {errors}")
    except Exception as e:
        logging.error(f"An error occurred while pushing data to BigQuery: {e}")

def subscribe_to_pubsub():
    logging.info("Subscribing to Pub/Sub.")
    try:
        # Load configuration
        config = load_config()

        project_id = config['gcp']['project_id']
        subscription_id = config['pubsub']['subscription_name']
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project_id, subscription_id)
        streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
        logging.info(f"Listening for messages on {subscription_path}...")

        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
    except Exception as e:
        logging.error(f"An error occurred while subscribing to Pub/Sub: {e}")

if __name__ == "__main__":
    subscribe_to_pubsub()
