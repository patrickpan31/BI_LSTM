import pandas as pd
import yaml

from model_trainning.main import *


def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Parse the YAML file safely
    return config


# Function to get the current row count of a file
def get_file_row_count(file_path):
    df = pd.read_csv(file_path)
    return len(df)


# Function to update the row count in the tracking file
def update_initial_row_count(config, new_count):
    config['retrain']['current_count'] = new_count

    # Save the updated YAML file
    with open("config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Updated initial row count to {new_count}")


# Function to retrain the model
def trigger_retrain_model(config):
    train_model(config)
    print("Model retrained successfully")


# Monitor the file and check for changes in the row count
def monitor_file(config):
    threshold, file_path = config['retrain']['threshold'], config['data']['path3']
    initial_row_count = config['retrain']['current_count']

    # Print current state for debugging
    print(f"Initial row count: {initial_row_count}")
    current_row_count = get_file_row_count(file_path)
    print(f"Current row count: {current_row_count}")

    row_difference = current_row_count - initial_row_count
    print(f"Row difference: {row_difference}, Threshold: {threshold}")

    if row_difference >= threshold:
        print('Model retrain triggered')
        trigger_retrain_model(config)

        # Update the initial row count after retraining
        update_initial_row_count(config, current_row_count)
    else:
        print("Threshold not met, no retraining")


config = load_config()
monitor_file(config)

