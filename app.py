from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import numpy as np
import tensorflow as tf
from model_trainning.main import *
# from model_trainning.data_augmentation import *
from io import BytesIO, StringIO
import csv
import yaml
import time
from keras.callbacks import Callback
from threading import Thread
import redis
import uuid

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessary for flashing messages

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Redis client setup
redis_host = os.getenv('REDIS_HOST', 'localhost')  # Default to localhost if not set
r = redis.Redis(host=redis_host, port=6379, db=0) # Default to localhost if not set


def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
model = tf.keras.models.load_model(config['model']['model_name'], custom_objects={'AttentionLayer': AttentionLayer})

# Helper functions to set and get progress in Redis

def get_progress():
    progress = r.get('training_progress') or 0
    status = r.get('training_status') or 'Not started'
    return {
        'progress': int(progress),
        'status': status.decode('utf-8')
    }

# Custom Keras callback for progress tracking
class TrainingProgressCallback(Callback):
    def __init__(self, total_epochs, redi, progress_key, status_key):
        super(TrainingProgressCallback, self).__init__()
        self.total_epochs = total_epochs
        self.r = redi
        self.progress_key = progress_key
        self.status_key = status_key

    def on_train_begin(self, logs=None):
        self.r.set(self.progress_key, 0)
        self.r.set(self.status_key, 'Training started')

    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.total_epochs * 100)
        self.r.set(self.progress_key, progress)
        self.r.set(self.status_key, f'Training progress: {progress}%')

    def on_train_end(self, logs=None):
        self.r.set(self.progress_key, 100)
        self.r.set(self.status_key, 'Training completed')

def train_model_thread(config, callbacks):
    # redis_host = os.getenv('REDIS_HOST', 'localhost')
    # thread_redis_client = redis.Redis(host=redis_host, port=6379, db=0)
    # try:
    #     thread_redis_client.ping()
    #     print('redis works works works')# Test if Redis is available
    # except redis.ConnectionError:
    #     print("Redis server is not available. Please check your Redis server settings.")
    # for callback in callbacks:
    #     if isinstance(callback, TrainingProgressCallback):
    #         callback.r = thread_redis_client
    train_model(config, progress_callback=callbacks)


@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/training_progress')
def get_training_progress():
    client_id = request.args.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID is required"}), 400

    # Retrieve client-specific progress and status from Redis
    progress_key = f'training_progress_{client_id}'
    status_key = f'training_status_{client_id}'
    progress = int(r.get(progress_key) or 0)
    status = r.get(status_key).decode('utf-8') if r.get(status_key) else 'Not started'

    return jsonify({
        'progress': progress,
        'status': status
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'retrain_model' in request.form:
        print('yes')
        client_id = str(uuid.uuid4())

        # Set Redis keys for client-specific progress and status
        progress_key = f'training_progress_{client_id}'
        status_key = f'training_status_{client_id}'

        # Initialize training progress in Redis
        r.set(progress_key, 0)
        r.set(status_key, 'Not started')


        try:
            r.ping()
            print('redis works works works')  # Test if Redis is available
        except redis.ConnectionError:
            print("Redis server is not available. Please check your Redis server settings.")

        new_config = load_config()
        config['model']['epoch'] = new_config['model']['epoch']

        results_str = request.form.get('modified_results')
        results = json.loads(results_str)

        # Save predictions to CSV
        file_exists = os.path.isfile(config['data']['path3'])
        mode = 'a' if file_exists else 'w'
        with open(config['data']['path3'], mode, newline='') as f:
            file_writer = csv.writer(f, lineterminator='\n')
            if not file_exists:
                file_writer.writerow(['Input', 'Prediction'])
            for input_value, prediction in results.items():
                file_writer.writerow([input_value, prediction])

        # Start training in a separate thread
        # progress_callback = TrainingProgressCallback(total_epochs=config['model']['epoch'], redi= r)
        progress_callback = TrainingProgressCallback(
            total_epochs=config['model']['epoch'],
            redi=r,
            progress_key=progress_key,
            status_key=status_key
        )
        thread = Thread(target=train_model_thread, args=(config, [progress_callback]))
        thread.start()

        return render_template('training.html', client_id=client_id)


    if "download_csv" in request.form:
        # Get the existing results from the hidden field
        results_str = request.form.get('modified_results')
        print(results_str)# This gets the results as a string
        results = eval(results_str)

        # Create a CSV file in memory using BytesIO (binary stream)
        output = StringIO()
        writer = csv.writer(output)

        file_exists = os.path.isfile(config['data']['path3'])
        mode = 'a' if file_exists else 'w'
        with open(config['data']['path3'], mode, newline='') as f:
            file_writer = csv.writer(f, lineterminator='\n')
            writer = csv.writer(output)

            # Write header for server response (always needed)
            writer.writerow(['Input', 'Prediction'])

            # Write header for local file only if it doesn't exist
            if not file_exists:
                file_writer.writerow(['Input', 'Prediction'])

            # Write rows for both in-memory CSV and local file
            for input_value, prediction in results.items():
                writer.writerow([input_value, prediction])
                file_writer.writerow([input_value, prediction])



        # Get the CSV content as a string
        csv_data = output.getvalue()

        # Convert the CSV text to bytes, since Flask's send_file expects binary data
        byte_output = BytesIO()
        byte_output.write(csv_data.encode('utf-8'))  # Encode the text to binary
        byte_output.seek(0)  # Move the pointer to the beginning of the file


            # Serve the CSV file as a downloadable attachment using BytesIO
        return send_file(byte_output, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')






    file = request.files.get('file')

    # Get the manual input (if any)
    manual_input = request.form.get('manual_input')

    if file:
        file = request.files['file']

        # Read the CSV file into a DataFrame
        try:
            data = pd.read_csv(file, header=None)
            data = data.iloc[1:]# Assuming no header in the CSV
        except Exception as e:
            return jsonify({'error': f'Failed to read CSV file: {str(e)}'}), 400

        # Ensure the data is reshaped to (m, 1), where m is the number of rows in the CSV
        input_data = data.iloc[:, 0]
        # input_data = data.values.reshape(-1, 1)
        word_index, index_word, word_vector_map, word_vector = load_dicts_from_json(config['model']['word_dict_name'])
        index_output, output_index, type_map = load_dicts_from_json(config['model']['output_dict'])
        # Reshape the CSV data to (m, 1)
        input_data = np.array(input_data)
        print(input_data.shape)
        play_indices = sentences_to_indices(input_data, word_index, 40)

    # If no file, check if JSON data is sent (for single or multiple string inputs)
    elif manual_input or request.is_json:
        try:
            # Extract the string input
            if not manual_input:
                data = request.get_json()
                input_string = data['input']
            else:
                input_string = manual_input
            # Assuming the JSON is like {"input": "Apple, banana"}

            # Split the string by commas to get individual items
            input_list = [item.strip() for item in input_string.split(',')]  # Remove any extra spaces

            # Convert the list of strings to a NumPy array of shape (m, 1)
            input_data = np.array(input_list).reshape(-1, )  # (m, 1), where m = len(input_list)


            print(input_data)

            word_index, index_word, word_vector_map, word_vector = load_dicts_from_json(config['model']['word_dict_name'])
            index_output, output_index, type_map = load_dicts_from_json(config['model']['output_dict'])
            # Reshape the CSV data to (m, 1)
            input_data = np.array(input_data)
            play_indices = sentences_to_indices(input_data, word_index, 40)

        except KeyError:
            return jsonify({'error': 'Invalid JSON format, expected {"input": "your_values"}'}), 400
    else:
        return jsonify({'error': 'No valid input found. Please send a string or upload a CSV file.'}), 400

    # Ensure the input data is in the correct shape for LSTM (m, 1)
    try:
        model = tf.keras.models.load_model(config['model']['model_name'], custom_objects={'AttentionLayer': AttentionLayer})

        # predictions = np.argmax(model.predict(play_indices), axis = 1)

        predictions = model.predict(play_indices)

        threshold = 0.92
        thresholded_output = tf.where(predictions < threshold, 0.0, predictions)
        results = []
        for sample in thresholded_output:
            if tf.reduce_any(sample > 0):
                # Step 4: If there are values greater than 0, find the index of the max value
                max_index = tf.argmax(sample).numpy()  # Get the index of the max value
            else:
                # Step 5: If all values are zero, return 0
                max_index = 0
            results.append(max_index)




        predictions = [index_output[idx] for idx in results]
    except Exception as e:
        return render_template('result.html', error=f'Prediction failed: {str(e)}'), 500

    # Prepare the data to send to the template
    results = {str(input_data[i]): predictions[i] for i in range(len(predictions))}

    # Render the result.html template with the predictions
    return render_template('result.html', results=results)




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
