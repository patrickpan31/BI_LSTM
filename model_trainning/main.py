import os
import numpy as np
import pandas as pd
import re
import mlflow
import tensorflow
import json
import yaml
import wandb
from keras.callbacks import Callback
# from model_trainning.data_augmentation import *
import random

#DATA AUGMENTATION PART:

def random_delete(drug_name, num = 1):
    """Randomly delete a character."""
    if len(drug_name) <= 1:
        return drug_name
    idx = random.randint(0, len(drug_name) - 1)
    return drug_name[:idx] + drug_name[idx+num:]

def random_substitution(drug_name):
    """Introduce random noise by substituting a character."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    idx = random.randint(0, len(drug_name) - 1)
    return drug_name[:idx] + random.choice(alphabet) + drug_name[idx+1:]

def random_insertion(drug_name):
    """Insert a random character."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    idx = random.randint(0, len(drug_name))
    return drug_name[:idx] + random.choice(alphabet) + drug_name[idx:]

def augment_drug_name(drug_name):
    """Apply a random augmentation."""
    augmentations = [random_delete, random_insertion]
    augmentation_func = random.choices(augmentations, weights=[0.9, 0.1], k = 1)[0]# Randomly choose an augmentation
    # augmentation_func = random_delete

    if augmentation_func == random_delete:
        # if len(drug_name)<8:
        #     dele_num = 1
        # else:
        #     dele_num = max(random.randint(1, len(drug_name)//2)//2, 1)
        potention_num = [1,2,3]
        dele_num = random.choices(potention_num, weights=[0.6, 0.3, 0.1], k = 1)[0]
        return augmentation_func(drug_name, dele_num)
    else:
        return augmentation_func(drug_name)

# Function to apply augmentation to the entire dataset
def apply_augmentation_to_dataset(df, column_name, augment_ratio=0.6, ep = 5):
    """
    Apply augmentations to a portion of the dataset.

    Args:
    df: DataFrame containing the dataset.
    column_name: The column containing drug names to augment.
    augment_ratio: The fraction of the dataset to augment. Default is 0.5 (50% of the dataset).

    Returns:
    A new DataFrame with original and augmented data combined.
    """
    for i in range(ep):
        augmented_data = []

        # Iterate over the dataset
        for _, row in df.iterrows():
            original_name = row[column_name]
            # Store the original row
            augmented_data.append(row)

            # Apply augmentation based on the specified ratio
            if len(original_name) >= 16 and random.random() < augment_ratio:
                augmented_row = row.copy()
                if len(augmented_row[column_name].split(' ')) > 1:
                    first_part = original_name.split(' ')[0]
                    unchanged = ' '.join(original_name.split(' ')[1:])
                    augmented_row[column_name] = augment_drug_name(first_part) + ' ' + unchanged
                else:
                    augmented_row[column_name] = augment_drug_name(original_name)
                augmented_data.append(augmented_row)

        # Return a new DataFrame with both original and augmented data
        augmented_df = pd.DataFrame(augmented_data).reset_index(drop = True)
        df = augmented_df
    return augmented_df

#####################################################

class AttentionLayer(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        e = tensorflow.keras.backend.tanh(tensorflow.keras.backend.dot(inputs, self.W) + self.b)
        e = tensorflow.keras.backend.squeeze(e, axis=-1)
        alpha = tensorflow.keras.backend.softmax(e)
        alpha = tensorflow.keras.backend.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = inputs * alpha
        context = tensorflow.keras.backend.sum(context, axis=1)
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config



def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Parse the YAML file safely
    return config


def load_data(path1, path2, path3=None):
    # Load the Excel file from path1
    df = pd.read_excel(path1, sheet_name=None)
    combined_df = pd.concat(df.values(), ignore_index=True)

    # Select relevant columns and remove duplicates
    target_columns = combined_df[['DRUG_NAME', 'ART', 'TYPE']]
    df_no_duplicates = target_columns.drop_duplicates()

    # Load the CSV from path2
    new_art = pd.read_csv(path2)
    # new_art = new_art.iloc[:, :2]
    new_art.columns = ['DRUG_NAME', 'ART', 'TYPE']

    # Concatenate the two dataframes
    result = pd.concat([df_no_duplicates, new_art]).reset_index(drop=True)

    # Check if path3 is provided
    if path3 is not None:
        # Load the CSV from path3
        additional_df = pd.read_csv(path3)
        print(additional_df)
        additional_df = additional_df.iloc[:,:2]
        additional_df.columns = ['DRUG_NAME', 'ART']
        additional_df['TYPE'] = None
        print('path3_file:')
        print(additional_df)

        # Concatenate the additional data
        result = pd.concat([result, additional_df]).reset_index(drop=True)

    return result


def output_transform(df):

    output = df['ART'].str.upper().explode().unique().tolist()
    output_index = {word:index+1 for index, word in enumerate(output)}
    index_output = {index+1:word for index, word in enumerate(output)}
    output_index['NOT_FOUND'] = 0
    index_output[0] = 'NOT_FOUND'
    df['ART'] = df['ART'].map(output_index)
    y = df['ART'].values
    print(y)
    return y, len(output_index), index_output, output_index


def type_mapping(df):

    newdf = df[['ART', 'TYPE']]
    newdf = newdf.drop_duplicates().reset_index(drop = True)

    type_map = {row.ART: row.TYPE for row in newdf.itertuples(index=False)}
    type_map['JULUCA'] = 'FDC2'
    type_map['DOVATO'] = 'FDC2'

    return type_map

def split_data(x,y):

    test_size = int(0 * len(x))

    # Shuffle the data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    X = x[indices]
    y = y[indices]

    X_train = X[test_size:]
    y_train = y[test_size:]
    X_test = X[:test_size]
    y_test = y[:test_size]

    return X_train, y_train, X_test, y_test


def word_dic_generator(df):

    unique_word = 'ABCDEFGHIJKLNMOPQRSTUVWSYZ0123456789+-/ '

    word_index = {word: index+1 for index, word in enumerate(unique_word)}
    index_word = {index+1: word for index, word in enumerate(unique_word)}

    word_vector = np.zeros((len(unique_word)+1, len(unique_word)))
    word_vector_map = {}

    for index, word in enumerate(unique_word):
        word_vector[index+1,index] = 1
        word_vector_map[index+1] = word_vector[index+1]


    return word_index, index_word, word_vector_map, word_vector


def save_word_dicts_as_json(word_index, index_word, word_vector_map, word_vector, filename):

    word_vector = word_vector.tolist()
    word_vector_map = {str(key): value.tolist() for key, value in word_vector_map.items()}

    data = {
        'word_index': word_index,
        'index_word': index_word,
        'word_vector_map': word_vector_map,
        'word_vector': word_vector
    }

    with open(filename, 'w') as file:
        json.dump(data, file)


def save_output_word_dicts_as_json(index_output, output_index, type_map, filename):
    data = {
        'index_output': index_output,
        'output_index': output_index,
        'type_map': type_map
    }

    with open(filename, 'w') as file:
        json.dump(data, file)


def load_dicts_from_json(filename):

    with open(filename, 'r') as file:
        data = json.load(file)

    if 'word_index' in data:

        index_word = {int(key): value for key, value in data['index_word'].items()}
        word_vector_map = {int(key): np.array(value) for key, value in data['word_vector_map'].items()}
        word_vector = np.array(data['word_vector'])

        return data['word_index'], index_word, word_vector_map, word_vector

    else:

        index_output = {int(key): value for key, value in data['index_output'].items()}
        return index_output, data['output_index'], data['type_map']


def sentences_to_indices(df, word_index, max_len, letter = True):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_index -- a dictionary containing each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    if letter:
        if isinstance(df, pd.DataFrame):
            m = df['DRUG_NAME'].shape[0]  # number of training examples
            x_indices = np.zeros((m, max_len), dtype=int)
            print(m)

            for i, row in df.iterrows():
                words = row['DRUG_NAME'][:max_len].upper()
                # words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
                x_indices[i, :len(words)] = [word_index.get(w, 0) for w in words]
            print(x_indices)
            np.savetxt('array_2d.txt', x_indices, fmt='%d', delimiter=',')

        else:
            m = df.shape[0]
            x_indices = np.zeros((m, max_len), dtype=int)
            for index, value in enumerate(df):
                words = value[:max_len].upper()
                # words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
                x_indices[index, :len(words)] = [word_index.get(w, 0) for w in words]

        return x_indices

    else:
        if isinstance(df, pd.DataFrame):
            m = df['DRUG_NAME'].shape[0]  # number of training examples
            x_indices = np.zeros((m, max_len), dtype=int)
            print(m)

            for i, row in df.iterrows():
                words = re.sub(r'[-,/]+', ' ', row['DRUG_NAME'].upper())
                words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
                x_indices[i, :len(words)] = [word_index.get(w, 0) for w in words]
            print(x_indices)

        else:
            m = df.shape[0]
            x_indices = np.zeros((m, max_len), dtype=int)
            for index, value in enumerate(df):
                words = re.sub(r'[-,/]+', ' ', value.upper())
                words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
                x_indices[index, :len(words)] = [word_index.get(w, 0) for w in words]

        return x_indices


def pretrained_embedding_layer(word_vector_map, word_vector, embedding_weights_file = None):
    """
    Creates a Keras Embedding() layer

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_size = len(word_vector)
    # any_word = list(word_vector_map.keys())[0]
    emb_dim = word_vector_map[1].shape[0]

    print('check', vocab_size, emb_dim)

    # Define Keras embedding layer with the correct input and output sizes
    embedding_layer = tensorflow.keras.layers.Embedding(vocab_size, emb_dim, trainable=True)

    # embedding_layer = tensorflow.keras.layers.Embedding(input_dim=vocab_size,
    #           output_dim=emb_dim,
    #           embeddings_initializer='he_uniform',
    #           trainable=True,
    #           name='embedding_layer')

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))  # Do not modify the "None".  This line of code is complete as-is.

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([word_vector])

    if embedding_weights_file is not None:
        embedding_weights = np.load(embedding_weights_file)
        embedding_layer.build((None,))  # Build the embedding layer
        embedding_layer.set_weights([embedding_weights])

    return embedding_layer


def Emojify_V2(input_shape, word_vector_map, word_vector, output_shape, embedding_weights_file = None):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = tensorflow.keras.Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_vector_map, word_vector, embedding_weights_file)

    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences, So, set return_sequences = True
    # If return_sequences = False, the LSTM returns only tht last output in output sequence



    # X = tensorflow.keras.layers.LSTM(units=128, return_sequences=True)(embeddings)
    # # Add dropout with a probability of 0.5
    # X = tensorflow.keras.layers.Dropout(0.5)(X)
    # # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # # The returned output should be a single hidden state, not a batch of sequences.
    # X = tensorflow.keras.layers.LSTM(units=128, return_sequences=True)(X)
    # # Add dropout with a probability of 0.5
    # X = tensorflow.keras.layers.Dropout(0.2)(X)
    #
    # X = tensorflow.keras.layers.LSTM(units=64, return_sequences=False)(X)
    #
    # X = tensorflow.keras.layers.Dropout(0.3)(X)

    bidirectional_lstm = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(
        embeddings)

    attention = AttentionLayer()(bidirectional_lstm)
    dropout = tensorflow.keras.layers.Dropout(0.5)(attention)

    dense = tensorflow.keras.layers.Dense(128, activation='relu')(dropout)

    dropout_final = tensorflow.keras.layers.Dropout(0.5)(dense)

    # Propagate X through a Dense layer with 5 units
    X = tensorflow.keras.layers.Dense(output_shape)(dropout_final)
    # Add a softmax activation
    X = tensorflow.keras.layers.Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = tensorflow.keras.Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model

def train_model(config, progress_callback=None, tracking = None):

    # config = load_config()

    if tracking:

        run = wandb.init(job_type="train")

    path3 = config['data']['path3']
    file_exists = os.path.isfile(path3)
    print(file_exists)
    if file_exists:
        df = load_data(config['data']['path1'], config['data']['path2'], path3)
    else:
        df = load_data(config['data']['path1'], config['data']['path2'])
    type_map = type_mapping(df)
    word_index, index_word, word_vector_map, word_vector = word_dic_generator(df)
    pretrained_embedding_layer(word_vector_map, word_vector)
    print('checking_length')
    print(len(df))
    df = apply_augmentation_to_dataset(df, 'DRUG_NAME', augment_ratio=0.3, ep = 8)
    print(len(df))
    print(df)
    features = sentences_to_indices(df, word_index, config['model']['maxlen'])
    labels, length, index_output, output_index = output_transform(df)
    print(len(features), len(labels))

    X_train, y_train, X_test, y_test = split_data(features, labels)

    Y_train_oh = tensorflow.one_hot(y_train, depth=length)

    embedding_exist = os.path.isfile(config['model']['embedding_weights'])
    if embedding_exist:
        embedding_weights_file = config['model']['embedding_weights']
    else:
        embedding_weights_file = None

    model = Emojify_V2((40,), word_vector_map, word_vector, length, embedding_weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train_oh, epochs=config['model']['epoch'], batch_size=config['model']['batch_size'], shuffle=True, callbacks=progress_callback )

    model.save(config['model']['model_name'])

    # Save the embedding layer's weights separately
    embedding_layer = model.get_layer('embedding_1')  # Assuming 'embedding' is the name of your embedding layer
    embedding_weights = embedding_layer.get_weights()
    np.save(config['model']['embedding_weights'], np.array(embedding_weights)) # Save embedding weights to a .npy file

    if tracking:
        artifact1 = wandb.Artifact(config['model']['wandb_model'], type='model')
        artifact1.add_file(config['model']['model_name'])
        run.log_artifact(artifact1)

    save_word_dicts_as_json(word_index, index_word, word_vector_map, word_vector, config['model']['word_dict_name'])

    if tracking:
        artifact2 = wandb.Artifact(config['model']['wanbd_word_dict'], type='word_dic')
        artifact2.add_file(config['model']['word_dict_name'])
        run.log_artifact(artifact2)

    save_output_word_dicts_as_json(index_output, output_index, type_map, config['model']['output_dict'])

    if tracking:
        artifact3 = wandb.Artifact(config['model']['wandb_output_dict'], type='output_dict')
        artifact3.add_file(config['model']['output_dict'])
        run.log_artifact(artifact3)

    return model, word_index, index_output


if __name__ == '__main__':

    config = load_config()
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    model, word_index, index_output = train_model(config)


####below is just a play section################
    play = np.array(["EFAVIR/TENOFOV/EMTRI"])
    play_indices = sentences_to_indices(play, word_index, config['model']['maxlen'])
    prediction = model.predict(play_indices)
    threshold = 0.9
    thresholded_output = tensorflow.where(prediction < threshold, 0.0, prediction)
    results = []
    for sample in thresholded_output:
        if tensorflow.reduce_any(sample > 0):
            # Step 4: If there are values greater than 0, find the index of the max value
            max_index = tensorflow.argmax(sample).numpy()  # Get the index of the max value
        else:
            # Step 5: If all values are zero, return 0
            max_index = 0
        results.append(max_index)
    print([index_output[idx] for idx in results])




