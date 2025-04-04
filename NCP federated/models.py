import tensorflow as tf
from ncps import wirings
from ncps.tf import CfC


# Experimental parameters
batch_size = 8
learning_rate = 0.0005
num_clients = 4
num_epochs = 50
num_rounds = 5
timesteps = 24
features = 5
train_percentage = 0.65


# Long Short-Term Memory (LSTM)
def lstm():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, features)),
            tf.keras.layers.LSTM(units=10, return_sequences=False),
            tf.keras.layers.Dense(32, 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(features, 'linear')
        ],
    )
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error', metrics=['mse']
    )
    name = 'lstm'
    return model, name


# Neural Circuit Policy (NCP)
def ncp():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(timesteps, features)),
            CfC(wirings.AutoNCP(10, features), return_sequences=False),
            tf.keras.layers.Dense(32, 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, 'relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(features, 'linear')
        ]
    )
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error', metrics=['mse']
    )
    name = 'ncp'
    return model, name


# Function to get the chosen model
def chosen_model():
    return ncp()