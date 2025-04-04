import pandas as pd
import numpy as np
from models import *
from sklearn.preprocessing import MinMaxScaler


# Function to get the chosen model
def load_model():
    return chosen_model()


# Function to get model parameters
def get_model_parameters(model):
    return model.get_weights()


# Function to set model parameters
def set_model_params(model, params):
    model.set_weights(params)


# Function to get client data
def getClientData(client_name, data_path="datasets"):
    client_df = pd.read_excel(data_path + "/" + client_name + '.xlsx', engine='openpyxl')
    client_df.rename(columns={client_df.columns[0]: 'Date'}, inplace=True)
    client_df['Date'] = pd.to_datetime(client_df['Date'])
    client_df.set_index('Date', inplace=True)
    return client_df


# Function to get train and test data
def getTrainAndTestData(client_df):
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(client_df)  # TODO: WARNING data leakage -> fit_transform train, transform test

    sequences = []
    labels = []
    for i in range(len(scaled_df) - timesteps):
        sequences.append(scaled_df[i:i + timesteps, :])
        labels.append(scaled_df[i + timesteps, :])
    
    X = np.array(sequences)
    y = np.array(labels)

    train_size = int(len(X) * train_percentage)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler

# Function to get inverse data
def inverseData(scaler, y_pred, y_test):
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test)
    return y_pred_inverse, y_test_inverse