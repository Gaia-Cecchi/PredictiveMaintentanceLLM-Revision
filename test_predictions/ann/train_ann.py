import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime

# Impostazioni per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)

def load_data(db_path):
    """Carica i dati dal database e li prepara per l'addestramento"""
    conn = sqlite3.connect(db_path)
    
    # Carica i dati con etichette
    query = """
    SELECT c.DateTime, c.Temperature, c.Vibration, c.Pressure, c.Current,
           c.Speed, c.Voltage, c.CosPhi, c.weather_temperature, c.weather_humidity,
           c.weather_wind_speed, c.weather_precipitation,
           CASE WHEN c.Anomaly = 1 THEN 1 ELSE 0 END as is_anomaly
    FROM compressor_data c
    ORDER BY c.DateTime
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Converti DateTime in formato datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Imposta DateTime come indice
    df.set_index('DateTime', inplace=True)
    
    return df

def prepare_data_for_ann(df, window_size=24):
    """Prepara i dati per l'ANN utilizzando statistiche su finestre temporali"""
    feature_columns = [col for col in df.columns if col != 'is_anomaly']
    scaler = StandardScaler()
    
    # Crea feature ingegnerizzate su finestre
    X_list = []
    y_list = []
    
    # Utilizziamo un approccio a finestra mobile con statistiche
    for i in range(len(df) - window_size + 1):
        window = df[feature_columns].iloc[i:i+window_size]
        
        # Calcola statistiche sulla finestra
        window_mean = window.mean().values
        window_std = window.std().values
        window_min = window.min().values
        window_max = window.max().values
        window_last = window.iloc[-1].values
        
        # Unisci tutte le statistiche
        window_features = np.concatenate([
            window_mean, window_std, window_min, window_max, window_last
        ])
        
        X_list.append(window_features)
        y_list.append(df['is_anomaly'].iloc[i+window_size-1])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Standardizza i dati
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_columns

def create_ann_model(input_shape):
    """Crea il modello ANN"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def main():
    print("=== Training ANN Model for Predictive Maintenance ===")
    
    # Imposta cartella per i modelli
    models_dir = os.path.join("test_predictions", "ann", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Imposta percorso del database
    db_path = os.path.join("test_predictions", "datasets", "compressor_data_2024_etichettato.db")
    
    # Carica i dati
    print("Loading data from database...")
    df = load_data(db_path)
    print(f"Loaded {len(df)} records with {df['is_anomaly'].sum()} anomalies")
    
    # Visualizza la distribuzione delle etichette
    class_distribution = df['is_anomaly'].value_counts()
    print("Class distribution:")
    print(f"Normal samples: {class_distribution.get(0, 0)}")
    print(f"Anomaly samples: {class_distribution.get(1, 0)}")
    
    # Chiedi la dimensione della finestra per le feature
    try:
        window_size = int(input("Enter window size for feature engineering (default: 24 hours): ") or "24")
    except ValueError:
        window_size = 24
        print("Invalid input, using default: 24 hours")
    
    # Prepara i dati
    print(f"Preparing data with window size {window_size}...")
    X, y, scaler, feature_columns = prepare_data_for_ann(df, window_size)
    
    # Ottieni forma dell'input
    input_shape = X.shape[1]
    print(f"Input shape: {input_shape}")
    
    # Dividi in train/validation/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Crea il modello
    print("Creating ANN model...")
    model = create_ann_model(input_shape)
    model.summary()
    
    # Callback per early stopping e model checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_dir, f'ann_model_{timestamp}.h5'),
        monitor='val_loss',
        save_best_only=True
    )
    
    # Impostazione di class weight per bilanciare le classi
    class_weight = {0: 1., 1: len(y_train) / max(1, sum(y_train))}
    print(f"Class weights: {class_weight}")
    
    # Addestramento del modello
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight
    )
    
    # Valutazione su test set
    print("Evaluating model...")
    test_results = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    print(f"Test precision: {test_results[2]:.4f}")
    print(f"Test recall: {test_results[3]:.4f}")
    
    # Salva il modello e lo scaler
    model_path = os.path.join(models_dir, f'ann_model_final_{timestamp}.h5')
    scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
    config_path = os.path.join(models_dir, f'config_{timestamp}.json')
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    # Salva la configurazione
    config = {
        'window_size': window_size,
        'feature_columns': feature_columns,
        'scaler_path': os.path.basename(scaler_path),
        'model_path': os.path.basename(model_path)
    }
    
    # Salva la configurazione come JSON
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Plot della loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot dell'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Salva la figura
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, f'training_history_{timestamp}.png'))
    plt.close()
    
    print(f"Model training completed and saved to {model_path}")
    print(f"Configuration saved to {config_path}")
    print(f"Use this timestamp for prediction: {timestamp}")

if __name__ == "__main__":
    main()
