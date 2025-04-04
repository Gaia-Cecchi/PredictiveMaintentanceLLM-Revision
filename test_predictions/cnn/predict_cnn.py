import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib
import json
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_absolute_error, mean_squared_error, roc_curve, auc,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(db_path):
    """Carica i dati dal database"""
    conn = sqlite3.connect(db_path)
    
    # Carica i dati per la previsione (escluse le etichette)
    query_pred = """
    SELECT c.DateTime, c.Temperature, c.Vibration, c.Pressure, c.Current,
           c.Speed, c.Voltage, c.CosPhi, c.weather_temperature, c.weather_humidity,
           c.weather_wind_speed, c.weather_precipitation
    FROM compressor_data c
    ORDER BY c.DateTime
    """
    
    # Carica le etichette reali per la valutazione
    query_labels = """
    SELECT c.DateTime, 
           CASE WHEN c.Anomaly = 1 THEN 1 ELSE 0 END as actual_anomaly,
           CASE 
               WHEN c.Notes LIKE '%Overheating%' THEN 'overheating'
               WHEN c.Notes LIKE '%Bearing%' THEN 'bearing failure'
               WHEN c.Notes LIKE '%Pressure%' THEN 'pressure drop'
               WHEN c.Notes LIKE '%Motor%' THEN 'motor imbalance'
               WHEN c.Notes LIKE '%Voltage%' THEN 'voltage fluctuation'
               WHEN c.Notes LIKE '%False Positive%' THEN 'false_positive'
               WHEN c.Anomaly = 1 THEN 'anomaly'
               ELSE 'normal'
           END as actual_type
    FROM compressor_data c
    ORDER BY c.DateTime
    """
    
    df_pred = pd.read_sql_query(query_pred, conn)
    df_labels = pd.read_sql_query(query_labels, conn)
    conn.close()
    
    # Converti DateTime in formato datetime
    df_pred['DateTime'] = pd.to_datetime(df_pred['DateTime'])
    df_labels['DateTime'] = pd.to_datetime(df_labels['DateTime'])
    
    return df_pred, df_labels

def prepare_test_sequences(df, scaler, feature_columns, sequence_length):
    """Prepara sequenze di test per CNN"""
    # Imposta DateTime come indice
    df = df.set_index('DateTime')
    
    # Standardizza i dati
    features_scaled = scaler.transform(df[feature_columns])
    
    # Crea un DataFrame con i dati scalati
    df_scaled = pd.DataFrame(features_scaled, index=df.index, columns=feature_columns)
    
    X = []
    timestamps = []
    
    # Crea sequenze
    for i in range(len(df_scaled) - sequence_length + 1):
        X.append(df_scaled[feature_columns].values[i:i+sequence_length])
        timestamps.append(df.index[i+sequence_length-1])
    
    return np.array(X), timestamps

def main():
    print("=== CNN Model Prediction and Evaluation ===")
    
    # Chiedi il timestamp del modello addestrato
    timestamp = input("Enter the timestamp of the trained model (format: YYYYMMDD_HHMMSS): ")
    
    # Imposta cartelle
    models_dir = os.path.join("test_predictions", "cnn", "models")
    results_dir = os.path.join("test_predictions", "cnn", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Carica la configurazione
    try:
        config_path = os.path.join(models_dir, f'config_{timestamp}.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sequence_length = config['sequence_length']
        feature_columns = config['feature_columns']
        scaler_path = os.path.join(models_dir, config['scaler_path'])
        model_path = os.path.join(models_dir, config['model_path'])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration: {e}")
        print("Please make sure you've entered the correct timestamp.")
        return
    
    # Carica lo scaler e il modello
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    
    print(f"Loaded model from {model_path}")
    
    # Imposta percorso del database
    db_path = os.path.join("test_predictions", "datasets", "compressor_data_2024_etichettato.db")
    
    # Carica i dati
    print("Loading data from database...")
    df_pred, df_labels = load_data(db_path)
    print(f"Loaded {len(df_pred)} records for prediction")
    
    # Prepara le sequenze di test
    print("Preparing test sequences...")
    X_test, timestamps = prepare_test_sequences(df_pred, scaler, feature_columns, sequence_length)
    
    # Esegui le predizioni
    print("Making predictions...")
    y_pred_prob = model.predict(X_test)
    
    # Converte le probabilità in predizioni binarie (0/1)
    threshold = 0.5
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # Crea un DataFrame con i risultati
    results_df = pd.DataFrame({
        'datetime': timestamps,
        'prediction_probability': y_pred_prob.flatten(),
        'predicted_anomaly': y_pred.flatten()
    })
    
    # Unisci con le etichette reali
    results_df['datetime'] = pd.to_datetime(results_df['datetime'])
    df_labels['DateTime'] = pd.to_datetime(df_labels['DateTime'])
    results_df = pd.merge(results_df, df_labels, left_on='datetime', right_on='DateTime')
    
    # Aggiungi i dati originali
    # Resetta l'indice di df_pred per consentire il merge
    df_pred_reset = df_pred.copy()
    results_df = pd.merge(results_df, df_pred_reset, left_on='datetime', right_on='DateTime')
    results_df = results_df.drop(columns=['DateTime_x', 'DateTime_y'])
    
    # Aggiungi colonna is_correct
    results_df['is_correct'] = results_df['predicted_anomaly'] == results_df['actual_anomaly']
    
    # Aggiungi confidence sulla base della probabilità di previsione
    def get_confidence(prob):
        if prob < 0.2 or prob > 0.8:
            return 'high'
        elif prob < 0.35 or prob > 0.65:
            return 'medium'
        else:
            return 'low'
    
    results_df['confidence'] = results_df['prediction_probability'].apply(get_confidence)
    
    # Aggiungi il tipo previsto (solo per le anomalie previste)
    def get_predicted_type(row):
        if row['predicted_anomaly'] == 0:
            return ''
        
        # Semplice euristica per determinare il tipo di anomalia
        if row['Temperature'] > 115:
            return 'overheating'
        elif row['Vibration'] > 4.0 and row['Temperature'] > 100:
            return 'bearing failure'
        elif row['Pressure'] < 5.5:
            return 'pressure drop'
        elif row['Vibration'] > 3.0 and (row['Speed'] < 2860 or row['Speed'] > 3040):
            return 'motor imbalance'
        elif 'Voltage' in results_df.columns and (row['Voltage'] < 390 or row['Voltage'] > 410):
            return 'voltage fluctuation'
        else:
            return 'unknown anomaly'
    
    results_df['predicted_type'] = results_df.apply(get_predicted_type, axis=1)
    
    # Crea raccomandazioni
    def get_recommendation(row):
        if row['predicted_anomaly'] == 0:
            return 'Continue standard monitoring.'
        else:
            pred_type = row['predicted_type']
            if pred_type == 'overheating':
                return 'Check cooling system and reduce load immediately.'
            elif pred_type == 'bearing failure':
                return 'Inspect and replace bearings if necessary.'
            elif pred_type == 'pressure drop':
                return 'Check for leaks and inspect pressure regulation system.'
            elif pred_type == 'motor imbalance':
                return 'Inspect motor and check for imbalance or alignment issues.'
            elif pred_type == 'voltage fluctuation':
                return 'Investigate power supply and electrical connections.'
            else:
                return 'Investigate system for potential issues.'
    
    results_df['recommendation'] = results_df.apply(get_recommendation, axis=1)
    
    # Aggiungi predicted_classification e actual_classification per compatibilità con lo script di reporting
    results_df['predicted_classification'] = results_df['predicted_anomaly'].map({0: 'NORMAL VALUE', 1: 'ANOMALY'})
    results_df['actual_classification'] = results_df['actual_anomaly'].map({0: 'NORMAL VALUE', 1: 'ANOMALY'})
    
    # Aggiungi key_indicators
    def get_key_indicators(row):
        indicators = []
        indicators.append(f"Temperature: {row['Temperature']:.1f}°C")
        indicators.append(f"Vibration: {row['Vibration']:.2f} mm/s")
        indicators.append(f"Pressure: {row['Pressure']:.2f} bar")
        if 'Speed' in row and not pd.isna(row['Speed']):
            indicators.append(f"Speed: {row['Speed']:.0f} RPM")
        return ", ".join(indicators)
    
    results_df['key_indicators'] = results_df.apply(get_key_indicators, axis=1)
    
    # Calcola le metriche
    y_true = results_df['actual_anomaly'].values
    y_pred_final = results_df['predicted_anomaly'].values
    y_pred_proba = results_df['prediction_probability'].values
    
    accuracy = accuracy_score(y_true, y_pred_final)
    precision = precision_score(y_true, y_pred_final, zero_division=0)
    recall = recall_score(y_true, y_pred_final, zero_division=0)
    f1 = f1_score(y_true, y_pred_final, zero_division=0)
    mae = mean_absolute_error(y_true, y_pred_final)
    mse = mean_squared_error(y_true, y_pred_final)
    
    # Calcola la curva ROC e AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Stampa le metriche
    print("\n=== Performance Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Calcola la confusion matrix
    cm = confusion_matrix(y_true, y_pred_final)
    print("\n=== Confusion Matrix ===")
    print(cm)
    
    # Estrai i valori dalla confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    # Calcola metriche addizionali
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate (Miss Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate (Fall-out)
    
    print(f"\nSpecificity: {specificity:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    
    # Visualizza la confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CNN Model Confusion Matrix')
    
    # Salva la confusion matrix
    confusion_matrix_path = os.path.join(results_dir, f'confusion_matrix_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()
    
    # Stampa il classification report completo
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred_final))
    
    # Salva i risultati
    results_file = os.path.join(results_dir, f'cnn_predictions_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Salva le metriche
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': roc_auc,
        'specificity': specificity,
        'npv': npv,
        'fnr': fnr,
        'fpr': fpr,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    
    metrics_file = os.path.join(results_dir, f'cnn_metrics_{timestamp}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
