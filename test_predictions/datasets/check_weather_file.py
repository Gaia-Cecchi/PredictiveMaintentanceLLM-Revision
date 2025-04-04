import pandas as pd
import os

# Ottieni il percorso del file meteo
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
weather_data_path = os.path.join(base_dir, 'weather_data', 'halfhourly_weather_data_db_optimized.csv')

print(f"Verifica file meteo: {weather_data_path}")
print(f"Il file esiste: {os.path.exists(weather_data_path)}")

if os.path.exists(weather_data_path):
    # Carica il file meteo
    weather_data = pd.read_csv(weather_data_path)
    
    # Informazioni di base
    print(f"\nDimensione del file: {weather_data.shape}")
    print(f"Colonne: {weather_data.columns.tolist()}")
    
    # Controlla i primi record
    print("\nPrimi 5 record:")
    print(weather_data.head())
    
    # Controlla i tipi di dati
    print("\nTipi di dati:")
    print(weather_data.dtypes)
    
    # Verifica la presenza di valori nulli
    print("\nValori nulli nelle colonne:")
    print(weather_data.isnull().sum())
    
    # Se c'è una colonna timestamp, controlla l'intervallo di date
    if 'timestamp' in weather_data.columns:
        weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        print("\nIntervallo di date:")
        print(f"Inizio: {weather_data['timestamp'].min()}")
        print(f"Fine: {weather_data['timestamp'].max()}")