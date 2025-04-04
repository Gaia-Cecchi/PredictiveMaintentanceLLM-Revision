import sqlite3
import pandas as pd
import os

# Funzione per controllare un database
def check_database(db_path):
    print(f"\n=== Controllo del database: {os.path.basename(db_path)} ===")
    if not os.path.exists(db_path):
        print(f"ERRORE: Il database {db_path} non esiste!")
        return
        
    conn = sqlite3.connect(db_path)
    
    # Lista tutte le tabelle nel database
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print("\nTabelle presenti nel database:")
    print(tables)
    
    # Per ogni tabella, mostra la struttura e i primi record
    for table_name in tables['name']:
        print(f"\n--- Struttura della tabella: {table_name} ---")
        # Ottieni informazione sulle colonne
        columns = pd.read_sql(f"PRAGMA table_info({table_name})", conn)
        print(columns[['name', 'type']])
        
        # Mostra i primi record
        print(f"\n--- Primi 3 record della tabella: {table_name} ---")
        data = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn)
        print(data)
        
        # Controlla esplicitamente le colonne meteo
        weather_cols = [col for col in data.columns if col.startswith('weather_')]
        if weather_cols:
            print(f"\nColonne meteo trovate in {table_name}: {weather_cols}")
    
    conn.close()

# Percorsi dei database
script_dir = os.path.dirname(os.path.abspath(__file__))
db1 = os.path.join(script_dir, 'compressor_data_2024.db')
db2 = os.path.join(script_dir, 'compressor_data_2024_etichettato.db')

# Controlla entrambi i database
check_database(db1)
check_database(db2)