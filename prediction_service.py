import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime, timedelta
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import json
import pickle  # Add pickle import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model constants
TIMESTEPS = 24
FEATURES = 5
LEARNING_RATE = 0.0005

def create_model():
    """Create LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIMESTEPS, FEATURES)),
        tf.keras.layers.LSTM(units=10, return_sequences=False),
        tf.keras.layers.Dense(32, 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(FEATURES, 'linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model

class PredictionService:
    def __init__(self, base_path="/teamspace/studios/this_studio"):
        self.base_path = Path(base_path)
        self.model = None
        self.scaler = None
        self.timesteps = TIMESTEPS
        self.features = FEATURES
        self.db_path = self.base_path / 'processed_data' / 'compressor_data.db'
        
        # Definizione delle correlazioni tra condizioni e parametri
        self.parameter_correlations = {
            'temperature': {
                'current': 0.3,      # Maggiore temperatura -> maggiore corrente
                'cosphi': -0.1,      # Temperatura elevata può ridurre l'efficienza
                'energy_consumption': 0.4,
                'reactive_energy': 0.2,
                'voltage': -0.05
            },
            'humidity': {
                'current': 0.1,
                'cosphi': -0.15,
                'energy_consumption': 0.2,
                'reactive_energy': 0.25,
                'voltage': -0.1
            },
            'pressure': {
                'current': 0.15,
                'cosphi': -0.05,
                'energy_consumption': 0.2,
                'reactive_energy': 0.1,
                'voltage': 0.05
            },
            'load_percentage': {
                'current': 0.5,
                'cosphi': -0.2,
                'energy_consumption': 0.6,
                'reactive_energy': 0.4,
                'voltage': -0.1
            }
        }
        
        self.load_model()
        self.initialize_scaler()

    def load_model(self):
        """Load model and saved weights"""
        try:
            logger.info("Loading model and weights...")
            self.model = create_model()
            weights_path = self.base_path / 'processed_data' / 'models' / 'weights_lstm.npz'
            
            if weights_path.exists():
                weights_npz = np.load(weights_path)
                weights = [weights_npz[f"arr_{i}"] for i in range(len(weights_npz.files))]
                self.model.set_weights(weights)
                logger.info("Model loaded with existing weights")
            else:
                logger.warning("No existing weights found, using initialized weights")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _initialize_scaler_from_data(self):
        """Inizializza lo scaler usando i dati storici (fallback method)"""
        try:
            logger.info("Initializing scaler from historical data...")
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT current, cosphi, energy_consumption, reactive_energy, voltage
                    FROM compressor_measurements
                    WHERE compressor_id = 'CSD102'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                df = pd.read_sql_query(query, conn)
                
                self.scaler = MinMaxScaler()
                self.scaler.fit(df)
                logger.info("Scaler initialized successfully from data")
        except Exception as e:
            logger.error(f"Error initializing scaler from data: {str(e)}")
            raise

    def initialize_scaler(self):
        """Inizializza lo scaler dai dati salvati"""
        try:
            logger.info("Loading saved scaler...")
            scaler_path = self.base_path / 'processed_data' / 'models' / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("No saved scaler found, initializing from data...")
                self._initialize_scaler_from_data()
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            self._initialize_scaler_from_data()

    def get_base_data(self, base_time: datetime, compressor_id: str = 'CSD102') -> pd.DataFrame:
        """Recupera i dati base per la predizione"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        current, cosphi, energy_consumption, reactive_energy, voltage
                    FROM compressor_measurements
                    WHERE compressor_id = ?
                    AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[compressor_id, base_time.strftime('%Y-%m-%d %H:%M:%S'), self.timesteps]
                )
                
                if len(df) < self.timesteps:
                    raise ValueError(f"Insufficient data: found {len(df)} samples, need {self.timesteps}")
                
                return df.iloc[::-1]  # Inverti l'ordine per avere sequenza temporale corretta
        except Exception as e:
            logger.error(f"Error getting base data: {str(e)}")
            raise

    def apply_modifications(self, base_data: np.ndarray, modifications: dict, current_conditions: dict = None) -> np.ndarray:
        """
        Applica le modifiche ai dati base secondo le correlazioni definite
        
        Args:
            base_data: Dati base normalizzati
            modifications: Modifiche specificate dall'utente
            current_conditions: Condizioni attuali dal sistema (se None, non applica differenziali)
        """
        try:
            modified_data = base_data.copy()
            
            # Se non abbiamo le condizioni attuali, non possiamo calcolare differenze
            if current_conditions is None:
                logger.warning("No current conditions provided, returning unmodified data")
                return modified_data
            
            # Per ogni condizione modificata
            for condition, value in modifications['weather'].items():
                if condition in self.parameter_correlations:
                    # Calcola la differenza rispetto alle condizioni attuali
                    if condition in current_conditions['weather']:
                        current_value = current_conditions['weather'][condition]
                        # Calcola il cambiamento percentuale rispetto al valore attuale
                        percent_change = ((value - current_value) / current_value) if current_value != 0 else 0
                        
                        # Per ogni parametro influenzato
                        for param_idx, (param, correlation) in enumerate(self.parameter_correlations[condition].items()):
                            # Applica la modifica proporzionale basata sul cambiamento percentuale
                            impact = correlation * percent_change
                            modified_data[:, param_idx] *= (1 + impact)
                            
                            logger.info(f"Modifica {condition}: {current_value} -> {value} " 
                                        f"(cambio: {percent_change:.2%}), impatto su {param}: {impact:.2%}")
            
            # Applica modifiche operative con stessa logica
            for condition, value in modifications['operational'].items():
                if condition in self.parameter_correlations:
                    if condition in current_conditions['operational']:
                        current_value = current_conditions['operational'][condition]
                        percent_change = ((value - current_value) / current_value) if current_value != 0 else 0
                        
                        for param_idx, (param, correlation) in enumerate(self.parameter_correlations[condition].items()):
                            impact = correlation * percent_change
                            modified_data[:, param_idx] *= (1 + impact)
            
            # Assicura che i valori rimangano nel range [0, 1] dopo lo scaling
            modified_data = np.clip(modified_data, 0, 1)
            
            return modified_data
                
        except Exception as e:
            logger.error(f"Error applying modifications: {str(e)}")
            raise

    def get_actual_conditions(self, timestamp: datetime, compressor_id: str = 'CSD102') -> dict:
        """
        Recupera i valori reali dal database al timestamp specificato
        
        Args:
            timestamp: Il momento temporale di riferimento
            compressor_id: L'identificatore del compressore
            
        Returns:
            dict: Dizionario con i valori ambientali e operativi attuali
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Recupera i dati meteo più vicini al timestamp
                weather_query = """
                    SELECT date, avg_temp as temperature, humidity, avg_pressure as pressure
                    FROM weather_data
                    WHERE date <= ?
                    ORDER BY date DESC
                    LIMIT 1
                """
                weather_df = pd.read_sql_query(
                    weather_query, 
                    conn, 
                    params=[timestamp.strftime('%Y-%m-%d %H:%M:%S')]
                )
                
                # Recupera i dati operativi più vicini al timestamp
                op_query = """
                    SELECT date, load_percentage, operating_hours
                    FROM operating_conditions
                    WHERE date <= ? AND compressor_id = ?
                    ORDER BY date DESC
                    LIMIT 1
                """
                op_df = pd.read_sql_query(
                    op_query, 
                    conn, 
                    params=[timestamp.strftime('%Y-%m-%d %H:%M:%S'), compressor_id]
                )
                
                # Imposta valori di default in caso di dati mancanti
                actual_conditions = {
                    'weather': {
                        'temperature': 20.0,
                        'humidity': 50.0,
                        'pressure': 1013.25
                    },
                    'operational': {
                        'load_percentage': 75.0,
                        'operating_hours': 8.0
                    }
                }
                
                # Aggiorna con valori dal database se disponibili
                if not weather_df.empty:
                    for key in ['temperature', 'humidity', 'pressure']:
                        if key in weather_df.columns:
                            actual_conditions['weather'][key] = float(weather_df[key].iloc[0])
                    
                    logger.info(f"Dati meteo trovati per {weather_df['date'].iloc[0]}")
                else:
                    logger.warning(f"Nessun dato meteo trovato precedente a {timestamp}")
                
                # Aggiorna con valori operativi se disponibili
                if not op_df.empty:
                    for key in ['load_percentage', 'operating_hours']:
                        if key in op_df.columns:
                            actual_conditions['operational'][key] = float(op_df[key].iloc[0])
                    
                    logger.info(f"Dati operativi trovati per {op_df['date'].iloc[0]}")
                else:
                    logger.warning(f"Nessun dato operativo trovato precedente a {timestamp}")
                
                return actual_conditions
        
        except Exception as e:
            logger.error(f"Errore nel recupero dei dati attuali: {str(e)}")
            # Ritorna valori di default in caso di errore
            return {
                'weather': {'temperature': 20.0, 'humidity': 50.0, 'pressure': 1013.25},
                'operational': {'load_percentage': 75.0, 'operating_hours': 8.0}
            }

    def calculate_impact(self, actual_value: float, modified_value: float, correlation: float, sensitivity: float = 1.0) -> float:
        """
        Calcola l'impatto di una modifica di parametro
        
        Args:
            actual_value: Valore attuale dal database
            modified_value: Valore modificato dall'utente
            correlation: Coefficiente di correlazione
            sensitivity: Fattore di sensibilità per modulare l'impatto
            
        Returns:
            float: Impatto calcolato
        """
        if actual_value == 0:
            # Evita divisione per zero
            percent_change = 0
        else:
            # Calcola la variazione percentuale
            percent_change = (modified_value - actual_value) / actual_value
        
        # Applica correlazione e sensibilità
        impact = correlation * percent_change * sensitivity
        
        # Log per debug
        logger.debug(f"Valore attuale: {actual_value}, Modificato: {modified_value}, " 
                   f"Variazione: {percent_change:.2%}, Correlazione: {correlation}, "
                   f"Sensibilità: {sensitivity}, Impatto: {impact:.4f}")
        
        return impact

    def apply_modifications_advanced(self, base_data: np.ndarray, modifications: dict, actual_conditions: dict) -> tuple:
        """
        Applica modifiche ai dati base in modo più preciso usando i valori effettivi dal database
        
        Args:
            base_data: Dati normalizzati di base
            modifications: Modifiche specificate dall'utente
            actual_conditions: Condizioni reali dal database
            
        Returns:
            tuple: (Dati modificati, Flag che indica se ci sono modifiche significative)
        """
        try:
            modified_data = base_data.copy()
            parameters = ['current', 'cosphi', 'energy_consumption', 'reactive_energy', 'voltage']
            
            # Definisci sensibilità per diverse categorie
            sensitivities = {
                'temperature': 0.8,      # Alta sensibilità alla temperatura
                'humidity': 0.5,         # Media sensibilità all'umidità
                'pressure': 0.3,         # Bassa sensibilità alla pressione
                'load_percentage': 1.5,  # Altissima sensibilità al carico
                'operating_hours': 0.7   # Media-alta sensibilità alle ore di funzionamento
            }
            
            # Traccia se ci sono modifiche significative
            has_significant_changes = False
            
            # Applica modifiche ai parametri meteo
            for condition, modified_value in modifications['weather'].items():
                if condition in self.parameter_correlations:
                    actual_value = actual_conditions['weather'][condition]
                    
                    # Controlla se c'è una differenza significativa
                    percent_diff = abs((modified_value - actual_value) / actual_value) if actual_value != 0 else 0
                    if percent_diff > 0.01:  # 1% di differenza è significativa
                        has_significant_changes = True
                        
                        # Applica l'impatto a tutti i parametri correlati
                        for param_idx, (param, correlation) in enumerate(self.parameter_correlations[condition].items()):
                            impact = self.calculate_impact(
                                actual_value, 
                                modified_value,
                                correlation,
                                sensitivities.get(condition, 1.0)
                            )
                            modified_data[:, param_idx] *= (1 + impact)
                            
                            logger.info(f"Modifica {condition}: {actual_value:.2f} → {modified_value:.2f} "
                                      f"(diff: {percent_diff:.2%}), impatto su {param}: {impact:.2%}")
            
            # Applica modifiche ai parametri operativi
            for condition, modified_value in modifications['operational'].items():
                if condition in self.parameter_correlations:
                    actual_value = actual_conditions['operational'][condition]
                    
                    # Controlla se c'è una differenza significativa
                    percent_diff = abs((modified_value - actual_value) / actual_value) if actual_value != 0 else 0
                    if percent_diff > 0.01:  # 1% di differenza è significativa
                        has_significant_changes = True
                        
                        # Applica l'impatto a tutti i parametri correlati
                        for param_idx, (param, correlation) in enumerate(self.parameter_correlations[condition].items()):
                            impact = self.calculate_impact(
                                actual_value, 
                                modified_value,
                                correlation,
                                sensitivities.get(condition, 1.0)
                            )
                            modified_data[:, param_idx] *= (1 + impact)
                            
                            logger.info(f"Modifica {condition}: {actual_value:.2f} → {modified_value:.2f} "
                                      f"(diff: {percent_diff:.2%}), impatto su {param}: {impact:.2%}")
            
            # Assicurati che i valori rimangano nel range valido dopo lo scaling
            modified_data = np.clip(modified_data, 0, 1)
            
            return modified_data, has_significant_changes
        
        except Exception as e:
            logger.error(f"Errore nell'applicazione delle modifiche: {str(e)}")
            raise

    def predict_with_modified_conditions(self, 
                                      base_time: datetime,
                                      modifications: dict,
                                      compressor_id: str = 'CSD102') -> dict:
        """
        Genera predizioni con condizioni modificate in modo più preciso
        
        Args:
            base_time: Timestamp di riferimento
            modifications: Modifiche specificate dall'utente
            compressor_id: ID del compressore
        """
        try:
            # Ottieni i dati base
            base_data = self.get_base_data(base_time, compressor_id)
            
            # Recupera le condizioni attuali dal database
            actual_conditions = self.get_actual_conditions(base_time, compressor_id)
            
            # Log delle condizioni attuali e modificate
            logger.info(f"Condizioni attuali al {base_time}: {actual_conditions}")
            logger.info(f"Modifiche richieste: {modifications}")
            
            # Scala i dati
            scaled_base = self.scaler.transform(base_data)
            
            # Applica le modifiche
            modified_data, has_significant_changes = self.apply_modifications_advanced(
                scaled_base, 
                modifications, 
                actual_conditions
            )
            
            # Genera predizione base
            base_pred = self.model.predict(
                scaled_base.reshape(1, self.timesteps, self.features),
                verbose=0
            )
            
            # Genera predizione modificata solo se ci sono cambiamenti significativi
            if has_significant_changes:
                mod_pred = self.model.predict(
                    modified_data.reshape(1, self.timesteps, self.features),
                    verbose=0
                )
            else:
                logger.info("Nessuna modifica significativa. Utilizzo della stessa predizione di base.")
                mod_pred = base_pred
            
            # Inverti lo scaling
            base_pred_inv = self.scaler.inverse_transform(base_pred)
            mod_pred_inv = self.scaler.inverse_transform(mod_pred)
            
            # Prepara il risultato
            parameters = ['current', 'cosphi', 'energy_consumption', 'reactive_energy', 'voltage']
            result = {
                'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
                'compressor_id': compressor_id,
                'base_prediction': {
                    param: float(base_pred_inv[0][i])
                    for i, param in enumerate(parameters)
                },
                'modified_prediction': {
                    param: float(mod_pred_inv[0][i])
                    for i, param in enumerate(parameters)
                },
                'differences': {
                    param: float(mod_pred_inv[0][i] - base_pred_inv[0][i])
                    for i, param in enumerate(parameters)
                },
                'modifications': modifications,
                'has_significant_changes': has_significant_changes,
                'actual_conditions': actual_conditions
            }
            
            # Salva la predizione nel database
            self.save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Errore nella generazione della predizione: {str(e)}")
            raise

    def save_prediction(self, prediction: dict):
        """Salva la predizione nel database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Crea la tabella se non esiste
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        compressor_id TEXT,
                        scenario TEXT,
                        modifications TEXT,
                        current REAL,
                        cosphi REAL,
                        energy_consumption REAL,
                        reactive_energy REAL,
                        voltage REAL
                    )
                """)
                
                # Inserisci predizione base
                base_values = [
                    prediction['timestamp'],
                    prediction['compressor_id'],
                    'base',
                    None
                ] + [prediction['base_prediction'][param] for param in [
                    'current', 'cosphi', 'energy_consumption', 'reactive_energy', 'voltage'
                ]]
                
                # Inserisci predizione modificata
                mod_values = [
                    prediction['timestamp'],
                    prediction['compressor_id'],
                    'modified',
                    json.dumps(prediction['modifications'])
                ] + [prediction['modified_prediction'][param] for param in [
                    'current', 'cosphi', 'energy_consumption', 'reactive_energy', 'voltage'
                ]]
                
                conn.executemany(
                    """
                    INSERT INTO predictions (
                        timestamp, compressor_id, scenario, modifications,
                        current, cosphi, energy_consumption, reactive_energy, voltage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [base_values, mod_values]
                )
                
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise

    def get_saved_predictions(self, 
                            start_time: datetime,
                            end_time: datetime,
                            compressor_id: str = 'CSD102') -> pd.DataFrame:
        """Recupera le predizioni salvate per un periodo specifico"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT *
                    FROM predictions
                    WHERE compressor_id = ?
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """
                return pd.read_sql_query(
                    query,
                    conn,
                    params=[
                        compressor_id,
                        start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        end_time.strftime('%Y-%m-%d %H:%M:%S')
                    ],
                    parse_dates=['timestamp']
                )
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            raise

def main():
    """Test function"""
    service = PredictionService()
    
    # Test prediction
    test_modifications = {
        'weather': {
            'temperature': 30.0,
            'humidity': 75.0,
            'pressure': 1015.0
        },
        'operational': {
            'load_percentage': 85.0,
            'operating_hours': 8.0
        }
    }
    
    result = service.predict_with_modified_conditions(
        base_time=datetime.utcnow(),
        modifications=test_modifications
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()