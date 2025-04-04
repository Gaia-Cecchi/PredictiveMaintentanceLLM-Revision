import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
from config import *
import numpy as np
from tqdm import tqdm
import PyPDF2
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        # Create processed_data directory
        self.processed_data_dir = DATA_BASE_PATH / 'processed_data'
        self.processed_data_dir.mkdir(exist_ok=True)
        
        # Initialize database in processed_data directory
        self.db_path = self.processed_data_dir / 'compressor_data.db'
        self.conn = sqlite3.connect(self.db_path)
        
        self.column_mappings = {
            'Unnamed: 0': 'timestamp',  # La colonna timestamp è chiamata 'Unnamed: 0'
            'Corrente (A)': 'current',
            'CosPhi (Units)': 'cosphi',
            'Energia consumo (kWh)': 'energy_consumption',
            'Energia reattiva (VARh)': 'reactive_energy',
            'Tensione (V)': 'voltage'
        }
        
    def create_database_schema(self):
        """Create the necessary tables in the SQLite database"""
        with self.conn:
            # Drop existing tables if they exist
            self.conn.execute('DROP TABLE IF EXISTS compressor_measurements')
            self.conn.execute('DROP TABLE IF EXISTS weather_data')
            self.conn.execute('DROP TABLE IF EXISTS anomalies')
            self.conn.execute('DROP TABLE IF EXISTS compressor_status')
            self.conn.execute('DROP TABLE IF EXISTS failures')
            self.conn.execute('DROP TABLE IF EXISTS operating_conditions')
            self.conn.execute('DROP TABLE IF EXISTS maintenance')
            self.conn.execute('DROP TABLE IF EXISTS feedback')
            
            # Compressor measurements table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS compressor_measurements (
                    timestamp DATETIME,
                    compressor_id TEXT,
                    measurement_type TEXT,
                    current REAL,
                    cosphi REAL,
                    energy_consumption REAL,
                    reactive_energy REAL,
                    voltage REAL,
                    PRIMARY KEY (timestamp, compressor_id, measurement_type)
                )
            ''')

            # Weather data table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS weather_data (
                    date DATE,
                    location TEXT,
                    avg_temp REAL,
                    min_temp REAL,
                    max_temp REAL,
                    dewpoint REAL,
                    humidity REAL,
                    visibility REAL,
                    avg_wind REAL,
                    max_wind REAL,
                    gust REAL,
                    pressure_slm REAL,
                    avg_pressure REAL,
                    rainfall REAL,
                    phenomena TEXT,
                    PRIMARY KEY (date, location)
                )
            ''')

            # Anomalies table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    timestamp DATETIME,
                    compressor_id TEXT,
                    parameter TEXT,
                    is_anomaly BOOLEAN,
                    true_value REAL,
                    predicted_value REAL,
                    error_value REAL,
                    PRIMARY KEY (timestamp, compressor_id, parameter)
                )
            ''')

            # Update failures table schema to match new CSV structure
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY,
                    date DATETIME NOT NULL,
                    failure_type TEXT,
                    frequency TEXT,
                    cause TEXT,
                    solution TEXT,
                    additional_info TEXT,
                    feedback TEXT,
                    compressor_id TEXT DEFAULT 'CSD102',
                    FOREIGN KEY (compressor_id) REFERENCES compressor_status (compressor_id)
                )
            ''')

            # Add new tables
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS operating_conditions (
                    date TIMESTAMP,
                    start_time TIME,
                    end_time TIME,
                    operating_hours INTEGER,
                    load_percentage INTEGER,
                    temperature FLOAT,
                    humidity INTEGER,
                    vibration FLOAT,
                    pressure FLOAT,
                    compressor_id TEXT,
                    PRIMARY KEY (date, start_time, compressor_id)
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS maintenance (
                    date TIMESTAMP,
                    intervention_number INTEGER,
                    intervention_type TEXT,
                    operating_hours INTEGER,
                    activities TEXT,
                    anomalies TEXT,
                    recommendations TEXT,
                    compressor_id TEXT,
                    PRIMARY KEY (date, intervention_number, compressor_id)
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    date TIMESTAMP,
                    feedback_type TEXT,
                    title TEXT,
                    description TEXT,
                    rating INTEGER,
                    comment TEXT,
                    compressor_id TEXT,
                    PRIMARY KEY (date, title, compressor_id)
                )
            ''')

    def clean_dataframe(self, df):
        """Clean DataFrame and standardize column names"""
        # Rename the timestamp column if it exists as 'Unnamed: 0'
        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'timestamp'})
        
        # Convert timestamp to datetime if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'DATA' in df.columns:
            df['timestamp'] = pd.to_datetime(df['DATA'])
            df = df.drop(columns=['DATA'])
            
        return df

    def process_compressor_1_data(self):
        """Process and store data for compressor 1"""
        logger.info("Processing Compressor 1 data...")

        # Load predictions file with all sheets
        xls = pd.ExcelFile(COMPRESSOR_1_FILES['predictions'])
        true_values = pd.read_excel(xls, 'True values')
        predictions = pd.read_excel(xls, 'Predictions')
        errors = pd.read_excel(xls, 'Errors')
        anomalies = pd.read_excel(xls, 'Anomalies')

        # Debug logging dei dati originali
        logger.info(f"True values columns: {true_values.columns.tolist()}")
        logger.info(f"Sample true values:\n{true_values.head()}")
        
        # Clean DataFrames
        true_values = self.clean_dataframe(true_values)
        predictions = self.clean_dataframe(predictions)
        errors = self.clean_dataframe(errors)
        anomalies = self.clean_dataframe(anomalies)

        # Process daily and hourly data
        daily_data = pd.read_excel(COMPRESSOR_1_FILES['daily_data'])
        hourly_data = pd.read_excel(COMPRESSOR_1_FILES['hourly_data'])

        # Clean and normalize column names
        daily_data = self.clean_dataframe(daily_data)
        hourly_data = self.clean_dataframe(hourly_data)
        
        # Debug logging
        logger.info(f"Original hourly data columns: {hourly_data.columns.tolist()}")
        logger.info(f"Original daily data columns: {daily_data.columns.tolist()}")
        
        # Process measurements
        for df, mtype in [(daily_data, 'daily'), (hourly_data, 'hourly')]:
            # Create a new DataFrame with required structure
            processed_df = pd.DataFrame()
            processed_df['timestamp'] = df['timestamp']
            processed_df['compressor_id'] = 'CSD102'
            processed_df['measurement_type'] = mtype
            
            # Map each column explicitly
            column_data = {
                'current': df['Corrente (A)'] if 'Corrente (A)' in df.columns else 0,
                'cosphi': df['CosPhi (Units)'] if 'CosPhi (Units)' in df.columns else 0,
                'energy_consumption': df['Energia consumo (kWh)'] if 'Energia consumo (kWh)' in df.columns else 0,
                'reactive_energy': df['Energia reattiva (VARh)'] if 'Energia reattiva (VARh)' in df.columns else 0,
                'voltage': df['Tensione (V)'] if 'Tensione (V)' in df.columns else 0
            }
            
            # Add columns to processed DataFrame
            for col_name, data in column_data.items():
                processed_df[col_name] = pd.to_numeric(data, errors='coerce').fillna(0)
            
            # Debug logging
            logger.info(f"Processed {mtype} data columns: {processed_df.columns.tolist()}")
            logger.info(f"Sample of processed {mtype} data:\n{processed_df.head()}")
            
            # Store in SQLite with explicit column order
            processed_df.to_sql(
                'compressor_measurements',
                self.conn,
                if_exists='append',
                index=False
            )

        # Process anomalies with explicit mappings
        anomalies_processed = []
        # Mappa invertita per tutti i parametri
        parameter_mapping = {
            'Current (A)': 'current',
            'CosPhi (Units)': 'cosphi',
            'Energy Consumption (kWh)': 'energy_consumption',
            'Reactive Energy (VARh)': 'reactive_energy',
            'Voltage (V)': 'voltage'
        }
        
        # Debug dei dati prima del processing
        logger.info(f"Processing anomalies with parameters: {parameter_mapping}")
        
        for idx in true_values.index:
            timestamp = true_values.loc[idx, 'timestamp']
            logger.debug(f"Processing row for timestamp {timestamp}")
            
            for orig_param, new_param in parameter_mapping.items():
                # Verifica se la colonna esiste
                if orig_param in true_values.columns:
                    try:
                        anomaly_data = {
                            'timestamp': timestamp,
                            'compressor_id': 'CSD102',
                            'parameter': new_param,
                            'is_anomaly': bool(anomalies.loc[idx, orig_param]),
                            'true_value': float(true_values.loc[idx, orig_param]),
                            'predicted_value': float(predictions.loc[idx, orig_param]),
                            'error_value': float(errors.loc[idx, orig_param])
                        }
                        anomalies_processed.append(anomaly_data)
                        logger.debug(f"Processed anomaly: {anomaly_data}")
                    except Exception as e:
                        logger.error(f"Error processing anomaly for parameter {orig_param}: {str(e)}")
        
        # Debug del risultato
        logger.info(f"Number of processed anomalies: {len(anomalies_processed)}")
        logger.info("Sample of processed anomalies by parameter:")
        for param in parameter_mapping.values():
            param_anomalies = [a for a in anomalies_processed if a['parameter'] == param]
            logger.info(f"{param}: {len(param_anomalies)} anomalies")
            if param_anomalies:
                logger.info(f"Sample {param} anomaly: {param_anomalies[0]}")
        
        anomalies_df = pd.DataFrame(anomalies_processed)
        anomalies_df.to_sql('anomalies', self.conn, if_exists='append', index=False)

    def process_weather_data(self):
        """Process and store weather data"""
        logger.info("Processing weather data...")
        
        weather_columns = {
            'DATA': 'date',
            'LOCALITA': 'location',
            'TMEDIA Â°C': 'avg_temp',
            'TMIN Â°C': 'min_temp',
            'TMAX Â°C': 'max_temp',
            'PUNTORUGIADA Â°C': 'dewpoint',
            'UMIDITA %': 'humidity',
            'VISIBILITA km': 'visibility',
            'VENTOMEDIA km/h': 'avg_wind',
            'VENTOMAX km/h': 'max_wind',
            'RAFFICA km/h': 'gust',
            'PRESSIONESLM mb': 'pressure_slm',
            'PRESSIONEMEDIA mb': 'avg_pressure',
            'PIOGGIA mm': 'rainfall',
            'FENOMENI': 'phenomena'
        }
        
        for file_key, file_path in WEATHER_FILES.items():
            logger.info(f"Processing weather file: {file_key}")
            
            # Leggi il file Excel
            df = pd.read_excel(file_path)
            
            # Rinomina le colonne
            df = df.rename(columns=weather_columns)
            
            # Assicurati che la data sia nel formato corretto
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Converti e pulisci le colonne numeriche
            numeric_columns = [
                'avg_temp', 'min_temp', 'max_temp', 'dewpoint',
                'humidity', 'visibility', 'avg_wind', 'max_wind',
                'gust', 'pressure_slm', 'avg_pressure', 'rainfall'
            ]
            
            # Assicurati che tutti i valori numerici siano float
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
                else:
                    df[col] = 0.0
            
            # Gestisci la colonna phenomena
            df['phenomena'] = df['phenomena'].fillna('').astype(str)
            
            # Debug logging
            logger.info(f"Weather data sample:\n{df.head()}")
            logger.info(f"Weather data types:\n{df.dtypes}")
            
            # Salva nel database
            df.to_sql('weather_data', self.conn, if_exists='append', index=False)

    def process_technical_docs(self):
        """Extract text from technical documentation"""
        logger.info("Processing technical documentation...")
        
        docs_dir = self.processed_data_dir / 'processed_docs'
        docs_dir.mkdir(exist_ok=True)
        
        def extract_pdf_text(pdf_path):
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text
        
        # Process manuals
        docs_text = {
            'CSD102_manual': COMPRESSOR_1_FILES['manual'],
            'climate_interference': TECH_SPECS_FILES['climate_interference'],
            'meeting_report': TECH_SPECS_FILES['meeting_report']
        }
        
        for doc_name, pdf_path in docs_text.items():
            text = extract_pdf_text(pdf_path)
            output_path = docs_dir / f'{doc_name}.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Processed {doc_name} -> {output_path}")

    def register_other_compressors(self):
        """Register the existence of other compressors without detailed data"""
        logger.info("Registering other compressors...")
        
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS compressor_status (
                    compressor_id TEXT PRIMARY KEY,
                    model TEXT,
                    status TEXT,
                    last_updated DATETIME
                )
            ''')
            
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            compressors = [
                ('BSD72', 'BSD72', 'operational', current_time),
                ('BS61', 'BS61', 'operational', current_time),
                ('SK21', 'SK21', 'operational', current_time)
            ]
            
            self.conn.executemany(
                'INSERT OR REPLACE INTO compressor_status VALUES (?, ?, ?, ?)',
                compressors
            )

    def process_failures(self):
        """Process failures data from CSV and store in database"""
        try:
            logger.info("Processing failures data...")
            
            # Read CSV file with explicit date parsing
            failures_df = pd.read_csv(
                MATERIALS_FILES['guasti'],
                parse_dates=['Data e Ora'],
                date_parser=lambda x: pd.to_datetime(x, errors='coerce')  # Handle parsing errors gracefully
            )
            
            # Log raw data for debugging
            logger.info("Raw CSV data sample:")
            logger.info(f"First 5 rows:\n{failures_df.head().to_string()}")
            logger.info(f"Raw columns: {failures_df.columns.tolist()}")
            logger.info(f"Raw data types:\n{failures_df.dtypes}")
            
            # Handle missing or invalid dates
            invalid_dates = failures_df['Data e Ora'].isna()
            if invalid_dates.any():
                logger.warning(f"Found {invalid_dates.sum()} rows with invalid dates")
                logger.warning("Invalid date rows:")
                logger.warning(failures_df[invalid_dates].to_string())
                failures_df = failures_df[~invalid_dates]
            
            # Column mapping for database
            column_mapping = {
                'Data e Ora': 'date',
                'Tipologia del Guasto': 'failure_type',
                'Frequenza': 'frequency',
                'Causa': 'cause',
                'Soluzione': 'solution',
                'Altre Informazioni': 'additional_info',
                'Feedback': 'feedback'
            }
            
            # Rename columns and add defaults
            failures_df = failures_df.rename(columns=column_mapping)
            failures_df['compressor_id'] = 'CSD102'  # Default compressor ID
            
            # Ensure date format is correct and sort by date
            failures_df['date'] = pd.to_datetime(failures_df['date'])
            failures_df = failures_df.sort_values('date')
            
            # Log processed data for verification
            logger.info("Processed data summary:")
            logger.info(f"Total records: {len(failures_df)}")
            logger.info(f"Date range: {failures_df['date'].min()} to {failures_df['date'].max()}")
            logger.info("Sample processed data:")
            logger.info(failures_df.head().to_string())
            
            # Database operations with explicit error handling
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # First backup existing data
                    backup_query = "SELECT * FROM failures"
                    backup_df = pd.read_sql_query(backup_query, conn)
                    logger.info(f"Backed up {len(backup_df)} existing failure records")
                    
                    # Clear existing data
                    conn.execute('DELETE FROM failures')
                    logger.info("Cleared existing failures table")
                    
                    # Insert new data
                    failures_df.to_sql(
                        'failures',
                        conn,
                        if_exists='append',
                        index=False
                    )
                    
                    # Verify insertion
                    verify_query = "SELECT * FROM failures ORDER BY date"
                    verify_df = pd.read_sql_query(verify_query, conn)
                    
                    logger.info("Verification after insertion:")
                    logger.info(f"Total records in database: {len(verify_df)}")
                    logger.info(f"Date range in database: {verify_df['date'].min()} to {verify_df['date'].max()}")
                    logger.info("Sample from database:")
                    logger.info(verify_df.head().to_string())
                    
                    # Verify no data was lost
                    if len(verify_df) != len(failures_df):
                        logger.error(f"Data loss detected! CSV had {len(failures_df)} records but DB has {len(verify_df)}")
                        # Restore backup if needed
                        conn.execute('DELETE FROM failures')
                        backup_df.to_sql('failures', conn, if_exists='append', index=False)
                        raise ValueError("Data loss detected during insertion")
                        
                logger.info("Failures data processing completed successfully")
                    
            except sqlite3.Error as e:
                logger.error(f"Database error: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error processing failures data: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def process_materials(self):
        """Process text files from materials directory"""
        logger.info("Processing materials text files...")
        
        # Process failures CSV first
        self.process_failures()
        
        # Create materials directory in processed_docs
        materials_dir = self.processed_data_dir / 'processed_docs' / 'materials'
        materials_dir.mkdir(exist_ok=True)
        
        # Process remaining text files
        text_files = {k: v for k, v in MATERIALS_FILES.items() if k != 'guasti'}
        
        for doc_name, file_path in text_files.items():
            try:
                # Read and process the text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Save to processed_docs with appropriate naming
                output_path = materials_dir / f'{doc_name}.txt'
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"Processed {doc_name} -> {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {doc_name}: {str(e)}")

    def process_operating_conditions(self):
        """Process operating conditions data from CSV"""
        try:
            logger.info("Processing operating conditions data...")
            
            df = pd.read_csv(MATERIALS_FILES['carico_operativo'])
            df['date'] = pd.to_datetime(df['Data'])
            
            # Add compressor_id if not present
            if 'compressor_id' not in df.columns:
                df['compressor_id'] = 'CSD102'
            
            # Prepare data for database
            operating_data = df.rename(columns={
                'Ora_Inizio': 'start_time',
                'Ora_Fine': 'end_time',
                'Ore_Funzionamento': 'operating_hours',
                'Carico_Operativo': 'load_percentage',
                'Temperatura': 'temperature',
                'Umidita': 'humidity',
                'Vibrazioni': 'vibration',
                'Pressione': 'pressure'
            })
            
            # Insert into database
            with self.conn:
                operating_data.to_sql('operating_conditions', self.conn, if_exists='replace', index=False)
            
            logger.info(f"Successfully processed {len(df)} operating condition records")
            
        except Exception as e:
            logger.error(f"Error processing operating conditions: {str(e)}")
            raise

    def process_maintenance(self):
        """Process maintenance data from CSV"""
        try:
            logger.info("Processing maintenance data...")
            
            df = pd.read_csv(MATERIALS_FILES['manutenzioni'])
            df['date'] = pd.to_datetime(df['Data'])
            
            if 'compressor_id' not in df.columns:
                df['compressor_id'] = 'CSD102'
            
            maintenance_data = df.rename(columns={
                'Numero_Intervento': 'intervention_number',
                'Tipo_Intervento': 'intervention_type',
                'Ore_Funzionamento': 'operating_hours',
                'Attivita': 'activities',
                'Anomalie': 'anomalies',
                'Raccomandazioni': 'recommendations'
            })
            
            with self.conn:
                maintenance_data.to_sql('maintenance', self.conn, if_exists='replace', index=False)
            
            logger.info(f"Successfully processed {len(df)} maintenance records")
            
        except Exception as e:
            logger.error(f"Error processing maintenance data: {str(e)}")
            raise

    def process_feedback(self):
        """Process feedback data from CSV"""
        try:
            logger.info("Processing feedback data...")
            
            df = pd.read_csv(MATERIALS_FILES['feedback'])
            
            # Log original data for debugging
            logger.info(f"Original columns: {df.columns.tolist()}")
            logger.info(f"Date sample before conversion:\n{df['Data'].head()}")
            
            # Convert dates from Italian format (DD/MM/YYYY)
            df['date'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
            logger.info(f"Date sample after conversion:\n{df['date'].head()}")
            
            if 'compressor_id' not in df.columns:
                df['compressor_id'] = 'CSD102'
            
            feedback_data = df.rename(columns={
                'Tipo': 'feedback_type',
                'Titolo': 'title',
                'Descrizione': 'description',
                'Valutazione': 'rating',
                'Commento': 'comment'
            })
            
            # Verify data types before insertion
            logger.info(f"Data types after processing:\n{feedback_data.dtypes}")
            
            with self.conn:
                feedback_data.to_sql('feedback', self.conn, if_exists='replace', index=False)
            
            logger.info(f"Successfully processed {len(df)} feedback records")
            
        except Exception as e:
            logger.error(f"Error processing feedback data: {str(e)}")
            logger.error(f"Data sample:\n{df['Data'].head() if 'df' in locals() else 'No data loaded'}")
            raise

    def prepare_vector_store_data(self):
        """Prepare data from all sources for vector store with balanced representation"""
        try:
            logger.info("Preparing data for vector store...")
            vector_store_dir = self.processed_data_dir / 'vector_store'
            vector_store_dir.mkdir(exist_ok=True)

            # Combine all relevant data from database
            with self.conn:
                # Updated failures query to match new schema
                failures_query = """
                    SELECT 
                        date,
                        failure_type as type,
                        frequency,
                        cause as description,
                        solution,
                        additional_info,
                        feedback,
                        compressor_id,
                        'failure' as source
                    FROM failures
                    ORDER BY date DESC
                """
                
                logger.info("Fetching failures for vector store...")
                failures_data = pd.read_sql(failures_query, self.conn)
                logger.info(f"Found {len(failures_data)} failures")
                
                if failures_data.empty:
                    logger.warning("No failures found in database!")
                else:
                    logger.info(f"Sample failure data:\n{failures_data.head()}")

                # Get maintenance data
                maintenance_data = pd.read_sql("""
                    SELECT 
                        date,
                        intervention_type as type,
                        activities as description,
                        anomalies,
                        recommendations,
                        'maintenance' as source
                    FROM maintenance
                """, self.conn)
                
                # Get feedback data
                feedback_data = pd.read_sql("""
                    SELECT 
                        date,
                        feedback_type as type,
                        description,
                        comment as additional_info,
                        rating,
                        'feedback' as source
                    FROM feedback
                """, self.conn)
                
                # Get operating conditions data with anomalies
                operating_data = pd.read_sql("""
                    SELECT 
                        oc.date,
                        'operating_condition' as type,
                        CASE 
                            WHEN a.is_anomaly = 1 THEN 'Anomaly detected'
                            ELSE 'Normal operation'
                        END as condition_status,
                        a.parameter,
                        a.true_value,
                        a.predicted_value,
                        'operating_condition' as source
                    FROM operating_conditions oc
                    LEFT JOIN anomalies a ON oc.date = a.timestamp
                    WHERE a.is_anomaly = 1
                """, self.conn)

                # Get anomalies specifically for better representation
                anomalies_data = pd.read_sql("""
                    SELECT 
                        timestamp as date,
                        'anomaly' as type,
                        parameter,
                        true_value,
                        predicted_value,
                        error_value,
                        'anomaly' as source
                    FROM anomalies
                    WHERE is_anomaly = 1
                    ORDER BY timestamp DESC 
                    LIMIT 500
                """, self.conn)

            # Balance representation by limiting each category to a similar number
            # First, determine the target number per category
            target_count = 200  # For example
            
            # Get balanced samples from each category
            if len(failures_data) > target_count:
                failures_data = failures_data.sample(target_count)
            
            if len(maintenance_data) > target_count:
                maintenance_data = maintenance_data.sample(target_count)
                
            if len(feedback_data) > target_count:
                feedback_data = feedback_data.sample(target_count)
                
            if len(operating_data) > target_count:
                operating_data = operating_data.sample(target_count)
                
            if len(anomalies_data) > target_count:
                anomalies_data = anomalies_data.sample(target_count)

            # Create text documents for vector store with more balanced representation
            docs = []
            
            # Update the document creation for failures to match new schema
            for _, row in failures_data.iterrows():
                doc = f"""Type: Failure
Category: Failure Record
Date: {row['date']}
Failure Type: {row['type']}
Frequency: {row['frequency']}
Description: {row['description']}
Solution: {row['solution']}
Additional Info: {row['additional_info']}
Feedback: {row['feedback']}
"""
                docs.append(doc)

            for _, row in maintenance_data.iterrows():
                doc = f"""Type: Maintenance
Category: Maintenance Record
Date: {row['date']}
Intervention Type: {row['type']}
Activities: {row['description']}
Anomalies: {row['anomalies']}
Recommendations: {row['recommendations']}
"""
                docs.append(doc)

            for _, row in feedback_data.iterrows():
                doc = f"""Type: User Feedback
Category: Operator Feedback
Date: {row['date']}
Feedback Type: {row['type']}
Description: {row['description']}
Comment: {row['additional_info']}
Rating: {row['rating']}/5
"""
                docs.append(doc)

            for _, row in operating_data.iterrows():
                doc = f"""Type: Operating Condition
Category: System Operation
Date: {row['date']}
Status: {row['condition_status']}
Parameter: {row['parameter']}
Actual Value: {row['true_value']}
Expected Value: {row['predicted_value']}
"""
                docs.append(doc)

            for _, row in anomalies_data.iterrows():
                doc = f"""Type: Anomaly Detection
Category: System Anomaly
Date: {row['date']}
Parameter: {row['parameter']}
True Value: {row['true_value']}
Predicted Value: {row['predicted_value']}
Error Value: {row['error_value']}
"""
                docs.append(doc)

            # Save documents
            vector_store_path = vector_store_dir / 'combined_data.txt'
            with open(vector_store_path, 'w', encoding='utf-8') as f:
                f.write('\n---\n'.join(docs))

            logger.info(f"Vector store data prepared with balanced representation: {len(docs)} documents created")
            return vector_store_path

        except Exception as e:
            logger.error(f"Error preparing vector store data: {str(e)}")
            raise

    def process_all(self):
        """Process all data sources"""
        try:
            logger.info("Starting data preprocessing...")
            
            # Create database schema
            self.create_database_schema()
            
            # Process all data sources
            self.process_compressor_1_data()
            self.process_weather_data()
            self.process_technical_docs()
            self.process_failures()  # Add failures processing
            self.process_materials()
            self.register_other_compressors()
            
            # Process new data sources
            self.process_operating_conditions()
            self.process_maintenance()
            self.process_feedback()
            
            # Prepare vector store data after all other processing
            self.prepare_vector_store_data()
            
            logger.info("Data preprocessing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise
        finally:
            self.conn.close()

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_all()