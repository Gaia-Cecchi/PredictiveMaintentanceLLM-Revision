import pandas as pd
import numpy np
import sqlite3
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path
from llm_interface import CompressorAssistant
from llm_chat import KnowledgeChatAssistant
from prediction_service import PredictionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data access functions
def get_compressors(db_path):
    """Get list of available compressors"""
    with sqlite3.connect(db_path) as conn:
        query = "SELECT DISTINCT compressor_id FROM compressor_measurements"
        df = pd.read_sql_query(query, conn)
        return df['compressor_id'].tolist()

def get_anomalies(db_path, compressor_id, start_date=None, end_date=None):
    """Get anomalies for a specific compressor"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT *
            FROM anomalies
            WHERE compressor_id = ?
        """
        params = [compressor_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        # Explicitly convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

def get_measurements(db_path, compressor_id, start_date=None, end_date=None, _sim_time=None):
    """Get measurements for a specific compressor"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT *
            FROM compressor_measurements
            WHERE compressor_id = ?
        """
        params = [compressor_id]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp DESC"
        
        return pd.read_sql_query(query, conn, params=params)

def get_weather_data(db_path, date):
    """Get weather data for a specific date"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT *
            FROM weather_data
            WHERE date = DATE(?)
        """
        return pd.read_sql_query(query, conn, params=[date])

def calculate_kpis(db_path, compressor_id, current_time, timeframe='7d'):
    """Calculate KPIs for a specific compressor"""
    end_date = current_time
    if timeframe == '7d':
        start_date = end_date - timedelta(days=7)
    elif timeframe == '30d':
        start_date = end_date - timedelta(days=30)
    else:
        start_date = end_date - timedelta(days=1)

    # Get all anomalies in the period
    anomalies_df = get_anomalies(db_path, compressor_id, start_date, end_date)
    # Filter only true anomalies
    true_anomalies = anomalies_df[anomalies_df['is_anomaly'] == True]
    
    # Get all measurements in the period
    measurements = get_measurements(db_path, compressor_id, start_date, end_date)
    
    # Calculate total number of unique measurements (by timestamp)
    total_timestamps = len(measurements['timestamp'].unique()) if not measurements.empty else 0
    
    # Calculate number of timestamps with anomalies
    anomaly_timestamps = len(true_anomalies['timestamp'].unique()) if not true_anomalies.empty else 0
    
    return {
        'anomaly_count': anomaly_timestamps,  # Now counts only timestamps with true anomalies
        'avg_cosphi': measurements['cosphi'].mean() if not measurements.empty else 0,
        'reliability': 1 - (anomaly_timestamps / total_timestamps) if total_timestamps > 0 else 1,
        'total_runtime': len(measurements) * 5 / 60  # Assuming 5-minute intervals
    }

def get_historical_values(db_path, start_date, end_date):
    """Get all values between two dates"""
    try:
        with sqlite3.connect(db_path) as conn:
            # Query anomalies with explicit hour selection
            anomalies_query = """
                SELECT 
                    timestamp,
                    compressor_id,
                    parameter,
                    is_anomaly,
                    true_value,
                    predicted_value,
                    error_value
                FROM anomalies
                WHERE DATE(timestamp) BETWEEN DATE(?) AND DATE(?)
                AND strftime('%M', timestamp) = '00'  -- Only exact hours
                ORDER BY timestamp ASC
            """
            
            anomalies_df = pd.read_sql_query(
                anomalies_query, 
                conn, 
                params=[start_date, end_date],
                parse_dates=['timestamp']
            )
            
            # Query weather data for the date range
            weather_query = """
                SELECT * FROM weather_data
                WHERE date BETWEEN DATE(?) AND DATE(?)
                ORDER BY date ASC
            """
            weather_df = pd.read_sql_query(
                weather_query,
                conn,
                params=[start_date, end_date]
            )
            
            # Process weather data
            if not weather_df.empty:
                weather_data = []
                for _, row in weather_df.iterrows():
                    weather_dict = {}
                    for key, value in row.items():
                        if pd.isna(value):
                            weather_dict[key] = None if key == 'phenomena' else 0.0
                        elif isinstance(value, pd.Timestamp):
                            weather_dict[key] = value.strftime('%Y-%m-%d')
                        else:
                            weather_dict[key] = float(value) if isinstance(value, (int, float)) else str(value)
                    weather_data.append(weather_dict)
            else:
                weather_data = None
            
            return {
                'anomalies': anomalies_df.to_dict('records') if not anomalies_df.empty else None,
                'weather': weather_data
            }
            
    except Exception as e:
        logger.error(f"Error retrieving historical data: {str(e)}")
        return None

def get_historical_maintenance(db_path, start_date, end_date):
    """Get maintenance records between dates"""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT 
                    date,
                    intervention_number,
                    intervention_type,
                    operating_hours,
                    activities,
                    anomalies,
                    recommendations,
                    compressor_id
                FROM maintenance
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        logger.error(f"Error loading maintenance records: {str(e)}")
        return pd.DataFrame()

def get_historical_failures(db_path, start_date, end_date):
    """Get failure records between dates from the database"""
    try:
        with sqlite3.connect(db_path) as conn:
            # First check if table exists
            check_query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='failures';
            """
            table_exists = pd.read_sql_query(check_query, conn)
            
            if table_exists.empty:
                logger.error("The 'failures' table does not exist in the database")
                return pd.DataFrame()
            
            # Check the actual schema
            schema_query = "PRAGMA table_info(failures)"
            schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Failures table columns: {schema['name'].tolist()}")
            
            # Updated query to match the new schema
            query = """
                SELECT 
                    date,
                    failure_type,
                    frequency,
                    cause,
                    solution,
                    additional_info,
                    feedback,
                    compressor_id
                FROM failures
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            logger.info(f"Found {len(df)} failures between {start_date} and {end_date}")
            if not df.empty:
                logger.info(f"Sample failure: {df.iloc[0].to_dict()}")
                
            return df
            
    except Exception as e:
        logger.error(f"Error loading failures: {str(e)}")
        logger.exception("Full traceback:")  # Show full traceback
        return pd.DataFrame()

def get_historical_workload(db_path, start_date, end_date):
    """Get workload data between dates"""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT 
                    date,
                    start_time,
                    end_time,
                    operating_hours,
                    load_percentage,
                    temperature,
                    humidity,
                    vibration,
                    pressure,
                    compressor_id
                FROM operating_conditions
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        logger.error(f"Error loading operational load: {str(e)}")
        return pd.DataFrame()

def get_historical_feedback(db_path, start_date, end_date):
    """Get user feedback between dates"""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
                SELECT 
                    date,
                    feedback_type,
                    title,
                    description,
                    rating,
                    comment,
                    compressor_id
                FROM feedback
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=[start_date, end_date],
                parse_dates=['date']
            )
            
            return df
            
    except Exception as e:
        logger.error(f"Error loading feedback: {str(e)}")
        return pd.DataFrame()

async def analyze_anomaly(assistant, timestamp, compressor_id):
    """Analyze a specific anomaly using the LLM assistant"""
    try:
        # Ensure timestamp is in correct format
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        analysis = await assistant.analyze_anomaly(timestamp, compressor_id)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing anomaly: {str(e)}")
        return f"Failed to analyze anomaly: {str(e)}"

def get_compressor_status(db_path, compressor_id, current_time):
    """Get current status of a compressor at specified time"""
    try:
        # Check for anomalies in the last 5 minutes
        current_anomalies = get_anomalies(
            db_path,
            compressor_id,
            current_time - timedelta(minutes=5),
            current_time
        )
        
        if current_anomalies is not None:
            true_anomalies = current_anomalies[current_anomalies['is_anomaly'] == True]
            has_anomalies = not true_anomalies.empty
            anomaly_count = len(true_anomalies) if has_anomalies else 0
        else:
            has_anomalies = False
            anomaly_count = 0
            
        return {
            'has_anomalies': has_anomalies,
            'anomaly_count': anomaly_count
        }
    except Exception as e:
        logger.error(f"Error getting compressor status: {str(e)}")
        return {'has_anomalies': False, 'anomaly_count': 0}

def get_current_environmental_values(db_path, timestamp):
    """Get environmental values for the exact simulated timestamp"""
    try:
        with sqlite3.connect(db_path) as conn:
            # First verify table structure
            schema_query = "PRAGMA table_info(weather_data)"
            schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Weather table schema: {schema['name'].tolist()}")
            
            # Check what data we have for this date
            all_columns = ", ".join(schema['name'].tolist())
            debug_query = f"""
                SELECT {all_columns}
                FROM weather_data 
                WHERE date = DATE(?)
            """
            debug_df = pd.read_sql_query(debug_query, conn, params=[timestamp])
            logger.info(f"DEBUG - Weather data for {timestamp.date()}: Found {len(debug_df)} records")
            
            if not debug_df.empty:
                logger.info(f"Sample record: {debug_df.iloc[0].to_dict()}")
                
                # Map correct column names
                column_mappings = {
                    'avg_temp': 'temperature',
                    'humidity': 'humidity',
                    'avg_pressure': 'pressure',
                    'TMEDIA °C': 'temperature',
                    'UMIDITA %': 'humidity',
                    'PRESSIONEMEDIA mb': 'pressure'
                }
                
                # Check which columns are actually present
                result = {}
                for db_col, api_col in column_mappings.items():
                    if db_col in debug_df.columns:
                        result[api_col] = debug_df.iloc[0][db_col]
                    
                # Ensure all required fields are present
                for field in ['temperature', 'humidity', 'pressure']:
                    if field not in result:
                        logger.warning(f"Field {field} not found in weather_data table")
                        result[field] = None
                
                logger.info(f"Mapped result: {result}")
                return result
            else:
                logger.warning(f"No weather data found for date {timestamp.date()}")
                return {
                    'temperature': None, 
                    'humidity': None, 
                    'pressure': None
                }
                
    except Exception as e:
        logger.error(f"Error getting environmental values: {str(e)}")
        return {'temperature': None, 'humidity': None, 'pressure': None}

def get_current_operational_values(db_path, timestamp):
    """Get operational values for the exact simulated timestamp"""
    try:
        with sqlite3.connect(db_path) as conn:
            # First verify table structure
            schema_query = "PRAGMA table_info(operating_conditions)"
            schema = pd.read_sql_query(schema_query, conn)
            logger.info(f"Operating conditions table schema: {schema['name'].tolist()}")
            
            # Check what data we have for this date
            all_columns = ", ".join(schema['name'].tolist())
            debug_query = f"""
                SELECT {all_columns}
                FROM operating_conditions 
                WHERE date = DATE(?)
            """
            debug_df = pd.read_sql_query(debug_query, conn, params=[timestamp])
            logger.info(f"DEBUG - Operating conditions for {timestamp.date()}: Found {len(debug_df)} records")
            
            if not debug_df.empty:
                logger.info(f"Sample record: {debug_df.iloc[0].to_dict()}")
                
                # Map correct column names
                column_mappings = {
                    'load_percentage': 'load_percentage',
                    'operating_hours': 'operating_hours',
                    'Carico_Operativo': 'load_percentage',
                    'Ore_Funzionamento': 'operating_hours'
                }
                
                # Check which columns are actually present
                result = {}
                for db_col, api_col in column_mappings.items():
                    if db_col in debug_df.columns:
                        result[api_col] = debug_df.iloc[0][db_col]
                
                # Ensure all required fields are present
                for field in ['load_percentage', 'operating_hours']:
                    if field not in result:
                        logger.warning(f"Field {field} not found in operating_conditions table")
                        result[field] = None
                
                logger.info(f"Mapped result: {result}")
                return result
            else:
                logger.warning(f"No operating data found for date {timestamp.date()}")
                return {
                    'load_percentage': None, 
                    'operating_hours': None
                }
                
    except Exception as e:
        logger.error(f"Error getting operational values: {str(e)}")
        return {'load_percentage': None, 'operating_hours': None}

def analyze_prediction_impact(results):
    """Analyze and format the impact of predicted variations"""
    try:
        impact_text = []
        
        for param, diff in results['differences'].items():
            param_name = param.replace('_', ' ').title()
            if abs(diff) > 0:
                change = "increase" if diff > 0 else "decrease"
                percentage = (diff / results['base_prediction'][param]) * 100
                impact_text.append(
                    f"**{param_name}**: {abs(percentage):.1f}% {change}\n"
                    f"Impact: {'High' if abs(percentage) > 10 else 'Medium' if abs(percentage) > 5 else 'Low'}"
                )
        
        return "\n\n".join(impact_text) if impact_text else "No significant variations predicted"
        
    except Exception as e:
        logger.error(f"Error in impact analysis: {str(e)}")
        return "Error in impact analysis"