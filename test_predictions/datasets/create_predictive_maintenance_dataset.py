import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3
import os

# Set random seed for reproducibility
np.random.seed(42)

# Get the correct file paths using relative paths from the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Go up two levels to reach code 10 directory

# Define paths for required files
weather_data_path = os.path.join(base_dir, 'weather_data', 'halfhourly_weather_data_db_optimized.csv')
may_data_path = os.path.join(base_dir, 'compressore_1', 'compressore1_predictions_onlymay.xlsx')

print(f"Base directory: {base_dir}")
print(f"Weather data path: {weather_data_path}")
print(f"May data path: {may_data_path}")

# Define date range (March 1, 2024 - May 31, 2024)
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 5, 31, 23, 0)
date_range = pd.date_range(start=start_date, end=end_date, freq='h')

# Create empty DataFrame
df = pd.DataFrame(index=date_range)
df.index.name = 'DateTime'

# Import weather data from halfhourly_weather_data_db_optimized.csv
try:
    print("Loading weather data...")
    if os.path.exists(weather_data_path):
        # Load weather data
        weather_data = pd.read_csv(weather_data_path)
        
        # Convert timestamp to datetime format - use the datetime column that already exists
        # This must match our expected date format - we'll use the 'datetime' column as it's already in the correct format
        weather_data['timestamp'] = pd.to_datetime(weather_data['datetime'])
        weather_data.set_index('timestamp', inplace=True)
        
        # Print some debugging information
        print(f"Weather data loaded with {len(weather_data)} records")
        print(f"Weather data date range: {weather_data.index.min()} to {weather_data.index.max()}")
        
        # Ensure we're working with hourly data by resampling to hourly frequency
        # Use proper aggregation methods for each type of data
        weather_hourly = weather_data.resample('H').agg({
            'temperature': 'mean',
            'humidity': 'mean', 
            'wind_speed': 'mean',
            'precipitation': 'sum',  # Sum precipitation within the hour
            'condition': 'first'     # Take first condition in the hour
        })
        
        print(f"After resampling to hourly: {len(weather_hourly)} records")
        
        # Reindex to match our date range exactly
        weather_reindex = weather_hourly.reindex(date_range)
        
        # For any missing dates, interpolate or forward fill
        weather_reindex = weather_reindex.interpolate()
        weather_reindex = weather_reindex.ffill().bfill()
        
        print(f"After reindexing to match our date range: {len(weather_reindex)} records")
        
        # Map the weather data to our main dataframe with the correct column names
        df['weather_temperature'] = weather_reindex['temperature']
        df['weather_humidity'] = weather_reindex['humidity']
        df['weather_wind_speed'] = weather_reindex['wind_speed']
        df['weather_precipitation'] = weather_reindex['precipitation']
        df['weather_condition'] = weather_reindex['condition']
        
        # Print sample to verify data was mapped correctly
        print("\nSample of weather data in main dataframe:")
        print(df[['weather_temperature', 'weather_humidity', 'weather_wind_speed', 
                 'weather_precipitation', 'weather_condition']].head())
                 
        print(f"Successfully loaded weather data with {len(weather_reindex)} records")
    else:
        print(f"Weather data file not found at: {weather_data_path}")
        raise FileNotFoundError(f"Weather data file not found at: {weather_data_path}")
except Exception as e:
    print(f"Could not load or process weather data: {e}")
    # Create placeholder weather data if file isn't available
    print("Using synthetic weather data instead...")
    df['weather_temperature'] = np.random.normal(15, 5, len(df))
    df['weather_humidity'] = np.random.normal(75, 15, len(df))
    df['weather_wind_speed'] = np.random.normal(10, 5, len(df))
    df['weather_precipitation'] = np.random.exponential(0.1, len(df))
    df['weather_condition'] = np.random.choice(['Clear', 'Cloudy', 'Rain', 'Storm'], len(df))

# Define normal operating parameters based on the compressor specifications
# Normal operation ranges (mean, std)
param_ranges = {
    'Current': (94, 8),               # Amps
    'CosPhi': (0.88, 0.02),           # Power factor
    'Energy_Consumption': (55, 5),    # kWh (nominal power)
    'Reactive_Energy': (15, 3),       # kVArh
    'Voltage': (400, 3),              # V (400V specified)
    'Vibration': (1.8, 0.2),          # mm/s
    'Temperature': (85, 5),           # °C (normal is 70-95°C)
    'Pressure': (7.2, 0.2),           # bar (normal is 6-7.5 bar)
    'Speed': (2950, 20)               # RPM (nominal speed)
}

# Generate base values for normal operation
for param, (mean, std) in param_ranges.items():
    df[param] = np.random.normal(mean, std, len(df))

# Add daily patterns (higher during working hours)
hour_factors = {
    'Current': np.array([0.85, 0.83, 0.82, 0.80, 0.80, 0.85, 0.95, 1.05, 1.12, 1.15, 1.18, 1.15, 
                        1.13, 1.15, 1.18, 1.14, 1.10, 1.05, 1.0, 0.95, 0.92, 0.90, 0.88, 0.86]),
    'Energy_Consumption': np.array([0.82, 0.80, 0.78, 0.75, 0.75, 0.80, 0.90, 1.05, 1.15, 1.20, 1.22, 1.20, 
                                  1.18, 1.20, 1.22, 1.18, 1.15, 1.10, 1.0, 0.95, 0.90, 0.88, 0.86, 0.84]),
    'Temperature': np.array([0.95, 0.93, 0.92, 0.90, 0.90, 0.93, 0.97, 1.02, 1.05, 1.07, 1.08, 1.07, 
                           1.06, 1.07, 1.08, 1.07, 1.05, 1.03, 1.0, 0.98, 0.97, 0.96, 0.96, 0.95]),
}

for param, factors in hour_factors.items():
    for hour in range(24):
        mask = df.index.hour == hour
        df.loc[mask, param] = df.loc[mask, param] * factors[hour]

# Add weekly pattern (less usage on weekends)
weekend_factors = {
    'Current': 0.75,
    'Energy_Consumption': 0.70,
    'Temperature': 0.90,
    'Pressure': 0.95,
    'Speed': 0.90,
}

for param, factor in weekend_factors.items():
    weekend_mask = df.index.dayofweek.isin([5, 6])  # Saturday and Sunday
    df.loc[weekend_mask, param] = df.loc[weekend_mask, param] * factor

# Add gradual drift to certain parameters (slight wear over time)
days = (df.index - df.index[0]).days.values / 90  # Normalized time (0 to 1)
df['Vibration'] = df['Vibration'] * (1 + days * 0.1)  # Vibration increases by up to 10% over 3 months
df['Temperature'] = df['Temperature'] * (1 + days * 0.05)  # Temperature increases by up to 5%

# Weather effects on compressor operation
if 'weather_temperature' in df.columns:
    # High ambient temperature increases compressor temperature
    high_ambient_mask = df['weather_temperature'] > 30
    df.loc[high_ambient_mask, 'Temperature'] = df.loc[high_ambient_mask, 'Temperature'] * 1.08
    df.loc[high_ambient_mask, 'Current'] = df.loc[high_ambient_mask, 'Current'] * 1.05
    df.loc[high_ambient_mask, 'CosPhi'] = df.loc[high_ambient_mask, 'CosPhi'] * 0.97
    
    # Very low ambient temperature affects start-up
    low_ambient_mask = df['weather_temperature'] < 3
    morning_hours_mask = df.index.hour.isin([5, 6, 7, 8])
    cold_morning_mask = low_ambient_mask & morning_hours_mask
    df.loc[cold_morning_mask, 'Current'] = df.loc[cold_morning_mask, 'Current'] * 1.15
    df.loc[cold_morning_mask, 'CosPhi'] = df.loc[cold_morning_mask, 'CosPhi'] * 0.92
    
    # High humidity affects efficiency
    if 'weather_humidity' in df.columns:
        high_humidity_mask = df['weather_humidity'] > 90
        df.loc[high_humidity_mask, 'Current'] = df.loc[high_humidity_mask, 'Current'] * 1.03
        df.loc[high_humidity_mask, 'Pressure'] = df.loc[high_humidity_mask, 'Pressure'] * 0.98
    
    # Storm conditions might cause voltage fluctuations
    if 'weather_condition' in df.columns:
        # Ensure the column is string type before using str accessor
        df['weather_condition'] = df['weather_condition'].astype(str)
        storm_mask = df['weather_condition'].str.contains('Storm|Thunder', na=False)
        if storm_mask.any():
            df.loc[storm_mask, 'Voltage'] = df.loc[storm_mask, 'Voltage'] * np.random.choice([0.97, 0.98, 1.02, 1.03], size=storm_mask.sum())
            df.loc[storm_mask, 'Current'] = df.loc[storm_mask, 'Current'] * np.random.choice([0.95, 0.97, 1.03, 1.05], size=storm_mask.sum())

# Determine Dependencies between parameters
df['CosPhi'] = df['CosPhi'] - (df['Temperature'] - param_ranges['Temperature'][0]) * 0.0008
df['Pressure'] = df['Pressure'] + (df['Speed'] - param_ranges['Speed'][0]) * 0.0005
df['Reactive_Energy'] = df['Energy_Consumption'] * (1 - df['CosPhi']) * 0.9

# Add columns for anomaly tracking
df['Anomaly'] = False
df['Anomaly_Detected'] = False
df['Notes'] = ''

# Define function to create an anomaly
def create_anomaly(date_time, anomaly_type, severity_factor=1.5):
    df.loc[date_time, 'Anomaly'] = True
    df.loc[date_time, 'Anomaly_Detected'] = True
    notes = f"Anomaly: {anomaly_type}"
    df.loc[date_time, 'Notes'] = notes
    
    if anomaly_type == "Bearing Failure":
        df.loc[date_time, 'Vibration'] *= 2.5 * severity_factor
        df.loc[date_time, 'Temperature'] += 15 * severity_factor
        df.loc[date_time, 'Current'] += 10 * severity_factor
        df.loc[date_time, 'Speed'] *= 0.97
    
    elif anomaly_type == "Overheating":
        df.loc[date_time, 'Temperature'] = 116 * severity_factor  # Above critical 115°C
        df.loc[date_time, 'Current'] *= 1.3 * severity_factor
        df.loc[date_time, 'CosPhi'] -= 0.08 * severity_factor
    
    elif anomaly_type == "Pressure Drop":
        df.loc[date_time, 'Pressure'] = 5.2 * (1/severity_factor)  # Below minimum 5.5 bar
        df.loc[date_time, 'Energy_Consumption'] *= 1.2 * severity_factor
    
    elif anomaly_type == "Motor Imbalance":
        df.loc[date_time, 'Vibration'] *= 2.2 * severity_factor
        df.loc[date_time, 'Current'] *= 1.15 * severity_factor
        df.loc[date_time, 'Speed'] *= 0.95
    
    elif anomaly_type == "Voltage Fluctuation":
        df.loc[date_time, 'Voltage'] *= np.random.choice([0.85, 1.15])
        df.loc[date_time, 'CosPhi'] -= 0.05 * severity_factor

# Define function to create a false positive
def create_false_positive(date_time, false_positive_type):
    df.loc[date_time, 'Anomaly_Detected'] = True
    df.loc[date_time, 'Notes'] = f"False Positive: {false_positive_type}"
    
    if false_positive_type == "Temporary Vibration":
        df.loc[date_time, 'Vibration'] *= 1.8
    
    elif false_positive_type == "Brief Current Spike":
        df.loc[date_time, 'Current'] *= 1.25
    
    elif false_positive_type == "Temperature Warning":
        df.loc[date_time, 'Temperature'] = 109  # High but below critical 115°C
    
    elif false_positive_type == "Minor Pressure Fluctuation":
        df.loc[date_time, 'Pressure'] *= np.random.choice([0.92, 1.08])
    
    elif false_positive_type == "Speed Variation":
        df.loc[date_time, 'Speed'] *= np.random.choice([0.93, 1.07])
    
    elif false_positive_type == "Weather-Related High Current":
        df.loc[date_time, 'Current'] *= 1.3
        df.loc[date_time, 'CosPhi'] *= 0.95
        
    elif false_positive_type == "Storm-Induced Voltage Spike":
        df.loc[date_time, 'Voltage'] *= 1.1
        
    elif false_positive_type == "High Ambient Temperature Effect":
        df.loc[date_time, 'Temperature'] = 105  # High due to ambient but not an anomaly

# Schedule anomalies (exactly 11)
anomaly_types = ["Bearing Failure", "Overheating", "Pressure Drop", "Motor Imbalance", "Voltage Fluctuation"]

# Define exact dates for anomalies (2024)
anomaly_dates = [
    pd.Timestamp('2024-03-08 14:00:00'),  # Friday
    pd.Timestamp('2024-03-23 09:00:00'),  # Saturday
    pd.Timestamp('2024-04-05 11:00:00'),  # Friday
    pd.Timestamp('2024-04-14 16:00:00'),  # Sunday
    pd.Timestamp('2024-04-20 10:00:00'),  # Saturday
    pd.Timestamp('2024-04-29 13:00:00'),  # Monday
    pd.Timestamp('2024-05-07 15:00:00'),  # Tuesday
    pd.Timestamp('2024-05-16 09:00:00'),  # Thursday
    pd.Timestamp('2024-05-22 17:00:00'),  # Wednesday
    pd.Timestamp('2024-05-26 14:00:00'),  # Sunday
    pd.Timestamp('2024-05-30 11:00:00')   # Thursday
]

# Create anomalies
for i, date_time in enumerate(anomaly_dates):
    anomaly_type = anomaly_types[i % len(anomaly_types)]
    severity = random.uniform(1.1, 1.5)
    create_anomaly(date_time, anomaly_type, severity)

# Schedule false positives (exactly 17)
false_positive_types = [
    "Temporary Vibration", 
    "Brief Current Spike", 
    "Temperature Warning",
    "Minor Pressure Fluctuation", 
    "Speed Variation",
    "Weather-Related High Current",
    "Storm-Induced Voltage Spike",
    "High Ambient Temperature Effect"
]

# Define exact dates for false positives (2024)
false_positive_dates = [
    pd.Timestamp('2024-03-05 10:00:00'),  # Tuesday - Weather-related
    pd.Timestamp('2024-03-12 13:00:00'),  # Tuesday
    pd.Timestamp('2024-03-18 16:00:00'),  # Monday
    pd.Timestamp('2024-03-25 09:00:00'),  # Monday
    pd.Timestamp('2024-03-30 14:00:00'),  # Saturday - Weather-related
    pd.Timestamp('2024-04-02 11:00:00'),  # Tuesday
    pd.Timestamp('2024-04-09 15:00:00'),  # Tuesday - Weather-related
    pd.Timestamp('2024-04-16 10:00:00'),  # Tuesday
    pd.Timestamp('2024-04-22 13:00:00'),  # Monday
    pd.Timestamp('2024-04-25 16:00:00'),  # Thursday - Weather-related
    pd.Timestamp('2024-04-30 09:00:00'),  # Tuesday
    pd.Timestamp('2024-05-03 12:00:00'),  # Friday
    pd.Timestamp('2024-05-10 14:00:00'),  # Friday - Weather-related
    pd.Timestamp('2024-05-15 11:00:00'),  # Wednesday
    pd.Timestamp('2024-05-20 16:00:00'),  # Monday
    pd.Timestamp('2024-05-25 10:00:00'),  # Saturday - Weather-related
    pd.Timestamp('2024-05-31 08:00:00')   # Friday
]

# Create false positives
for i, date_time in enumerate(false_positive_dates):
    # Use weather-related false positives for specific days
    if i in [0, 4, 6, 9, 12, 15]:  # Weather-related dates
        fp_type = random.choice(["Weather-Related High Current", "Storm-Induced Voltage Spike", "High Ambient Temperature Effect"])
    else:
        fp_type = random.choice(false_positive_types[:5])  # Non-weather related
    create_false_positive(date_time, fp_type)

# Load May data from the Excel file
try:
    print("Loading May data from Excel...")
    if os.path.exists(may_data_path):
        may_data = pd.read_excel(may_data_path)
        
        # Print the first few rows to check
        print(f"May data preview:")
        print(may_data.head())
        
        # First, handle the datetime column which is 'Unnamed: 0'
        if 'Unnamed: 0' in may_data.columns and pd.api.types.is_datetime64_any_dtype(may_data['Unnamed: 0']):
            print("Found datetime column as 'Unnamed: 0', renaming to 'DateTime'")
            may_data = may_data.rename(columns={'Unnamed: 0': 'DateTime'})
            may_data.set_index('DateTime', inplace=True)
            
            # Define column mapping from Excel to our dataset
            column_mapping = {
                'Current (A)': 'Current',
                'CosPhi (Units)': 'CosPhi',
                'Energy Consumption (kWh)': 'Energy_Consumption',
                'Reactive Energy (VARh)': 'Reactive_Energy',
                'Voltage (V)': 'Voltage'
            }
            
            # Rename columns to match our dataset
            may_data = may_data.rename(columns=column_mapping)
            
            # Make sure we only use dates in May 2024
            print(f"Original year in May data: {may_data.index.year[0]}")
            if may_data.index.year[0] != 2024:
                print(f"Updating year to 2024 in May data")
                may_data.index = may_data.index.map(lambda x: x.replace(year=2024))
            
            # Filter to just May data
            may_filter = (may_data.index >= '2024-05-01') & (may_data.index <= '2024-05-31 23:59:59')
            may_filtered = may_data[may_filter]
            
            print(f"Found {len(may_filtered)} records in May data")
            
            # Replace our generated data with the real data for May
            for col in may_filtered.columns:
                if col in df.columns:
                    # Find matching dates in our dataframe
                    common_dates = df.index.intersection(may_filtered.index)
                    
                    if len(common_dates) > 0:
                        print(f"Replacing {len(common_dates)} records for column {col} with actual May data")
                        df.loc[common_dates, col] = may_filtered.loc[common_dates, col]
        else:
            print(f"First column is not a datetime or is not named 'Unnamed: 0'")
    else:
        print(f"May data file not found at: {may_data_path}")
except Exception as e:
    print(f"Could not load May data from Excel: {e}")
    print("Continuing with synthetic data for May")

# Apply some reasonable limits and cleanup based on compressor specs
df['CosPhi'] = df['CosPhi'].clip(0.75, 0.98)
df['Vibration'] = df['Vibration'].clip(0.5, 15)
df['Temperature'] = df['Temperature'].clip(50, 120)  # Based on specs (normal 70-95°C, critical >115°C)
df['Pressure'] = df['Pressure'].clip(4, 8.5)  # Based on specs (max 8 bar)
df['Current'] = df['Current'].clip(50, 150)  # Reasonable range for a 55kW compressor
df['Voltage'] = df['Voltage'].clip(380, 420)  # Based on 400V nominal with ±5% tolerance
df['Speed'] = df['Speed'].clip(2800, 3100)  # Around nominal 2950 rpm

# Round values to reasonable precision
df['Current'] = df['Current'].round(2)
df['CosPhi'] = df['CosPhi'].round(3)
df['Energy_Consumption'] = df['Energy_Consumption'].round(2)
df['Reactive_Energy'] = df['Reactive_Energy'].round(2)
df['Voltage'] = df['Voltage'].round(1)
df['Vibration'] = df['Vibration'].round(2)
df['Temperature'] = df['Temperature'].round(1)
df['Pressure'] = df['Pressure'].round(2)
df['Speed'] = df['Speed'].round(0)
if 'weather_temperature' in df.columns:
    df['weather_temperature'] = df['weather_temperature'].round(1)
if 'weather_humidity' in df.columns:
    df['weather_humidity'] = df['weather_humidity'].round(0)
if 'weather_wind_speed' in df.columns:
    df['weather_wind_speed'] = df['weather_wind_speed'].round(1)
if 'weather_precipitation' in df.columns:
    df['weather_precipitation'] = df['weather_precipitation'].round(2)

# Create a new dataframe that includes both operating parameters and weather data
df_with_weather = df.copy()

# Reset index to have DateTime as a column for both dataframes
df_clean = df.copy().reset_index()
df_with_weather = df_with_weather.reset_index()

# Save output files to the script's directory
output_dir = script_dir
print(f"Saving output files to: {output_dir}")

# Save as CSV files
df_clean.to_csv(os.path.join(output_dir, 'compressor_dataset_2024.csv'), index=False)
df_with_weather.to_csv(os.path.join(output_dir, 'compressor_dataset_with_weather_2024.csv'), index=False)

# Save as Excel files
df_clean.to_excel(os.path.join(output_dir, 'compressor_dataset_2024.xlsx'), index=False)
df_with_weather.to_excel(os.path.join(output_dir, 'compressor_dataset_with_weather_2024.xlsx'), index=False)

# Create dataframes for unlabeled data (to be used by the LLM)
df_unlabeled = df_clean.drop(columns=['Anomaly', 'Anomaly_Detected', 'Notes'])
df_with_weather_unlabeled = df_with_weather.drop(columns=['Anomaly', 'Anomaly_Detected', 'Notes'])

# -----------------------------------------------------------------------------
# MODIFIED PART: Create two separate database files - one with labels, one without
# -----------------------------------------------------------------------------

# 1. Create database WITHOUT labels (for LLM analysis)
print("Creating SQLite database WITHOUT labels (for LLM)...")
db_path = os.path.join(output_dir, 'compressor_data_2024.db')

# Remove file if it exists
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to SQLite database
conn = sqlite3.connect(db_path)

# Create tables without label columns
df_unlabeled.to_sql('compressor_data', conn, index=False)
df_with_weather_unlabeled.to_sql('compressor_data_with_weather', conn, index=False)

# Create a table with just the weather data
weather_cols = ['DateTime'] + [col for col in df_with_weather.columns if col.startswith('weather_')]
weather_data = df_with_weather[weather_cols].copy()
weather_data.to_sql('weather_data', conn, index=False)

# Close connection to the unlabeled database
conn.close()
print(f"Database without labels created: {db_path}")

# 2. Create database WITH labels (for evaluation)
print("Creating SQLite database WITH labels (for evaluation)...")
db_path_labeled = os.path.join(output_dir, 'compressor_data_2024_etichettato.db')

# Remove file if it exists
if os.path.exists(db_path_labeled):
    os.remove(db_path_labeled)

# Connect to SQLite database
conn_labeled = sqlite3.connect(db_path_labeled)

# Create tables with all columns including labels
df_clean.to_sql('compressor_data', conn_labeled, index=False)
df_with_weather.to_sql('compressor_data_with_weather', conn_labeled, index=False)

# Create a table for anomalies for easier querying
anomalies = df_clean[df_clean['Anomaly']].copy()
anomalies.to_sql('anomalies', conn_labeled, index=False)

# Create a table for false positives
false_positives = df_clean[(df_clean['Anomaly_Detected']) & (~df_clean['Anomaly'])].copy()
false_positives.to_sql('false_positives', conn_labeled, index=False)

# Create a table with just the weather data (same as in the unlabeled db)
weather_data.to_sql('weather_data', conn_labeled, index=False)

# Create a reference table that maps timestamps to their true status
reference_data = df_clean[['DateTime', 'Anomaly', 'Anomaly_Detected', 'Notes']].copy()
reference_data.to_sql('reference_data', conn_labeled, index=False)

# Close connection to the labeled database
conn_labeled.close()
print(f"Database with labels created: {db_path_labeled}")

print("\nDataset created with:")
print(f"Total records: {len(df_clean)}")
print(f"True anomalies: {df_clean['Anomaly'].sum()}")
print(f"False positives: {df_clean['Anomaly_Detected'].sum() - df_clean[df_clean['Anomaly'] & df_clean['Anomaly_Detected']].shape[0]}")
print(f"Files saved to: {output_dir}")
print(f"Files saved:")
print(f"1. CSVs/Excel: compressor_dataset_2024.csv, compressor_dataset_2024.xlsx, compressor_dataset_with_weather_2024.csv, compressor_dataset_with_weather_2024.xlsx")
print(f"2. Databases: compressor_data_2024.db (without labels), compressor_data_2024_etichettato.db (with labels)")

# Create a summary of anomalies and false positives
anomalies = df_clean[df_clean['Anomaly']].copy()
false_positives = df_clean[(df_clean['Anomaly_Detected']) & (~df_clean['Anomaly'])].copy()

print("\nSummary of Anomalies:")
for i, row in anomalies.iterrows():
    print(f"{row['DateTime']} - {row['Notes']}")

print("\nSummary of False Positives:")
for i, row in false_positives.iterrows():
    print(f"{row['DateTime']} - {row['Notes']}")