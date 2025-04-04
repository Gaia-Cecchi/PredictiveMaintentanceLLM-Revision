import os
import pandas as pd
import re
from datetime import datetime
import codecs
import json

# Enhanced function to clean temperature values and other values with special characters
def clean_special_chars(value):
    if pd.isna(value) or value is None:
        return value
        
    # Convert to string if not already
    value = str(value)
    
    # Direct replacement of the problematic character
    value = value.replace('Â', '')
    
    # Clean up temperature format
    value = value.replace(' °C', '°C')
    # Fix multiple spaces
    value = re.sub(r'\s+', ' ', value).strip()
    
    return value

# Function to standardize datetime format to YYYY-MM-DD HH:mm:ss
def standardize_datetime(df):
    if 'Date' in df.columns and 'Time' in df.columns:
        try:
            print("Standardizing datetime format to YYYY-MM-DD HH:mm:ss...")
            
            # Create a standardized DateTime column
            combined_dt = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
            
            # Handle any parsing errors by trying alternative formats
            mask_nat = combined_dt.isna()
            if mask_nat.any():
                alt_formats = ['%Y-%m-%d %I:%M %p', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']
                for fmt in alt_formats:
                    try:
                        parsed = pd.to_datetime(df.loc[mask_nat, 'Date'] + ' ' + df.loc[mask_nat, 'Time'], 
                                             format=fmt, errors='coerce')
                        combined_dt.loc[mask_nat] = parsed
                        mask_nat = combined_dt.isna()
                        if not mask_nat.any():
                            break
                    except:
                        continue
            
            # Format to standard ISO format without timezone
            df['Time'] = combined_dt.dt.strftime('%H:%M:%S')
            
            # Create a standard datetime column for reference
            df['DateTime'] = combined_dt.dt.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Datetime standardization complete. Sample: {df['DateTime'].iloc[0] if not df.empty else 'No data'}")
            
        except Exception as e:
            print(f"Error during datetime standardization: {e}")
    
    return df

# Function to prepare a Supabase-friendly version of the data
def prepare_supabase_version(df):
    # Make a copy of the dataframe to avoid modifying the original
    supabase_df = df.copy()
    
    print("Preparing Supabase-friendly version...")
    
    # Convert temperature values: replace "°C" with "celsius degrees"
    temp_columns = ['Temperature', 'Dew Point']
    for col in temp_columns:
        if col in supabase_df.columns:
            supabase_df[col] = supabase_df[col].astype(str)
            supabase_df[col] = supabase_df[col].str.replace('°C', ' celsius degrees', regex=False)
    
    # Standardize column names - replace spaces and special chars with underscores
    supabase_df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col).lower() for col in supabase_df.columns]
    
    # Ensure all text fields are valid strings (no NaN)
    for col in supabase_df.columns:
        if supabase_df[col].dtype == object:
            supabase_df[col] = supabase_df[col].fillna('')
            
    print("Supabase version prepared with the following columns:")
    print(", ".join(supabase_df.columns))
    
    return supabase_df

# Function to prepare a database-optimized version of the data for LLM usage
def prepare_db_optimized_version(df):
    """
    Create a database-optimized version of the weather data with the following features:
    1. Clean, consistent column names (snake_case)
    2. Proper data types
    3. Structured datetime format (ISO 8601)
    4. Numeric values without units for easier calculations
    5. Categorized weather conditions
    6. Metadata to help LLMs understand the data
    """
    print("Preparing database-optimized version for LLM usage...")
    
    # Make a deep copy to avoid modifying the original
    db_df = df.copy()
    
    # 1. Standardize column names to snake_case
    db_df.columns = [re.sub(r'[^a-zA-Z0-9]', '_', col).lower().strip('_') for col in db_df.columns]
    
    # 2. Clean data types and extract numeric values
    
    # Process temperature columns - extract numeric values
    temp_columns = ['temperature', 'dew_point'] 
    for col in temp_columns:
        if col in db_df.columns:
            # Extract numeric values and convert to float
            db_df[col] = db_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').astype(float)
    
    # Process humidity - extract percentage as numeric
    if 'humidity' in db_df.columns:
        db_df['humidity'] = db_df['humidity'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Process wind speed and gust - extract numeric values
    wind_columns = ['wind_speed', 'wind_gust']
    for col in wind_columns:
        if col in db_df.columns:
            db_df[col] = db_df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Process pressure - extract numeric values
    if 'pressure' in db_df.columns:
        db_df['pressure'] = db_df['pressure'].astype(str).str.extract(r'([\d\,\.]+)').astype(str)
        db_df['pressure'] = db_df['pressure'].str.replace(',', '').astype(float)
    
    # Process precipitation - extract numeric values
    if 'precip' in db_df.columns or 'precip_' in db_df.columns:
        precip_col = 'precip' if 'precip' in db_df.columns else 'precip_'
        db_df['precipitation'] = db_df[precip_col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        if precip_col != 'precipitation':
            db_df.drop(columns=[precip_col], inplace=True)
    
    # 3. Create a proper datetime field in ISO format
    if 'date' in db_df.columns and 'time' in db_df.columns:
        # Create standardized timestamp
        db_df['timestamp'] = pd.to_datetime(db_df['date'] + ' ' + db_df['time'], errors='coerce')
        db_df['timestamp'] = db_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Extract date components for easier filtering
        datetime_obj = pd.to_datetime(db_df['timestamp'])
        db_df['year'] = datetime_obj.dt.year
        db_df['month'] = datetime_obj.dt.month
        db_df['day'] = datetime_obj.dt.day
        db_df['hour'] = datetime_obj.dt.hour
        db_df['minute'] = datetime_obj.dt.minute
        db_df['dayofweek'] = datetime_obj.dt.dayofweek  # 0 = Monday, 6 = Sunday
        db_df['is_weekend'] = db_df['dayofweek'].isin([5, 6]).astype(int)  # 1 for weekend, 0 for weekday
    
    # 4. Process wind direction into categories and degrees
    if 'wind' in db_df.columns:
        # Extract wind direction abbreviation
        db_df['wind_direction'] = db_df['wind'].astype(str).str.extract(r'([A-Z]+)')[0]
        
        # Map wind directions to degrees (approximate)
        direction_to_degrees = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        db_df['wind_degrees'] = db_df['wind_direction'].map(direction_to_degrees)
        
    # 5. Categorize weather conditions
    if 'condition' in db_df.columns:
        # Define weather condition categories
        sunny_patterns = ['sunny', 'clear', 'fair']
        cloudy_patterns = ['cloudy', 'overcast', 'partly cloudy', 'mostly cloudy']
        rainy_patterns = ['rain', 'shower', 'drizzle', 'thunderstorm']
        snowy_patterns = ['snow', 'sleet', 'hail', 'ice']
        foggy_patterns = ['fog', 'mist', 'haze']
        
        # Create condition category
        db_df['condition_lower'] = db_df['condition'].astype(str).str.lower()
        
        # Apply categorization
        def categorize_condition(condition):
            if any(p in condition for p in sunny_patterns):
                return 'sunny'
            elif any(p in condition for p in cloudy_patterns):
                return 'cloudy'
            elif any(p in condition for p in rainy_patterns):
                return 'rainy'
            elif any(p in condition for p in snowy_patterns):
                return 'snowy'
            elif any(p in condition for p in foggy_patterns):
                return 'foggy'
            else:
                return 'other'
                
        db_df['weather_category'] = db_df['condition_lower'].apply(categorize_condition)
        db_df.drop(columns=['condition_lower'], inplace=True)
    
    # 6. Add metadata as separate JSON file
    metadata = {
        "data_source": "Weather Underground (wunderground.com)",
        "location": "Pisa, Italy (LIRP)",
        "time_period": f"{db_df['year'].min()}-{db_df['month'].min()} to {db_df['year'].max()}-{db_df['month'].max()}",
        "coordinates": {"latitude": 43.6830, "longitude": 10.3991},
        "columns": {
            "timestamp": "ISO 8601 datetime string (YYYY-MM-DDTHH:MM:SS)",
            "temperature": "Temperature in Celsius degrees (numeric)",
            "dew_point": "Dew point temperature in Celsius degrees (numeric)",
            "humidity": "Relative humidity percentage (numeric)",
            "wind_direction": "Wind direction abbreviation (e.g., N, SE, WSW)",
            "wind_degrees": "Wind direction in degrees (0-360, where 0/360 is North)",
            "wind_speed": "Wind speed in km/h (numeric)",
            "wind_gust": "Wind gust speed in km/h (numeric)",
            "pressure": "Atmospheric pressure in hPa (numeric)",
            "precipitation": "Precipitation in mm (numeric)",
            "condition": "Weather condition description (text)",
            "weather_category": "Simplified weather category (sunny, cloudy, rainy, snowy, foggy, other)"
        },
        "schema_version": "1.0",
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }
    
    # Reorganize columns in a logical order
    column_order = [
        'timestamp', 'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'is_weekend',
        'temperature', 'dew_point', 'humidity', 
        'wind_direction', 'wind_degrees', 'wind_speed', 'wind_gust', 
        'pressure', 'precipitation', 'condition', 'weather_category'
    ]
    
    # Keep only columns that exist
    final_columns = [col for col in column_order if col in db_df.columns]
    
    # Add any columns that might exist but aren't in our predefined order
    extra_columns = [col for col in db_df.columns if col not in column_order]
    final_columns.extend(extra_columns)
    
    db_df = db_df[final_columns]
    
    print(f"Created optimized version with {len(db_df)} rows and {len(db_df.columns)} columns")
    return db_df, metadata

def merge_weather_data():
    print("Starting to merge weather data files...")
    
    # Get the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the scraping output directory
    input_dir = os.path.join(current_dir, "scraping_weather_data", "output")
    
    # Check if the directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return False
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.startswith("weather_data_") and f.endswith(".csv")]
    
    if not csv_files:
        print("No weather data CSV files found to merge")
        return False
    
    print(f"Found {len(csv_files)} CSV files to merge: {', '.join(csv_files)}")
    
    # Read all CSV files into a list of DataFrames with explicit encoding
    dfs = []
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {e}")
                    raise
            
            if not df.empty:
                print(f"Read {len(df)} rows from {file}")
                
                # Clean data immediately when loading
                for column in df.columns:
                    df[column] = df[column].apply(clean_special_chars)
                
                dfs.append(df)
            else:
                print(f"Warning: {file} is empty, skipping")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid data found in the CSV files")
        return False
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after concatenation: {len(merged_df)}")
    
    # Apply more aggressive cleaning specifically for the Â character
    print("Performing direct byte-level replacement of special characters...")
    for column in merged_df.columns:
        if merged_df[column].dtype == object:  # Only process string columns
            # Two-step approach to remove the character
            merged_df[column] = merged_df[column].astype(str).str.replace('Â', '')
    
    # Sort by Date and Time
    if 'Date' in merged_df.columns and 'Time' in merged_df.columns:
        try:
            # Create a temporary DateTime column for sorting
            print("Creating DateTime column for sorting...")
            
            # Try to parse both 12-hour and 24-hour time formats
            try:
                merged_df['DateTime'] = pd.to_datetime(
                    merged_df['Date'] + ' ' + merged_df['Time'], 
                    errors='coerce',
                    format='%Y-%m-%d %I:%M %p'  # 12-hour format with AM/PM
                )
            except:
                pass
                
            # Handle NaT values by trying 24-hour format
            mask_nat = merged_df['DateTime'].isna() if 'DateTime' in merged_df.columns else pd.Series(True, index=merged_df.index)
            if mask_nat.any():
                try:
                    merged_df.loc[mask_nat, 'DateTime'] = pd.to_datetime(
                        merged_df.loc[mask_nat, 'Date'] + ' ' + merged_df.loc[mask_nat, 'Time'],
                        errors='coerce',
                        format='%Y-%m-%d %H:%M'  # 24-hour format
                    )
                except:
                    pass
            
            # Sort by DateTime
            if 'DateTime' in merged_df.columns:
                merged_df = merged_df.sort_values('DateTime')
                
                # Keep the DateTime column with standardized format
                if 'DateTime' in merged_df.columns:
                    merged_df['DateTime'] = merged_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                print("Data sorted by date and time")
            else:
                # Fall back to simpler sorting
                merged_df = merged_df.sort_values(['Date', 'Time'])
                print("Data sorted by Date and Time columns (simple sort)")
                
                # Apply standardization after sorting
                merged_df = standardize_datetime(merged_df)
        except Exception as e:
            print(f"Error during datetime sorting: {e}")
            # Fallback sort by Date only
            merged_df = merged_df.sort_values('Date')
            print("Data sorted by Date column only due to errors")
    else:
        print("Warning: Date or Time column not found, data may not be properly sorted")
    
    # Check for and clean any empty rows or duplicate data
    print("Cleaning data...")
    
    # Remove rows where both Date and Time are missing
    if 'Date' in merged_df.columns and 'Time' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['Date', 'Time'], how='all')
    
    # Remove duplicate rows
    initial_rows = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    if len(merged_df) < initial_rows:
        print(f"Removed {initial_rows - len(merged_df)} duplicate rows")
    
    # Final verification and aggressive cleanup for temperature values
    if 'Temperature' in merged_df.columns:
        print("Final temperature value cleaning...")
        
        # First check
        sample_temps = merged_df['Temperature'].dropna().sample(min(5, len(merged_df))).tolist()
        print(f"Sample temperature values before final cleaning: {sample_temps}")
        
        # Direct string replacement for all temperature-related columns
        temp_columns = ['Temperature', 'Dew Point']
        for col in temp_columns:
            if col in merged_df.columns:
                # Convert all values to strings first
                merged_df[col] = merged_df[col].astype(str)
                # Apply the most aggressive cleaning
                merged_df[col] = merged_df[col].str.replace('Â', '', regex=False)
                merged_df[col] = merged_df[col].str.replace('  ', ' ', regex=False)
        
        # Verify again
        sample_temps = merged_df['Temperature'].dropna().sample(min(5, len(merged_df))).tolist()
        print(f"Sample temperature values after final cleaning: {sample_temps}")
    
    # Save the merged data to CSV and Excel formats
    output_csv = os.path.join(current_dir, "weather_data.csv")
    output_excel = os.path.join(current_dir, "weather_data.xlsx")
    
    # Save as CSV
    merged_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Merged data saved to CSV: {output_csv} ({len(merged_df)} rows)")
    
    # Save as Excel
    try:
        merged_df.to_excel(output_excel, index=False, engine='openpyxl')
        print(f"Merged data also saved to Excel: {output_excel}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
    
    # Create and save the Supabase version
    supabase_df = prepare_supabase_version(merged_df)
    supabase_csv = os.path.join(current_dir, "halfhourly_weather_data_supabase.csv")
    supabase_df.to_csv(supabase_csv, index=False, encoding='utf-8')
    print(f"Supabase-friendly version saved to: {supabase_csv}")
    
    # Create and save the database-optimized version for LLM usage
    db_df, metadata = prepare_db_optimized_version(merged_df)
    db_csv = os.path.join(current_dir, "weather_data_db_optimized.csv")
    db_df.to_csv(db_csv, index=False, encoding='utf-8')
    
    # Save the metadata
    metadata_file = os.path.join(current_dir, "weather_data_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Database-optimized version saved to: {db_csv}")
    print(f"Metadata saved to: {metadata_file}")
    
    # Final manual cleanup to ensure all Â characters are removed
    try:
        print("Performing final manual file cleanup...")
        
        # List of files to clean
        files_to_clean = [output_csv, supabase_csv, db_csv]
        
        for file_path in files_to_clean:
            # Manual cleanup for CSV file
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Replace the character at the byte level
            cleaned_content = content.replace('Â', '')
            
            with codecs.open(file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_content)
            
            print(f"File {os.path.basename(file_path)} manually cleaned")
        
        # Show sample lines from both files
        print("\nSample lines from standard CSV file:")
        with open(output_csv, 'r', encoding='utf-8') as f:
            sample_lines = [next(f) for _ in range(min(3, len(merged_df)+1))]
        for line in sample_lines:
            print(f"  {line.strip()}")
            
        print("\nSample lines from Supabase CSV file:")
        with open(supabase_csv, 'r', encoding='utf-8') as f:
            sample_lines = [next(f) for _ in range(min(3, len(supabase_df)+1))]
        for line in sample_lines:
            print(f"  {line.strip()}")
        
        # Show sample lines from the DB optimized file
        print("\nSample lines from database-optimized CSV file:")
        with open(db_csv, 'r', encoding='utf-8') as f:
            sample_lines = [next(f) for _ in range(min(3, len(db_df)+1))]
        for line in sample_lines:
            print(f"  {line.strip()}")
        
    except Exception as e:
        print(f"Error during manual cleanup: {e}")
    
    return True

if __name__ == "__main__":
    if merge_weather_data():
        print("✅ Weather data merging completed successfully!")
    else:
        print("❌ Weather data merging failed")
