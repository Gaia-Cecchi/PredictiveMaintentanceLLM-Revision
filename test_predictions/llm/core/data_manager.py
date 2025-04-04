import os
import pandas as pd
import sqlite3
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import timedelta

class CompressorDataManager:
    """Manages access to compressor data from SQLite database"""
    
    def __init__(self, db_path: str, labeled_db_path: Optional[str] = None):
        """
        Initialize the data manager with paths to databases
        
        Args:
            db_path: Path to the main database containing sensor data
            labeled_db_path: Path to database with labeled events (anomalies, false positives)
                             If None, uses the same as db_path
        """
        self.db_path = db_path
        self.labeled_db_path = labeled_db_path or db_path
        self.logger = logging.getLogger(__name__)
        
        # Validate database paths
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        if self.labeled_db_path != self.db_path and not os.path.exists(self.labeled_db_path):
            raise FileNotFoundError(f"Labeled database file not found: {self.labeled_db_path}")
        
        # Add cache for data windows to avoid redundant database queries
        self._data_window_cache = {}
        
    def get_data_window(self, event_time, window_hours: int = 12) -> Dict[str, Any]:
        """
        Extracts data in a time window around a specific event - optimized for speed
        
        Args:
            event_time: DateTime of the event to analyze
            window_hours: Number of hours to include in the window (centered on event_time)
                          Default reduced from 24 to 12 for faster processing
            
        Returns:
            Dictionary containing sensor data, weather data, and formatted string versions
        """
        # Check cache first
        cache_key = f"{event_time}_{window_hours}"
        if cache_key in self._data_window_cache:
            self.logger.info(f"Using cached data window for {event_time}")
            return self._data_window_cache[cache_key]
        
        # Calculate time window boundaries - reduced from 24 to 12 hours by default
        half_window = timedelta(hours=window_hours//2)
        start_time = pd.Timestamp(event_time) - half_window
        end_time = pd.Timestamp(event_time) + half_window
        
        self.logger.info(f"Extracting data window from {start_time} to {end_time}")
        
        # Connect to database
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Optimization: Sample the data points for efficiency
            # If window is large, select fewer points to reduce data size
            query = f"""
            SELECT * FROM compressor_data_with_weather 
            WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
            AND (DateTime = '{event_time}' OR rowid % 3 = 0)  -- Take exact time plus every 3rd row
            ORDER BY DateTime
            """
            
            # Get data frame
            df = pd.read_sql(query, conn)
            
            # Convert DateTime column to proper datetime
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
            
            # Separate compressor data from weather data
            compressor_cols = [col for col in df.columns if not col.startswith('weather_')]
            weather_cols = ['DateTime'] + [col for col in df.columns if col.startswith('weather_')]
            
            compressor_data = df[compressor_cols]
            weather_data = df[weather_cols]
            
            # Extract exact readings at event time
            exact_readings = {}
            exact_time_data = df[df['DateTime'] == event_time]
            
            if not exact_time_data.empty:
                for col in ['Temperature', 'Vibration', 'Pressure', 'Current', 'Speed', 'Voltage', 'CosPhi']:
                    if col in df.columns:
                        exact_readings[col.lower()] = exact_time_data[col].iloc[0]
            
            # Optimize string representation to be more concise
            compact_compressor = compressor_data.copy()
            
            # Round values to reduce string length
            for col in compact_compressor.select_dtypes(include=['float']).columns:
                compact_compressor[col] = compact_compressor[col].round(1)
            
            # Create concise string representation
            compressor_rows = []
            for _, row in compact_compressor.iterrows():
                # Format: DateTime,Temp,Vib,Pressure,Current,Speed,Voltage,CosPhi
                values = [
                    row['DateTime'].strftime('%Y-%m-%d %H:%M'),
                    f"{row.get('Temperature', 'N/A')}",
                    f"{row.get('Vibration', 'N/A')}",
                    f"{row.get('Pressure', 'N/A')}",
                    f"{row.get('Current', 'N/A')}",
                    f"{row.get('Speed', 'N/A')}" if 'Speed' in row else "N/A",
                    f"{row.get('Voltage', 'N/A')}" if 'Voltage' in row else "N/A",
                    f"{row.get('CosPhi', 'N/A')}" if 'CosPhi' in row else "N/A"
                ]
                compressor_rows.append(",".join(values))
            
            compressor_str = "DateTime,Temp,Vib,Press,Curr,Speed,Voltage,CosPhi\n" + "\n".join(compressor_rows)
            
            # Optimize weather string (only include essential data)
            weather_rows = []
            for _, row in weather_data.iterrows():
                if 'weather_temperature' in row:
                    values = [
                        row['DateTime'].strftime('%Y-%m-%d %H:%M'),
                        f"{row.get('weather_temperature', 'N/A')}",
                        f"{row.get('weather_humidity', 'N/A')}",
                        f"{row.get('weather_windspeed', 'N/A')}"
                    ]
                    weather_rows.append(",".join(values))
            
            weather_str = "DateTime,Temp,Humidity,WindSpeed\n" + "\n".join(weather_rows)
            
            # Gather results
            result = {
                'compressor_data': compressor_data,
                'weather_data': weather_data,
                'exact_readings': exact_readings,
                'compressor_str': compressor_str,
                'weather_str': weather_str,
                'window_start': start_time,
                'window_end': end_time
            }
            
            # Store in cache
            self._data_window_cache[cache_key] = result
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving data window: {str(e)}")
            if 'conn' in locals() and conn:
                conn.close()
            raise
        
    def get_labeled_events(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves all labeled events (anomalies and false positives) from database
        
        Returns:
            Tuple containing (anomalies DataFrame, false positives DataFrame)
        """
        try:
            conn = sqlite3.connect(self.labeled_db_path)
            
            # Query anomalies and false positives
            anomalies = pd.read_sql("SELECT * FROM anomalies", conn)
            false_positives = pd.read_sql("SELECT * FROM false_positives", conn)
            
            # Convert DateTime columns to proper datetime objects
            anomalies['DateTime'] = pd.to_datetime(anomalies['DateTime'])
            false_positives['DateTime'] = pd.to_datetime(false_positives['DateTime'])
            
            # Add event type and classification for consistency
            anomalies['event_type'] = 'anomaly'
            anomalies['actual_classification'] = 'ANOMALY'
            
            false_positives['event_type'] = 'false_positive'
            false_positives['actual_classification'] = 'NORMAL VALUE'
            
            conn.close()
            
            self.logger.info(f"Retrieved {len(anomalies)} anomalies and {len(false_positives)} false positives")
            return anomalies, false_positives
            
        except Exception as e:
            self.logger.error(f"Error retrieving labeled events: {str(e)}")
            if 'conn' in locals() and conn:
                conn.close()
            raise
    
    def select_normal_cases(self, exclude_dates: List, n_samples: int = 20) -> pd.DataFrame:
        """
        Selects normal cases that are not in the list of labeled events
        Strategically selects cases that are challenging (near thresholds)
        
        Args:
            exclude_dates: List of dates to exclude (labeled events)
            n_samples: Number of normal cases to select
            
        Returns:
            DataFrame containing selected normal cases with classification labels
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all compressor data
            all_data = pd.read_sql("SELECT * FROM compressor_data_with_weather", conn)
            all_data['DateTime'] = pd.to_datetime(all_data['DateTime'])
            
            # Exclude the dates of labeled events
            exclude_dates = pd.to_datetime(exclude_dates)
            normal_data = all_data[~all_data['DateTime'].isin(exclude_dates)]
            
            self.logger.info(f"Found {len(normal_data)} total normal records after excluding labeled events")
            
            # Select a variety of normal cases with different characteristics
            selected_samples = []
            remaining_samples = n_samples
            
            # 1. Cases with high temperature but below critical threshold (95-115°C)
            high_temp_mask = (normal_data['Temperature'] >= 95) & (normal_data['Temperature'] <= 114)
            if sum(high_temp_mask) > 0:
                samples_to_take = min(round(n_samples * 0.25), sum(high_temp_mask))
                high_temp_cases = normal_data[high_temp_mask].sample(samples_to_take)
                selected_samples.append(high_temp_cases)
                remaining_samples -= samples_to_take
                
            # 2. Cases with moderate vibration (2.0-3.9 mm/s)
            if remaining_samples > 0:
                vibration_mask = (normal_data['Vibration'] >= 2.0) & (normal_data['Vibration'] <= 3.9)
                if sum(vibration_mask) > 0:
                    samples_to_take = min(round(n_samples * 0.25), sum(vibration_mask), remaining_samples)
                    vibration_cases = normal_data[vibration_mask].sample(samples_to_take)
                    selected_samples.append(vibration_cases)
                    remaining_samples -= samples_to_take
            
            # 3. Cases with pressure near threshold (5.5-6.0 bar)
            if remaining_samples > 0:
                pressure_mask = (normal_data['Pressure'] >= 5.5) & (normal_data['Pressure'] <= 6.0)
                if sum(pressure_mask) > 0:
                    samples_to_take = min(round(n_samples * 0.2), sum(pressure_mask), remaining_samples)
                    pressure_cases = normal_data[pressure_mask].sample(samples_to_take)
                    selected_samples.append(pressure_cases)
                    remaining_samples -= samples_to_take
            
            # 4. Add completely normal cases to fill the remaining quota
            if remaining_samples > 0:
                # Exclude already selected rows
                selected_indices = pd.concat(selected_samples).index if selected_samples else pd.Index([])
                remaining_data = normal_data[~normal_data.index.isin(selected_indices)]
                
                if len(remaining_data) > 0:
                    samples_to_take = min(remaining_samples, len(remaining_data))
                    normal_cases = remaining_data.sample(samples_to_take)
                    selected_samples.append(normal_cases)
            
            # Combine all selected samples
            selected = pd.concat(selected_samples) if selected_samples else pd.DataFrame()
            
            # Add classification information
            selected['event_type'] = 'normal'
            selected['actual_classification'] = 'NORMAL VALUE'
            selected['Notes'] = 'Normal operation'
            
            conn.close()
            
            self.logger.info(f"Selected {len(selected)} normal cases with varied characteristics")
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting normal cases: {str(e)}")
            if 'conn' in locals() and conn:
                conn.close()
            raise
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the dataset
        
        Returns:
            Dictionary containing dataset statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Count total records
            total_count = pd.read_sql("SELECT COUNT(*) as count FROM compressor_data_with_weather", conn).iloc[0]['count']
            
            # Get date range
            date_range = pd.read_sql(
                "SELECT MIN(DateTime) as min_date, MAX(DateTime) as max_date FROM compressor_data_with_weather", 
                conn
            )
            
            # Get sensor value ranges
            sensor_ranges = pd.read_sql("""
                SELECT 
                    MIN(Temperature) as min_temp, MAX(Temperature) as max_temp,
                    MIN(Vibration) as min_vibration, MAX(Vibration) as max_vibration,
                    MIN(Pressure) as min_pressure, MAX(Pressure) as max_pressure,
                    MIN(Current) as min_current, MAX(Current) as max_current
                FROM compressor_data_with_weather
            """, conn)
            
            conn.close()
            
            return {
                'total_records': total_count,
                'start_date': date_range.iloc[0]['min_date'],
                'end_date': date_range.iloc[0]['max_date'],
                'sensor_ranges': sensor_ranges.to_dict('records')[0]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data statistics: {str(e)}")
            if 'conn' in locals() and conn:
                conn.close()
            raise