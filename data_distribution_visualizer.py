import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats

# Define units of measurement for common sensor parameters
UNITS = {
    'Current': 'A',
    'Voltage': 'V',
    'Temperature': '°C',
    'Pressure': 'bar',
    'Vibration': 'mm/s',
    'Speed': 'RPM',
    'Energy_Consumption': 'kWh',
    'Reactive_Energy': 'kVArh',
    'CosPhi': '',  # Dimensionless
    'weather_temperature': '°C',
    'weather_humidity': '%',
    'weather_wind_speed': 'km/h',
    'weather_precipitation': 'mm'
}

def get_label_with_unit(column_name):
    """
    Returns a formatted label with units for a column
    
    Args:
        column_name: Name of the column
        
    Returns:
        String with the column name and unit
    """
    base_name = column_name.split('(')[0].strip()  # Handle cases like "Current (A)"
    
    # Check for exact matches
    if column_name in UNITS:
        unit = UNITS[column_name]
        return f"{column_name} ({unit})" if unit else column_name
    
    # Check for partial matches (e.g., if column is "Motor Temperature" and we have "Temperature" in UNITS)
    for key, unit in UNITS.items():
        if key in column_name:
            return f"{column_name} ({unit})" if unit else column_name
    
    return column_name  # Return unchanged if no unit found

def create_distribution_plots(excel_path, output_dir):
    """
    Generates distribution plots for the main columns of the dataset
    
    Args:
        excel_path: Path to the Excel dataset file
        output_dir: Directory where to save the generated figures
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from: {excel_path}")
    # Load the dataset
    try:
        df = pd.read_excel(excel_path)
        print(f"Dataset loaded successfully. Size: {df.shape}")
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        return
    
    # Use the entire dataset instead of just the first 10 rows
    df_sample = df
    print(f"Analyzing the entire dataset with {len(df_sample)} rows")
    
    # Check if we have enough data for analysis
    if len(df_sample) < 2:
        print("Not enough data for analysis. Need at least 2 rows.")
        return
    
    # Check if anomaly-related columns exist
    has_anomaly_columns = all(col in df.columns for col in ['Anomaly', 'Anomaly_Detected', 'Notes'])
    
    # If we have anomaly labels, create anomaly-specific visualizations
    if has_anomaly_columns:
        print("Anomaly columns detected. Creating anomaly-focused visualizations...")
        create_anomaly_visualizations(df, output_dir)
    else:
        # Try to load labeled data from a separate file if available
        labeled_path = excel_path.replace('.xlsx', '_etichettato.xlsx')
        if not os.path.exists(labeled_path):
            labeled_path = excel_path.replace('with_weather_2024', 'with_weather_2024_etichettato')
        
        if os.path.exists(labeled_path):
            print(f"Loading labeled dataset from: {labeled_path}")
            try:
                labeled_df = pd.read_excel(labeled_path)
                if all(col in labeled_df.columns for col in ['Anomaly', 'Anomaly_Detected', 'Notes']):
                    print("Creating anomaly visualizations from labeled dataset...")
                    create_anomaly_visualizations(labeled_df, output_dir)
                else:
                    print("Labeled dataset doesn't contain required anomaly columns.")
            except Exception as e:
                print(f"Error loading labeled dataset: {str(e)}")
        else:
            print("No anomaly columns found in dataset and no labeled version available.")
    
    # 1. Overview of numerical columns
    numerical_cols = df_sample.select_dtypes(include=['number']).columns
    print(f"Numerical columns identified: {len(numerical_cols)}")
    
    # 2. Histograms for main numerical columns (excluding indices or timestamps)
    excluded_columns = ['index', 'idx', 'id', 'timestamp', 'date', 'time', 'Anomaly', 'Anomaly_Detected']
    sensor_cols = [col for col in numerical_cols if not any(excl in col.lower() for excl in excluded_columns)]
    
    # Limit to maximum 8 columns of sensors for visualizations
    main_sensor_cols = sensor_cols[:8] if len(sensor_cols) > 8 else sensor_cols
    
    print(f"Sensor columns for visualization: {len(main_sensor_cols)}")
    
    # Create distribution plots for the entire dataset
    print("Generating distribution plots for the entire dataset...")
    
    # Generate histogram for each main sensor column
    for col in main_sensor_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_sample[col], kde=True, bins=30)
        plt.title(f'Distribution of {get_label_with_unit(col)}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        filename = f"{output_dir}/distribution_{col.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")
    
    # Create statistical summary visualizations
    create_statistical_summary(df_sample, main_sensor_cols, output_dir)
    
    # Create combined distribution visualizations
    create_combined_distribution(df_sample, main_sensor_cols, output_dir)
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df_sample[main_sensor_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt=".2f", linewidths=0.5)
    plt.title('Sensor Correlation Matrix')
    plt.tight_layout()
    filename = f"{output_dir}/sensor_correlation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix: {filename}")
    
    # Create pair plot for the main 4 sensor columns
    if len(main_sensor_cols) >= 4:
        print("Generating pair plot (this may take some time for large datasets)...")
        top_cols = main_sensor_cols[:4]
        sns_plot = sns.pairplot(df_sample[top_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
        sns_plot.fig.suptitle('Relationships between Main Sensors', y=1.02, fontsize=16)
        filename = f"{output_dir}/main_sensor_relationships.png"
        sns_plot.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pair plot: {filename}")
    
    # Temporal analysis if timestamp column exists
    time_cols = [col for col in df.columns if any(time_str in col.lower() for time_str in ['time', 'date', 'timestamp'])]
    if time_cols:
        time_col = time_cols[0]
        try:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Create time series plots for each main sensor
            create_time_series_plots(df, time_col, main_sensor_cols, output_dir)
            
            # Create the new combined time series with anomalies
            create_combined_anomaly_time_series(df, time_col, main_sensor_cols, output_dir)
            
            # Create hourly distribution
            df['hour'] = df[time_col].dt.hour
            plt.figure(figsize=(12, 6))
            sns.countplot(x='hour', data=df)
            plt.title('Hourly Distribution of Measurements')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Measurements')
            plt.grid(True, alpha=0.3, axis='y')
            filename = f"{output_dir}/hourly_distribution.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved hourly distribution: {filename}")
        except Exception as e:
            print(f"Unable to create temporal analysis: {str(e)}")
    
    # Add a call to the new comprehensive visualization function
    create_comprehensive_visualization(df_sample, main_sensor_cols, output_dir)
    
    # Aggiunta del nuovo grafico compatto per il paper
    create_compact_distribution_plot(df_sample, main_sensor_cols, output_dir)
    
    print(f"\nAnalysis completed successfully!")
    print(f"Plots have been saved to: {output_dir}")

def create_anomaly_visualizations(df, output_dir):
    """
    Creates visualizations specifically focused on anomalies
    
    Args:
        df: DataFrame with the data including anomaly columns
        output_dir: Directory to save the plots
    """
    # Create a subdirectory for anomaly visualizations
    anomaly_dir = os.path.join(output_dir, "anomaly_analysis")
    Path(anomaly_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Ensure datetime column exists and is in the right format
    time_cols = [col for col in df.columns if any(time_str in col.lower() for time_str in ['datetime', 'time', 'date', 'timestamp'])]
    if not time_cols:
        print("No datetime column found for temporal analysis of anomalies.")
        return
    
    time_col = time_cols[0]
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Make a copy of the DataFrame with the time column as index
    df_time = df.copy()
    df_time.set_index(time_col, inplace=True)
    
    # 2. Calculate basic statistics
    n_anomalies = df['Anomaly'].sum()
    n_detected = df['Anomaly_Detected'].sum()
    n_false_positives = (df['Anomaly_Detected'] & ~df['Anomaly']).sum()
    n_false_negatives = (df['Anomaly'] & ~df['Anomaly_Detected']).sum()
    total_records = len(df)
    
    # 3. Create a timeline visualization of anomalies
    plt.figure(figsize=(16, 6))
    
    # Plot all points as gray dots
    plt.scatter(df[time_col], np.zeros(len(df)), color='lightgray', s=5, alpha=0.5, label='Normal data')
    
    # Plot true anomalies as red dots
    anomalies = df[df['Anomaly']]
    plt.scatter(anomalies[time_col], np.zeros(len(anomalies)), color='red', s=80, label='True anomalies')
    
    # Plot false positives as orange triangles
    false_positives = df[(df['Anomaly_Detected']) & (~df['Anomaly'])]
    plt.scatter(false_positives[time_col], np.zeros(len(false_positives)), 
               color='orange', marker='^', s=80, label='False positives')
    
    # Format the plot
    plt.title('Anomaly Timeline Distribution', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('')
    plt.yticks([])
    plt.grid(True, axis='x', alpha=0.3)
    plt.legend(loc='upper right')
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    # Add annotations for anomaly types
    annotation_height = 0.1
    for idx, row in anomalies.iterrows():
        anomaly_type = row['Notes'].replace("Anomaly: ", "") if isinstance(row['Notes'], str) else "Unknown"
        plt.annotate(anomaly_type, 
                    xy=(row[time_col], 0), 
                    xytext=(0, 15), 
                    textcoords="offset points",
                    ha='center', 
                    va='bottom',
                    rotation=90,
                    fontsize=8)
    
    plt.tight_layout()
    filename = f"{anomaly_dir}/anomaly_timeline.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly timeline: {filename}")
    
    # 4. Create pie charts for anomaly statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Pie chart of anomaly distribution
    anomaly_counts = [n_anomalies, total_records - n_anomalies]
    ax1.pie(anomaly_counts, labels=['Anomalies', 'Normal'], autopct='%1.1f%%', 
           colors=['red', 'lightgray'], explode=(0.1, 0), shadow=True)
    ax1.set_title('Distribution of Anomalies in Dataset')
    
    # Pie chart of detection performance
    detection_counts = [n_anomalies - n_false_negatives, n_false_positives, n_false_negatives] 
    ax2.pie(detection_counts, 
           labels=['True Positives', 'False Positives', 'False Negatives'], 
           autopct='%1.1f%%', 
           colors=['green', 'orange', 'red'], 
           explode=(0.1, 0.05, 0.05), 
           shadow=True)
    ax2.set_title('Anomaly Detection Performance')
    
    plt.tight_layout()
    filename = f"{anomaly_dir}/anomaly_statistics_pie.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly statistics pie charts: {filename}")
    
    # 5. Extract anomaly types and create a bar chart of their distribution
    anomaly_types = []
    for note in anomalies['Notes']:
        if isinstance(note, str) and "Anomaly: " in note:
            anomaly_type = note.replace("Anomaly: ", "")
            anomaly_types.append(anomaly_type)
        else:
            anomaly_types.append("Unknown")
    
    type_counts = pd.Series(anomaly_types).value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values)
    plt.title('Distribution of Anomaly Types')
    plt.xlabel('Anomaly Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"{anomaly_dir}/anomaly_types_distribution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly types distribution: {filename}")
    
    # 6. Create visualizations showing sensor behavior during anomalies
    numerical_cols = df.select_dtypes(include=['number']).columns
    excluded_columns = ['index', 'idx', 'id', 'timestamp', 'date', 'time', 'Anomaly', 'Anomaly_Detected']
    sensor_cols = [col for col in numerical_cols if not any(excl in col.lower() for excl in excluded_columns)]
    
    # Limit to first 6 sensor columns for clarity
    main_sensor_cols = sensor_cols[:6]
    
    # Create boxplot comparison between normal and anomaly periods
    plt.figure(figsize=(14, 8))
    
    # Prepare data in long format for seaborn
    melted_df = pd.melt(df.reset_index(), 
                        id_vars=['Anomaly'], 
                        value_vars=main_sensor_cols,
                        var_name='Sensor', 
                        value_name='Value')
    
    # Create the boxplot
    sns.boxplot(x='Sensor', y='Value', hue='Anomaly', data=melted_df)
    plt.title('Comparison of Sensor Values: Normal vs. Anomaly')
    plt.xlabel('')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Is Anomaly', labels=['Normal', 'Anomaly'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{anomaly_dir}/sensor_values_anomaly_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sensor values comparison: {filename}")
    
    # 7. Create visualizations for each anomaly type
    for anomaly_type in type_counts.index:
        # Filter to show just this anomaly type
        type_mask = df['Notes'].str.contains(anomaly_type, na=False) if 'Notes' in df.columns else False
        type_df = df[type_mask].copy()
        
        if len(type_df) == 0:
            continue
        
        # Select a random anomaly instance of this type
        sample_anomaly = type_df.sample(1).iloc[0]
        anomaly_time = sample_anomaly[time_col]
        
        # Get data for 24 hours before and after the anomaly
        time_window = pd.Timedelta(hours=24)
        window_start = anomaly_time - time_window
        window_end = anomaly_time + time_window
        
        window_data = df[(df[time_col] >= window_start) & (df[time_col] <= window_end)]
        
        # Create a time series plot showing sensor behavior before/during/after the anomaly
        fig, axes = plt.subplots(len(main_sensor_cols), 1, figsize=(12, 3*len(main_sensor_cols)), sharex=True)
        
        for i, col in enumerate(main_sensor_cols):
            # Plot the sensor data
            axes[i].plot(window_data[time_col], window_data[col], 'b-')
            
            # Highlight the anomaly point
            anomaly_point = window_data[window_data[time_col] == anomaly_time]
            if len(anomaly_point) > 0:
                axes[i].scatter(anomaly_point[time_col], anomaly_point[col], color='red', s=50)
            
            # Add vertical line at anomaly time
            axes[i].axvline(anomaly_time, color='red', linestyle='--', alpha=0.5)
            
            # Set title and labels
            axes[i].set_title(get_label_with_unit(col))
            axes[i].grid(True, alpha=0.3)
            
            # Get unit for y-label
            unit = ''
            for key, val in UNITS.items():
                if key in col:
                    unit = val
                    break
            
            # Add y-label with unit if available
            if unit:
                axes[i].set_ylabel(f"{col} ({unit})")
            else:
                axes[i].set_ylabel(col)
        
        # Format the date axis
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axes[-1].set_xlabel('Time')
        
        # Add overall title
        plt.suptitle(f'Sensor Behavior Around {anomaly_type} Anomaly', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        safe_type = anomaly_type.replace(" ", "_").replace("/", "_")
        filename = f"{anomaly_dir}/anomaly_behavior_{safe_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved anomaly behavior for {anomaly_type}: {filename}")
    
    # 8. Create a heatmap showing correlations between anomalies and sensor values
    # First, create dummy variables for each anomaly type
    for anomaly_type in type_counts.index:
        df[f'Anomaly_{anomaly_type.replace(" ", "_")}'] = df['Notes'].str.contains(anomaly_type, na=False).astype(int)
    
    # Calculate correlation between sensors and anomaly types
    anomaly_cols = [f'Anomaly_{atype.replace(" ", "_")}' for atype in type_counts.index]
    corr_df = df[main_sensor_cols + anomaly_cols].corr()
    
    # Extract just the correlations between sensors and anomaly types
    sensor_anomaly_corr = corr_df.loc[main_sensor_cols, anomaly_cols]
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sensor_anomaly_corr, cmap="coolwarm", center=0, annot=True, 
               fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Between Sensors and Anomaly Types')
    plt.tight_layout()
    filename = f"{anomaly_dir}/sensor_anomaly_correlation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sensor-anomaly correlation heatmap: {filename}")
    
    # 9. Create a summary dashboard of anomalies
    fig = plt.figure(figsize=(18, 12))
    grid = plt.GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.3)
    
    # Anomaly timeline (top row, spans all columns)
    ax_timeline = fig.add_subplot(grid[0, :])
    ax_timeline.scatter(df[time_col], np.zeros(len(df)), color='lightgray', s=5, alpha=0.5)
    ax_timeline.scatter(anomalies[time_col], np.zeros(len(anomalies)), color='red', s=80)
    ax_timeline.scatter(false_positives[time_col], np.zeros(len(false_positives)), color='orange', marker='^', s=80)
    ax_timeline.set_title('Anomaly Timeline')
    ax_timeline.set_xlabel('Date')
    ax_timeline.set_yticks([])
    ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_timeline.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    # Anomaly statistics (middle row, left column)
    ax_stats = fig.add_subplot(grid[1, 0])
    stats_data = [
        ['Total Records', total_records],
        ['Anomalies', n_anomalies],
        ['Anomaly Rate', f"{(n_anomalies/total_records)*100:.2f}%"],
        ['False Positives', n_false_positives],
        ['False Negatives', n_false_negatives]
    ]
    ax_stats.axis('tight')
    ax_stats.axis('off')
    stats_table = ax_stats.table(cellText=stats_data, loc='center', cellLoc='center')
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(12)
    stats_table.scale(1, 1.5)
    ax_stats.set_title('Anomaly Statistics')
    
    # Anomaly type distribution (middle row, middle and right columns)
    ax_types = fig.add_subplot(grid[1, 1:])
    sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax_types)
    ax_types.set_title('Anomaly Types Distribution')
    ax_types.set_xlabel('')
    ax_types.set_xticklabels(ax_types.get_xticklabels(), rotation=45, ha='right')
    
    # Sensor behavior during anomalies (bottom row, spans all columns)
    ax_behavior = fig.add_subplot(grid[2, :])
    sns.boxplot(x='Sensor', y='Value', hue='Anomaly', data=melted_df, ax=ax_behavior)
    ax_behavior.set_title('Sensor Values: Normal vs. Anomaly')
    ax_behavior.set_xlabel('')
    ax_behavior.set_xticklabels(ax_behavior.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Anomaly Analysis Dashboard', fontsize=20)
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    
    filename = f"{anomaly_dir}/anomaly_dashboard.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly dashboard: {filename}")

    # 10. Create calendar heatmap of anomalies by day of week and hour
    if len(anomalies) > 0:
        # Extract day of week and hour
        anomalies['day_of_week'] = anomalies[time_col].dt.day_name()
        anomalies['hour'] = anomalies[time_col].dt.hour
        
        # Create a pivot table for the heatmap
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        anomalies['day_of_week'] = pd.Categorical(anomalies['day_of_week'], categories=days_order, ordered=True)
        
        # Count anomalies by day and hour
        anomaly_counts = anomalies.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(anomaly_counts, cmap="Reds", linewidths=0.5, annot=True, fmt='d')
        plt.title('Anomaly Occurrence by Day of Week and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        plt.tight_layout()
        
        filename = f"{anomaly_dir}/anomaly_calendar_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved anomaly calendar heatmap: {filename}")

def create_statistical_summary(df, columns, output_dir):
    """
    Creates a statistical summary visualization for the dataset
    
    Args:
        df: DataFrame with the data
        columns: List of column names to include
        output_dir: Directory to save the plot
    """
    # Create a boxplot for all columns
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df[columns])
    plt.title('Boxplot of Sensor Measurements')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filename = f"{output_dir}/boxplot_all_sensors.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot: {filename}")
    
    # Create a violinplot for better distribution visualization
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df[columns])
    plt.title('Violin Plot of Sensor Distributions')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    filename = f"{output_dir}/violinplot_all_sensors.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved violin plot: {filename}")
    
    # Create a statistical summary table
    stats_df = df[columns].describe().transpose().reset_index()
    stats_df = stats_df.rename(columns={'index': 'Sensor'})
    
    # Add units to sensor names
    stats_df['Sensor with Units'] = stats_df['Sensor'].apply(get_label_with_unit)
    
    # Create a visualized table of statistics
    plt.figure(figsize=(14, len(columns)*0.8))
    cell_text = []
    for _, row in stats_df.iterrows():
        cell_text.append([
            row['Sensor with Units'],
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.2f}",
            f"{row['25%']:.2f}",
            f"{row['50%']:.2f}",
            f"{row['75%']:.2f}",
            f"{row['max']:.2f}"
        ])
    
    table = plt.table(
        cellText=cell_text,
        colLabels=['Sensor', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    plt.title('Statistical Summary of Sensor Data', pad=20)
    filename = f"{output_dir}/statistical_summary.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistical summary: {filename}")

def create_combined_distribution(df, columns, output_dir):
    """
    Creates a single combined plot showing the distribution of all selected columns
    
    Args:
        df: DataFrame with the data
        columns: List of column names to include
        output_dir: Directory to save the plot
    """
    # Create a grid of distribution plots
    n_cols = len(columns)
    if n_cols <= 3:
        n_rows, n_cols = 1, n_cols
    elif n_cols <= 6:
        n_rows, n_cols = 2, 3
    else:
        n_rows, n_cols = 3, 3
    
    # Create a figure for all distributions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    # Flatten axes array for easy iteration
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each distribution
    for i, col in enumerate(columns):
        if i < len(axes):
            sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
            axes[i].set_title(get_label_with_unit(col))
            
            # Add stats to the plot
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Add a text box with mean and std
            axes[i].annotate(f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}", 
                        xy=(0.7, 0.85), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Add vertical line at the mean
            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7)
            
            axes[i].grid(True, alpha=0.3)
            
            # Get unit for x-label from our UNITS dictionary
            unit = ''
            for key, val in UNITS.items():
                if key in col:
                    unit = val
                    break
            
            # Set proper x-label with unit
            if unit:
                axes[i].set_xlabel(f"Value ({unit})")
            else:
                axes[i].set_xlabel("Value")
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Combined Distribution of Sensor Measurements\n(Each plot uses actual values with original units)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Add more space for the title
    
    # Add a note about the scale at the bottom of the figure
    plt.figtext(0.5, 0.01, 
                "Note: Each subplot shows the actual distribution with its original scale (not standardized).",
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the combined plot
    filename = f"{output_dir}/combined_sensor_distributions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined distribution plot: {filename}")
    
    # Create density plot of all distributions on one axis (standardized)
    plt.figure(figsize=(12, 8))
    for col in columns:
        # Standardize the data for comparison
        if df[col].std() > 0:  # Skip constant columns
            standardized_data = (df[col] - df[col].mean()) / df[col].std()
            sns.kdeplot(standardized_data, label=get_label_with_unit(col))
    
    plt.title('Overlaid Density Distributions of Standardized Sensor Data\n(Z-scores for fair comparison)')
    plt.xlabel('Standardized Value (Z-score = (x - mean) / std)')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add explanation of Z-score
    explanation = """Z-score standardization allows comparing distributions of different scales:
• Z-score = 0: Value equals the mean
• Z-score = ±1: Value is 1 standard deviation above/below mean
• Z-score = ±2: Value is 2 standard deviations above/below mean"""
    
    plt.annotate(explanation, xy=(0.02, 0.02), xycoords='figure fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 fontsize=9)
    
    plt.tight_layout()
    
    # Save the overlaid plot
    filename = f"{output_dir}/overlaid_sensor_distributions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overlaid distribution plot: {filename}")

def create_time_series_plots(df, time_col, columns, output_dir):
    """
    Creates time series plots for sensor measurements
    
    Args:
        df: DataFrame with the data
        time_col: Name of the timestamp column
        columns: List of column names to include
        output_dir: Directory to save the plots
    """
    # Check if time_col is actually a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        print(f"Warning: {time_col} is not a datetime column. Skipping time series plots.")
        return
    
    # Set time column as index for time series plotting
    df_time = df.copy().set_index(time_col)
    
    # Create a time series plot for each column
    for col in columns:
        plt.figure(figsize=(14, 6))
        plt.plot(df_time.index, df_time[col], linewidth=1)
        plt.title(f'Time Series of {get_label_with_unit(col)}')
        plt.xlabel('Time')
        
        # Get unit for y-label
        unit = ''
        for key, val in UNITS.items():
            if key in col:
                unit = val
                break
        
        # Add y-label with unit if available
        if unit:
            plt.ylabel(f"{col} ({unit})")
        else:
            plt.ylabel(col)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with better filename
        filename = f"{output_dir}/time_series_{col.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time series plot: {filename}")
    
    # Create a combined time series plot for all columns (standardized)
    plt.figure(figsize=(14, 8))
    
    for col in columns:
        # Standardize the data
        if df_time[col].std() > 0:  # Skip constant columns
            standardized = (df_time[col] - df_time[col].mean()) / df_time[col].std()
            plt.plot(df_time.index, standardized, linewidth=1, label=col)
    
    plt.title('Combined Standardized Time Series of All Sensors')
    plt.xlabel('Time')
    plt.ylabel('Standardized Value (Z-score)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{output_dir}/combined_time_series.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined time series plot: {filename}")

def create_comprehensive_visualization(df, columns, output_dir):
    """
    Creates a single comprehensive visualization that shows all important aspects of the dataset
    
    Args:
        df: DataFrame with the data
        columns: List of column names to include
        output_dir: Directory to save the plot
    """
    # Check if we have timestamp data for time series
    time_cols = [col for col in df.columns if any(time_str in col.lower() for time_str in ['datetime', 'time', 'date', 'timestamp'])]
    has_time_data = len(time_cols) > 0
    if has_time_data:
        time_col = time_cols[0]
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Check if we have anomaly data
    has_anomalies = 'Anomaly' in df.columns
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create a figure with gridspec
    fig = plt.figure(figsize=(20, 16))
    
    # Set up the grid - different layouts based on what data we have
    if has_time_data and has_anomalies:
        # More complex grid with time series and anomalies
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.5], width_ratios=[1.5, 1, 1, 1])
    elif has_time_data:
        # Grid with time series but no anomalies
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5])
    else:
        # Simpler grid without time series
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5])
    
    # 1. Main title
    fig.suptitle('Comprehensive Dataset Overview\nSensor Distributions, Correlations, and Anomalies', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 2. Statistical Summary Table - Top Left
    ax_stats = fig.add_subplot(gs[0, 0])
    stats_df = df[columns].describe().round(2).T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    # Add normality test results
    stats_df['normal'] = [stats.shapiro(df[col])[1] > 0.05 for col in stats_df.index]
    
    # Create the table
    cell_text = []
    for idx, row in stats_df.iterrows():
        is_normal = "✓" if row['normal'] else "✗"
        cell_text.append([
            get_label_with_unit(idx), 
            f"{row['mean']:.2f}",
            f"{row['std']:.2f}",
            f"{row['min']:.2f}",
            f"{row['max']:.2f}",
            is_normal
        ])
    
    stats_table = ax_stats.table(
        cellText=cell_text,
        colLabels=['Sensor', 'Mean', 'Std', 'Min', 'Max', 'Normal'],
        loc='center',
        cellLoc='center'
    )
    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(9)
    stats_table.scale(1, 1.4)
    ax_stats.axis('off')
    ax_stats.set_title('Statistical Summary', fontsize=12)
    
    # 3. Correlation Matrix Heatmap - Middle Left
    ax_corr = fig.add_subplot(gs[1, 0])
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt=".2f", linewidths=0.5, cbar=False, ax=ax_corr)
    ax_corr.set_title('Sensor Correlation Matrix', fontsize=12)
    
    # 4. Standardized Distributions - Top Middle and Top Right
    # Split columns between two plots if we have many columns
    mid_idx = len(columns) // 2
    first_cols = columns[:mid_idx] if mid_idx > 0 else columns
    second_cols = columns[mid_idx:] if mid_idx < len(columns) else []
    
    # First plot of standardized distributions
    ax_dist1 = fig.add_subplot(gs[0, 1])
    for col in first_cols:
        if df[col].std() > 0:  # Skip constant columns
            standardized_data = (df[col] - df[col].mean()) / df[col].std()
            sns.kdeplot(standardized_data, label=col, ax=ax_dist1)
    ax_dist1.set_title('Standardized Distributions (Part 1)', fontsize=12)
    ax_dist1.set_xlabel('Z-score value')
    ax_dist1.set_ylabel('Density')
    ax_dist1.legend(fontsize=8)
    ax_dist1.grid(True, alpha=0.3)
    
    # Second plot of standardized distributions if needed
    if second_cols:
        ax_dist2 = fig.add_subplot(gs[0, 2])
        for col in second_cols:
            if df[col].std() > 0:
                standardized_data = (df[col] - df[col].mean()) / df[col].std()
                sns.kdeplot(standardized_data, label=col, ax=ax_dist2)
        ax_dist2.set_title('Standardized Distributions (Part 2)', fontsize=12)
        ax_dist2.set_xlabel('Z-score value')
        ax_dist2.set_ylabel('Density')
        ax_dist2.legend(fontsize=8)
        ax_dist2.grid(True, alpha=0.3)
    
    # 5. Box plots with outliers - Middle Center/Right or Middle Right
    if len(columns) > 3:
        # Split boxplots if we have many columns
        ax_box1 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=df[first_cols], ax=ax_box1)
        ax_box1.set_title('Boxplot with Outliers (Part 1)', fontsize=12)
        ax_box1.set_xticklabels([get_label_with_unit(col) for col in first_cols], rotation=45, ha='right')
        ax_box1.grid(True, alpha=0.3, axis='y')
        
        if second_cols:
            ax_box2 = fig.add_subplot(gs[1, 2])
            sns.boxplot(data=df[second_cols], ax=ax_box2)
            ax_box2.set_title('Boxplot with Outliers (Part 2)', fontsize=12)
            ax_box2.set_xticklabels([get_label_with_unit(col) for col in second_cols], rotation=45, ha='right')
            ax_box2.grid(True, alpha=0.3, axis='y')
    else:
        # One boxplot panel is enough
        ax_box = fig.add_subplot(gs[1, 1:3])
        sns.boxplot(data=df[columns], ax=ax_box)
        ax_box.set_title('Boxplot with Outliers', fontsize=12)
        ax_box.set_xticklabels([get_label_with_unit(col) for col in columns], rotation=45, ha='right')
        ax_box.grid(True, alpha=0.3, axis='y')
    
    # 6. Time series and anomalies if available (Bottom Row)
    if has_time_data:
        # Set time as index for time series plots
        df_time = df.copy()
        df_time.set_index(time_col, inplace=True)
        
        # First time series - Temperature/Pressure/key parameter
        key_param = next((col for col in columns if 'Temperature' in col), columns[0])
        ax_time1 = fig.add_subplot(gs[2, 0])
        ax_time1.plot(df_time.index, df_time[key_param], 'b-', linewidth=1)
        
        # If we have anomalies, highlight them in the time series
        if has_anomalies:
            anomaly_times = df[df['Anomaly']][time_col]
            anomaly_values = df[df['Anomaly']][key_param]
            ax_time1.scatter(anomaly_times, anomaly_values, color='red', s=30, zorder=5, label='Anomalies')
            
            # Annotate anomalies if there are not too many
            if len(anomaly_times) <= 15:
                for i, (t, v) in enumerate(zip(anomaly_times, anomaly_values)):
                    ax_time1.annotate(f"A{i+1}", (t, v), xytext=(5, 5), 
                                    textcoords='offset points', fontsize=8)
        
        ax_time1.set_title(f'Time Series: {get_label_with_unit(key_param)}', fontsize=12)
        ax_time1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_time1.xaxis.set_major_locator(mdates.MonthLocator())
        ax_time1.grid(True, alpha=0.3)
        plt.setp(ax_time1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        if has_anomalies:
            ax_time1.legend(fontsize=8)
        
        # Second time series - Current/Voltage/another key parameter
        key_param2 = next((col for col in columns if 'Current' in col or 'Voltage' in col), 
                        next((col for col in columns if col != key_param), None))
        
        if key_param2:
            ax_time2 = fig.add_subplot(gs[2, 1])
            ax_time2.plot(df_time.index, df_time[key_param2], 'g-', linewidth=1)
            
            # If we have anomalies, highlight them in the time series
            if has_anomalies:
                anomaly_times = df[df['Anomaly']][time_col]
                anomaly_values = df[df['Anomaly']][key_param2]
                ax_time2.scatter(anomaly_times, anomaly_values, color='red', s=30, zorder=5)
            
            ax_time2.set_title(f'Time Series: {get_label_with_unit(key_param2)}', fontsize=12)
            ax_time2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_time2.xaxis.set_major_locator(mdates.MonthLocator())
            ax_time2.grid(True, alpha=0.3)
            plt.setp(ax_time2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Third time series - Another parameter
        remaining_cols = [col for col in columns if col not in [key_param, key_param2]]
        if remaining_cols:
            key_param3 = remaining_cols[0]
            ax_time3 = fig.add_subplot(gs[2, 2])
            ax_time3.plot(df_time.index, df_time[key_param3], 'r-', linewidth=1)
            
            # If we have anomalies, highlight them in the time series
            if has_anomalies:
                anomaly_times = df[df['Anomaly']][time_col]
                anomaly_values = df[df['Anomaly']][key_param3]
                ax_time3.scatter(anomaly_times, anomaly_values, color='red', s=30, zorder=5)
            
            ax_time3.set_title(f'Time Series: {get_label_with_unit(key_param3)}', fontsize=12)
            ax_time3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_time3.xaxis.set_major_locator(mdates.MonthLocator())
            ax_time3.grid(True, alpha=0.3)
            plt.setp(ax_time3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 7. Add anomaly stats panel if anomalies exist
    if has_anomalies and has_time_data:
        ax_anomaly = fig.add_subplot(gs[2, 3])
        
        # Calculate anomaly statistics
        n_anomalies = df['Anomaly'].sum()
        n_normal = len(df) - n_anomalies
        anomaly_rate = n_anomalies / len(df) * 100
        
        # Extract anomaly types
        if 'Notes' in df.columns:
            anomaly_types = []
            for note in df[df['Anomaly']]['Notes']:
                if isinstance(note, str) and "Anomaly: " in note:
                    anomaly_type = note.replace("Anomaly: ", "")
                    anomaly_types.append(anomaly_type)
            
            # Count occurrences of each type
            type_counts = pd.Series(anomaly_types).value_counts()
            
            # Create a pie chart of anomaly types
            ax_anomaly.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
                        startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})
            ax_anomaly.set_title(f'Anomaly Types Distribution\n(Total: {n_anomalies}, Rate: {anomaly_rate:.2f}%)', 
                                fontsize=12)
        else:
            # Just show normal vs anomaly distribution
            ax_anomaly.pie([n_normal, n_anomalies], labels=['Normal', 'Anomaly'], 
                        autopct='%1.1f%%', colors=['lightblue', 'red'],
                        explode=(0, 0.1), startangle=90, 
                        wedgeprops={'edgecolor': 'w', 'linewidth': 1})
            ax_anomaly.set_title(f'Dataset Composition\n(Anomaly Rate: {anomaly_rate:.2f}%)', 
                                fontsize=12)
    
    # 8. Add explanatory note to z-score plot
    if has_time_data:
        fig.text(0.91, 0.85, """Z-score values:
• Z=0: Mean
• Z=±1: 1 std
• Z=±2: 2 std
• Z=±3: 3 std""", 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=9)
    
    # 9. Add dataset summary at bottom
    dataset_info = f"""Dataset Summary:
• Total Records: {len(df)}
• Time Range: {min(df[time_col]).date()} to {max(df[time_col]).date() if has_time_data else 'N/A'}
• Sensors: {len(columns)}
• Contains Anomalies: {"Yes" if has_anomalies else "No"}
• Anomaly Rate: {anomaly_rate:.2f}% ({n_anomalies} records)""" if has_time_data and has_anomalies else f"""Dataset Summary:
• Total Records: {len(df)}
• Sensors: {len(columns)}"""
    
    fig.text(0.5, 0.01, dataset_info, 
             bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'),
             fontsize=10, ha='center')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.08)
    
    # Save the comprehensive visualization
    filename = f"{output_dir}/comprehensive_dataset_visualization.png"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive dataset visualization: {filename}")

def create_combined_anomaly_time_series(df, time_col, columns, output_dir):
    """
    Creates a single combined time series plot with all parameters overlaid
    and anomalies highlighted
    
    Args:
        df: DataFrame with the data
        time_col: Name of the timestamp column
        columns: List of column names to include
        output_dir: Directory to save the plot
    """
    print("Creating combined time series with anomalies...")
    
    # Check if we have anomaly data
    has_anomalies = 'Anomaly' in df.columns
    
    # Set time as index for time series plotting
    df_time = df.copy()
    
    # Create a larger figure for better readability
    plt.figure(figsize=(16, 10))
    
    # Create a colormap for the parameters
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    
    # Plot each parameter as a standardized line (z-score)
    for i, col in enumerate(columns):
        if df[col].std() > 0:  # Skip constant columns
            # Standardize the data for fair comparison
            standardized = (df[col] - df[col].mean()) / df[col].std()
            
            # Plot the standardized data
            plt.plot(df[time_col], standardized, '-', color=colors[i], 
                     linewidth=1.5, alpha=0.7, label=get_label_with_unit(col))
    
    # If we have anomalies, highlight them for each parameter
    if has_anomalies:
        anomalies = df[df['Anomaly']]
        
        # Create a legend handler for anomalies
        anomaly_types = {}
        
        # Check if we have Notes column for anomaly types
        if 'Notes' in df.columns:
            # Extract unique anomaly types
            for note in anomalies['Notes']:
                if isinstance(note, str) and "Anomaly: " in note:
                    anomaly_type = note.replace("Anomaly: ", "")
                    if anomaly_type not in anomaly_types:
                        anomaly_types[anomaly_type] = []
        
        # If no specific types, just use a generic "Anomaly" type
        if not anomaly_types:
            anomaly_types = {"Anomaly": []}
            
        # Plot each anomaly as a vertical span
        for idx, row in anomalies.iterrows():
            # Get anomaly type if available
            anomaly_type = "Anomaly"
            if 'Notes' in row and isinstance(row['Notes'], str) and "Anomaly: " in row['Notes']:
                anomaly_type = row['Notes'].replace("Anomaly: ", "")
            
            # Add a vertical line at the anomaly time
            plt.axvline(x=row[time_col], color='red', linestyle='--', alpha=0.3)
            
            # Add the timestamp to the list for this anomaly type
            if anomaly_type in anomaly_types:
                anomaly_types[anomaly_type].append(row[time_col])
        
        # Plot each anomaly point for each parameter
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
        for i, col in enumerate(columns):
            if df[col].std() > 0:
                # For each type of anomaly, mark the points differently
                for j, (anomaly_type, timestamps) in enumerate(anomaly_types.items()):
                    for timestamp in timestamps:
                        # Find the row with this timestamp
                        if timestamp in df[time_col].values:
                            # Get the standardized value for this parameter at this timestamp
                            row_idx = df[df[time_col] == timestamp].index[0]
                            value = df.iloc[row_idx][col]
                            std_value = (value - df[col].mean()) / df[col].std()
                            
                            # Plot the point with a unique marker for this anomaly type
                            marker_idx = j % len(markers)
                            plt.scatter(timestamp, std_value, 
                                      color='red', s=80, marker=markers[marker_idx],
                                      zorder=10, alpha=0.8)
        
        # Add legend entries for anomaly types
        for j, anomaly_type in enumerate(anomaly_types.keys()):
            marker_idx = j % len(markers)
            plt.scatter([], [], color='red', s=80, marker=markers[marker_idx],
                       label=f"Anomaly: {anomaly_type}")
    
    # Add grid for readability
    plt.grid(True, alpha=0.3)
    
    # Format axes
    plt.title('Combined Time Series of All Sensors with Anomalies Highlighted', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Standardized Value (Z-score)')
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    # Add horizontal line at zero (mean value)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add horizontal lines at +/- 1, 2, and 3 standard deviations
    for sd in [1, 2, 3]:
        plt.axhline(y=sd, color='gray', linestyle=':', alpha=0.3)
        plt.axhline(y=-sd, color='gray', linestyle=':', alpha=0.3)
        plt.text(plt.xlim()[0], sd, f"+{sd}σ", ha='left', va='center', color='gray', fontsize=8)
        plt.text(plt.xlim()[0], -sd, f"-{sd}σ", ha='left', va='center', color='gray', fontsize=8)
    
    # Add explanation of standardization
    plt.figtext(0.02, 0.02, "Note: All values are standardized (Z-score) for comparison across different scales.", 
               fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Create a custom legend with two columns - one for parameters, one for anomaly types
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Separate parameter handles from anomaly handles
    param_handles = handles[:len(columns)]
    param_labels = labels[:len(columns)]
    anomaly_handles = handles[len(columns):]
    anomaly_labels = labels[len(columns):]
    
    # Create a first legend for parameters
    first_legend = plt.legend(param_handles, param_labels, 
                            title="Sensors", loc='upper left', 
                            bbox_to_anchor=(1.01, 1.0))
    plt.gca().add_artist(first_legend)
    
    # Create a second legend for anomalies if present
    if anomaly_handles:
        plt.legend(anomaly_handles, anomaly_labels, 
                 title="Anomaly Types", loc='upper left', 
                 bbox_to_anchor=(1.01, 0.5))
    
    # Add annotations for major outliers
    for i, col in enumerate(columns):
        if df[col].std() > 0:
            # Find extreme values (beyond 3 standard deviations)
            standardized = (df[col] - df[col].mean()) / df[col].std()
            extreme_mask = (standardized.abs() > 3) & (~df['Anomaly'] if 'Anomaly' in df.columns else True)
            
            # Limit to at most 5 extreme points per parameter to avoid overcrowding
            extreme_indices = extreme_mask[extreme_mask].index[:5] if sum(extreme_mask) > 5 else extreme_mask[extreme_mask].index
            
            for idx in extreme_indices:
                plt.scatter(df.loc[idx, time_col], standardized.loc[idx], 
                           color=colors[i], edgecolor='black', s=60, zorder=5)
    
    plt.tight_layout()
    
    # Save the combined plot
    filename = f"{output_dir}/combined_time_series_with_anomalies.png"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved combined time series with anomalies: {filename}")
    
    # Create a more zoomed version focusing on each anomaly
    if has_anomalies and len(anomalies) > 0:
        create_anomaly_focus_plots(df, time_col, columns, anomalies, output_dir)

def create_anomaly_focus_plots(df, time_col, columns, anomalies, output_dir):
    """
    Creates zoomed-in plots focused on each anomaly period
    
    Args:
        df: DataFrame with the data
        time_col: Name of the timestamp column
        columns: List of column names to include
        anomalies: DataFrame with anomaly rows
        output_dir: Directory to save the plots
    """
    # Create output directory for focus plots
    focus_dir = os.path.join(output_dir, "anomaly_focus")
    Path(focus_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a single plot showing all anomalies in context
    plt.figure(figsize=(16, 8))
    
    # Set up colors for parameters
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    
    # Plot each parameter as a light line
    for i, col in enumerate(columns):
        if df[col].std() > 0:
            standardized = (df[col] - df[col].mean()) / df[col].std()
            plt.plot(df[time_col], standardized, '-', color=colors[i], 
                    linewidth=1, alpha=0.3)
    
    # Highlight each anomaly with a vertical band
    for idx, row in anomalies.iterrows():
        anomaly_time = row[time_col]
        
        # Create a 24-hour window around the anomaly
        window_start = anomaly_time - pd.Timedelta(hours=12)
        window_end = anomaly_time + pd.Timedelta(hours=12)
        
        # Add a shaded band for this anomaly period
        plt.axvspan(window_start, window_end, color='pink', alpha=0.3)
        
        # Mark the exact anomaly time
        plt.axvline(anomaly_time, color='red', linestyle='--')
        
        # Annotate with anomaly type if available
        if 'Notes' in row and isinstance(row['Notes'], str):
            anomaly_type = row['Notes'].replace("Anomaly: ", "")
            plt.annotate(anomaly_type, xy=(anomaly_time, plt.ylim()[1]*0.9),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', rotation=90,
                        fontsize=8, color='red')
    
    plt.title('Overview of Anomaly Periods (24-hour windows)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Standardized Value (Z-score)')
    plt.grid(True, alpha=0.3)
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    
    # Save the overview plot
    filename = f"{focus_dir}/anomaly_periods_overview.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly periods overview: {filename}")

def create_compact_distribution_plot(df, columns, output_dir):
    """
    Creates a single, compact ridge plot showing the distribution of all sensors
    with optional anomaly indicators - ideal for academic papers with limited space
    
    Args:
        df: DataFrame with the data
        columns: List of column names to include
        output_dir: Directory to save the plot
    """
    print("Creating compact ridge distribution plot for paper...")
    
    # Import joypy for ridge plots - if not installed, use alternative
    try:
        import joypy
        has_joypy = True
    except ImportError:
        has_joypy = False
        print("joypy not installed, using alternative compact visualization")
    
    # Check if we have anomaly data
    has_anomalies = 'Anomaly' in df.columns
    
    # Prepare data for plotting
    plot_data = df[columns].copy()
    
    # Add column with units for better labels
    columns_with_units = [get_label_with_unit(col) for col in columns]
    
    # Choose an appropriate colormap - viridis is good for black & white printing too
    cmap = plt.cm.viridis
    
    # Create the visualization based on available libraries
    if has_joypy:
        # Ridge plot is best for compact distribution visualization
        plt.figure(figsize=(12, 10))
        
        # Create the ridge plot
        fig, axes = joypy.joyplot(
            plot_data, 
            figsize=(12, 10), 
            colormap=cmap,
            linewidth=1,
            alpha=0.8,
            labels=columns_with_units,
            range_style='own',
            grid=True
        )
        
        # Add title
        fig.suptitle('Sensor Data Distributions', fontsize=16, y=0.97)
        
        # Add subtitle with dataset stats
        plt.figtext(
            0.5, 0.01, 
            f"Dataset: {len(df)} records | {'Includes anomalies' if has_anomalies else 'No anomalies identified'} " +
            f"| All values shown in original units", 
            ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
    else:
        # Alternative: Create a single-panel plot with KDE for each variable
        plt.figure(figsize=(12, 8))
        
        # Standardize all variables for comparison
        standardized_data = pd.DataFrame()
        for col in columns:
            if df[col].std() > 0:  # Skip constant columns
                mean = df[col].mean()
                std = df[col].std()
                standardized_data[col] = (df[col] - mean) / std
        
        # Colors for different sensors - use colormap
        colors = cmap(np.linspace(0, 1, len(columns)))
        
        # Plot standardized distribution for each sensor
        for i, col in enumerate(columns):
            if col in standardized_data.columns:
                sns.kdeplot(
                    standardized_data[col], 
                    label=columns_with_units[i],
                    color=colors[i],
                    fill=True,
                    alpha=0.3,
                    linewidth=2,
                    common_norm=False
                )
        
        # Add vertical lines for standard deviations
        for sd in [-3, -2, -1, 0, 1, 2, 3]:
            plt.axvline(
                x=sd, 
                color='gray', 
                linestyle='--', 
                alpha=0.7 if sd == 0 else 0.4,
                linewidth=1.5 if sd == 0 else 0.8
            )
            if sd != 0:
                plt.text(
                    sd, plt.ylim()[1]*0.98, 
                    f"{sd}σ", 
                    ha='center', va='top', 
                    fontsize=8, color='dimgray'
                )
        
        # Mark anomaly regions if available
        if has_anomalies:
            # For each sensor type with anomalies, add a scatter point at its zscore
            anomaly_df = df[df['Anomaly']]
            anomaly_markers = ['o', 's', '^', 'd', 'v']
            
            # Group anomalies by type if Notes available
            if 'Notes' in df.columns:
                anomaly_types = []
                for note in anomaly_df['Notes']:
                    if isinstance(note, str) and "Anomaly: " in note:
                        anomaly_types.append(note.replace("Anomaly: ", ""))
                    else:
                        anomaly_types.append("Unknown")
                
                anomaly_df['AnomalyType'] = anomaly_types
                
                # For each anomaly type, mark the z-scores of affected sensors
                for i, (atype, group) in enumerate(anomaly_df.groupby('AnomalyType')):
                    marker = anomaly_markers[i % len(anomaly_markers)]
                    
                    # For each sensor affected by this anomaly type, show its z-scores
                    for j, col in enumerate(columns):
                        if col in standardized_data.columns:
                            # Get z-scores for this column in the anomaly group
                            z_scores = standardized_data.loc[group.index, col]
                            
                            # Create a small histogram/rug at the bottom to show anomaly z-scores
                            if not z_scores.empty:
                                plt.scatter(
                                    z_scores, 
                                    [0.005] * len(z_scores),  # Small y position
                                    color=colors[j],
                                    marker=marker,
                                    s=60,
                                    alpha=0.8,
                                    label=f"{atype} ({col.split()[0]})" if j == 0 else "",
                                    zorder=5
                                )
        
        # Add grid for readability
        plt.grid(True, alpha=0.3)
        
        # Format axis and labels
        plt.title('Standardized Distributions of All Sensors', fontsize=16)
        plt.xlabel('Standardized Value (Z-score = (x - μ) / σ)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Create legend with smaller font
        plt.legend(
            fontsize=9, 
            title="Sensors",
            loc='upper left', 
            bbox_to_anchor=(1, 1)
        )
        
        # Add explanation
        explanation = """Standardization allows comparing all sensors on same scale:
• Z=0: mean value
• Z=±1: 1 standard deviation from mean
• Z=±2: 2 standard deviations (95% of normal data)
• Z=±3: 3 standard deviations (99.7% of normal data)"""
        
        plt.annotate(
            explanation,
            xy=(0.02, 0.02), 
            xycoords='figure fraction',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
        
        # Add dataset info at the bottom
        plt.figtext(
            0.5, 0.01, 
            f"Dataset: {len(df)} records | {'Includes anomalies' if has_anomalies else 'No anomalies identified'} | Z-score standardization", 
            ha='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{output_dir}/compact_distribution_for_paper.png"
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Saved compact distribution plot for paper: {filename}")

def main():
    # Path to the Excel file
    excel_path = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10\test_predictions\datasets\compressor_dataset_with_weather_2024 PER IMMAGINE.xlsx"
    
    # Output directory
    output_dir = r"C:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\Figure paper"
    
    print("Starting data distribution plot generation for the entire dataset...")
    print("-" * 80)
    
    # Generate the plots
    create_distribution_plots(excel_path, output_dir)

if __name__ == "__main__":
    main()
