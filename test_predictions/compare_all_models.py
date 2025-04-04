import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime
from pathlib import Path
import re

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_metrics_file(file_path):
    """Load metrics from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics file {file_path}: {e}")
        return None

def find_latest_metrics(base_dir, model_name, pattern="*_metrics_*.json"):
    """Find the most recent metrics file for a model"""
    search_path = os.path.join(base_dir, model_name, "results", pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No metrics files found for {model_name} in {search_path}")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Found latest metrics for {model_name}: {os.path.basename(latest_file)}")
    return latest_file

def load_qwen_metrics(qwen_results_dir):
    """Load QWEN metrics from its specific format"""
    try:
        # First look for metrics_*.json files
        files = glob.glob(os.path.join(qwen_results_dir, "metrics_*.json"))
        
        # If none found, try metrics.json
        if not files:
            metrics_file = os.path.join(qwen_results_dir, "metrics.json")
            if os.path.exists(metrics_file):
                files = [metrics_file]
        
        # If still none found, try to extract metrics from CSV
        if not files:
            csv_path = os.path.join(qwen_results_dir, "prediction_results.csv")
            if os.path.exists(csv_path):
                print(f"No metrics JSON found, calculating metrics from {csv_path}")
                metrics = calculate_metrics_from_csv(csv_path)
                if metrics:
                    return metrics
            
            print(f"No QWEN metrics files found in {qwen_results_dir}")
            return None
            
        latest_file = max(files, key=os.path.getmtime)
        print(f"Found QWEN metrics: {os.path.basename(latest_file)}")
        metrics = load_metrics_file(latest_file)
        
        if metrics:
            # Standardize metrics format
            standardized_metrics = {
                'accuracy': metrics.get('accuracy', 0) * 100 if metrics.get('accuracy', 0) < 1 else metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'auc_roc': metrics.get('auc_roc', 0),
                'specificity': metrics.get('specificity', 0),
                'tn': metrics.get('tn', 0),
                'fp': metrics.get('fp', 0),
                'fn': metrics.get('fn', 0),
                'tp': metrics.get('tp', 0),
                'model_name': 'QWEN2.5',
                'file': latest_file
            }
            return standardized_metrics
        return None
    except Exception as e:
        print(f"Error loading QWEN metrics: {e}")
        return None

def calculate_metrics_from_csv(csv_path):
    """Calculate metrics from a prediction CSV file"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_cols = ['actual_class', 'prediction']
        if not all(col in df.columns for col in required_cols):
            rename_map = {
                'Actual': 'actual_class',
                'Predicted': 'prediction',
                'actual_classification': 'actual_class',
                'predicted_classification': 'prediction'
            }
            df = df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})
        
        # Check again after potential renaming
        if not all(col in df.columns for col in required_cols):
            # Try to identify possible column names
            print(f"Required columns not found in CSV. Available columns: {df.columns.tolist()}")
            return None
        
        # Convert to binary format for metric calculation
        df['actual_binary'] = df['actual_class'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
        df['prediction_binary'] = df['prediction'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            roc_auc_score, confusion_matrix
        )
        
        y_true = df['actual_binary'].values
        y_pred = df['prediction_binary'].values
        
        # For ROC AUC we need probabilities, if they exist
        if 'confidence_score' in df.columns:
            y_scores = df['confidence_score'].values
        elif 'probability' in df.columns:
            y_scores = df['probability'].values
        else:
            y_scores = y_pred  # Use binary predictions if no probabilities
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,  # Convert to percentage
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': specificity,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'model_name': 'QWEN2.5',
            'file': csv_path
        }
        
        # Try to calculate ROC AUC if we have meaningful scores
        try:
            if len(set(y_scores)) > 1:  # Only calculate if we have more than one value
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            else:
                metrics['auc_roc'] = 0.5  # Default value when prediction is constant
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
            metrics['auc_roc'] = 0.5
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics from CSV: {e}")
        return None

def find_latest_prediction_file(base_dir, model_name, pattern="*_predictions_*.csv"):
    """Find the most recent prediction results CSV file for a model"""
    search_path = os.path.join(base_dir, model_name, "results", pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No prediction files found for {model_name} in {search_path}")
        return None
    
    # Sort by modification time (newest first)
    latest_file = max(files, key=os.path.getmtime)
    print(f"Found latest predictions for {model_name}: {os.path.basename(latest_file)}")
    return latest_file

def find_qwen_prediction_file(qwen_results_dir):
    """Find the Qwen prediction file"""
    # Try common filenames for Qwen predictions
    possible_files = [
        os.path.join(qwen_results_dir, "prediction_results.csv"),
        os.path.join(qwen_results_dir, "predictions_results.csv"),
        os.path.join(qwen_results_dir, "qwen_predictions.csv"),
        os.path.join(qwen_results_dir, "predictions.csv")
    ]
    
    # Also search for any CSV files
    csv_files = glob.glob(os.path.join(qwen_results_dir, "*.csv"))
    if csv_files:
        possible_files.extend(csv_files)
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Found QWEN predictions: {os.path.basename(file_path)}")
            return file_path
    
    print(f"No prediction files found for QWEN in {qwen_results_dir}")
    return None

def plot_metrics_comparison(all_metrics, output_dir):
    """Create bar plots comparing different metrics across models"""
    # Extract metrics for comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'specificity']
    model_names = [m['model_name'] for m in all_metrics]
    
    # Create directory for comparison results
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each metric
    for metric in metrics_to_compare:
        plt.figure(figsize=(10, 6))
        
        # Normalize values: all metrics should be displayed as percentages (0-100)
        values = []
        for m in all_metrics:
            value = m.get(metric, 0)
            # Convert all values to percentage format for display consistency
            if value <= 1.0:
                value = value * 100
            values.append(value)
        
        bars = plt.bar(model_names, values, color=colors[:len(model_names)])
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"{value:.1f}%", ha='center', va='bottom')
        
        plt.title(f'Comparison of {metric.upper()}', fontsize=15)
        plt.ylabel(f'{metric.upper()} (%)')
        plt.ylim(0, max(max(values) * 1.1, 100))  # Add 10% padding or go to 100%
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=300)
        plt.close()
    
    # Create a radar chart for overall comparison
    create_radar_chart(all_metrics, metrics_to_compare, output_dir)
    
    # Create a summary table
    create_summary_table(all_metrics, metrics_to_compare, output_dir)

def create_radar_chart(all_metrics, metrics_to_compare, output_dir):
    """Create a radar chart comparing all models across multiple metrics"""
    # Number of metrics to compare
    N = len(metrics_to_compare)
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set angles for each metric (evenly spaced)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize positions and labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], [m.upper() for m in metrics_to_compare], fontsize=12)
    
    # Draw axis lines for each angle
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, metrics in enumerate(all_metrics):
        # Extract values and ensure they are normalized between 0 and 1
        values = []
        for metric in metrics_to_compare:
            value = metrics.get(metric, 0)
            # Normalize all values to 0-1 scale for the radar chart
            value = value / 100 if value > 1 else value
            values.append(value)
        
        # Close the loop
        values += values[:1]
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=metrics['model_name'], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison', size=16, y=1.1)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_metrics, metrics_to_compare, output_dir):
    """Create a summary table of all metrics"""
    # Create a DataFrame for the table
    table_data = []
    for metrics in all_metrics:
        row = {'Model': metrics['model_name']}
        for metric in metrics_to_compare:
            value = metrics.get(metric, 0)
            # Normalize all values to percentage format for consistent display
            value = value * 100 if value <= 1 else value
            row[metric.upper()] = f"{value:.2f}%"
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary table saved to {csv_path}")
    
    # Also create an HTML version with formatting
    html_path = os.path.join(output_dir, 'metrics_summary.html')
    with open(html_path, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predictive Maintenance Model Comparison</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }
                th, td {
                    border: 1px solid #e0e0e0;
                    padding: 12px;
                    text-align: center;
                }
                th {
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }
                tr:nth-child(even) {
                    background-color: #f2f7ff;
                }
                tr:hover {
                    background-color: #e9f2ff;
                }
                .header {
                    text-align: center;
                    padding: 20px 0;
                    margin-bottom: 30px;
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    border-radius: 8px;
                }
                .header h1 {
                    color: #ecf0f1;
                    margin-bottom: 10px;
                }
                .header p {
                    font-size: 16px;
                    opacity: 0.8;
                }
                .metrics-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                .metric-card {
                    background-color: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .metric-title {
                    font-weight: bold;
                    color: #3498db;
                    margin-bottom: 8px;
                }
                .metric-description {
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-bottom: 15px;
                }
                .charts-section {
                    margin: 40px 0;
                }
                .chart-container {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }
                .chart-title {
                    text-align: center;
                    margin-bottom: 20px;
                    color: #2c3e50;
                }
                .chart-img {
                    width: 100%;
                    height: auto;
                    max-width: 100%;
                    display: block;
                    margin: 0 auto;
                }
                .footer {
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #e0e0e0;
                    color: #7f8c8d;
                }
                .metrics-description {
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 30px 0;
                }
                .metrics-description h3 {
                    color: #3498db;
                    margin-bottom: 15px;
                }
                .metrics-description ul {
                    padding-left: 20px;
                }
                .metrics-description li {
                    margin-bottom: 10px;
                }
                .best-value {
                    font-weight: bold;
                    color: #27ae60;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Predictive Maintenance Model Comparison</h1>
                <p>Comparative analysis of different machine learning models for anomaly detection</p>
                <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
            
            <h2>Performance Summary</h2>
            <p>
                This report compares several machine learning models used for predictive maintenance
                on compressor monitoring data. The goal is to detect anomalies and potential failures
                before they cause equipment downtime.
            </p>
        """)
        
        # Add metrics grid with descriptions
        f.write("""
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-title">ACCURACY</div>
                    <div class="metric-description">Percentage of correct predictions (both normal and anomaly)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">PRECISION</div>
                    <div class="metric-description">Proportion of positive identifications (anomalies) that were actually correct</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">RECALL</div>
                    <div class="metric-description">Proportion of actual positives (anomalies) that were identified correctly</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">F1 SCORE</div>
                    <div class="metric-description">Harmonic mean of precision and recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">AUC-ROC</div>
                    <div class="metric-description">Area Under the Receiver Operating Characteristic curve</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">SPECIFICITY</div>
                    <div class="metric-description">Proportion of actual negatives correctly identified</div>
                </div>
            </div>
            
            <h2>Metrics Comparison</h2>
        """)
        
        # Add the table with highlighted best values
        # First, find the best value for each metric
        best_values = {}
        for metric in metrics_to_compare:
            values = [m.get(metric, 0) * 100 if m.get(metric, 0) <= 1 else m.get(metric, 0) for m in all_metrics]
            best_values[metric.upper()] = max(values)
        
        # Convert DataFrame to HTML with custom formatting to highlight best values
        html_table = df.to_html(index=False)
        
        # For each model and metric, check if it's the best value and highlight it
        for metric in metrics_to_compare:
            metric_upper = metric.upper()
            for metrics in all_metrics:
                model_name = metrics['model_name']
                value = metrics.get(metric, 0) * 100 if metrics.get(metric, 0) <= 1 else metrics.get(metric, 0)
                if abs(value - best_values[metric_upper]) < 0.01:  # If it's very close to the best value
                    # Replace the value with a highlighted version
                    formatted_value = f"{value:.2f}%"
                    html_table = html_table.replace(
                        f"<td>{formatted_value}</td>", 
                        f'<td class="best-value">{formatted_value}</td>'
                    )
        
        f.write(html_table)
        
        # Add chart sections
        f.write("""
            <div class="charts-section">
                <h2>Visual Comparison</h2>
                
                <div class="chart-container">
                    <h3 class="chart-title">Radar Chart - Overall Model Performance</h3>
                    <img class="chart-img" src="radar_comparison.png" alt="Radar Comparison">
                    <p>This radar chart shows all metrics for each model on a scale from 0% to 100%. A larger area indicates better overall performance.</p>
                </div>
                
                <div class="chart-container">
                    <h3 class="chart-title">Confusion Matrix Comparison</h3>
                    <img class="chart-img" src="confusion_matrices_comparison.png" alt="Confusion Matrices">
                    <p>Confusion matrices show true positives, false positives, true negatives, and false negatives for each model.</p>
                </div>
                
                <div class="chart-container">
                    <h3 class="chart-title">Anomaly Detection Timeline</h3>
                    <img class="chart-img" src="anomaly_detection_timeline.png" alt="Detection Timeline">
                    <p>This timeline compares when each model detected anomalies. Vertical dashed lines represent actual anomalies.</p>
                </div>
        """)
        
        # Add individual metric comparison charts
        f.write('<h2>Detailed Metric Comparisons</h2>')
        f.write('<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">')
        
        for metric in metrics_to_compare:
            f.write(f"""
                <div class="chart-container">
                    <h3 class="chart-title">{metric.upper()} Comparison</h3>
                    <img class="chart-img" src="comparison_{metric}.png" alt="{metric} Comparison">
                </div>
            """)
        
        f.write('</div>')
        
        # Add explanation of metrics
        f.write("""
            <div class="metrics-description">
                <h3>Understanding the Metrics</h3>
                <ul>
                    <li><strong>ACCURACY</strong>: The percentage of all predictions (both normal and anomaly) that were correct. While a high accuracy is generally good, it can be misleading in imbalanced datasets where anomalies are rare.</li>
                    <li><strong>PRECISION</strong>: When the model predicts an anomaly, how often is it correct? High precision means few false alarms.</li>
                    <li><strong>RECALL</strong>: What percentage of actual anomalies did the model detect? High recall means few missed anomalies.</li>
                    <li><strong>F1 SCORE</strong>: The harmonic mean of precision and recall, providing a single metric that balances both concerns.</li>
                    <li><strong>AUC-ROC</strong>: Area Under the Receiver Operating Characteristic curve, measuring the model's ability to discriminate between normal and anomaly classes across different threshold settings.</li>
                    <li><strong>SPECIFICITY</strong>: The proportion of actual normal samples correctly identified as normal.</li>
                </ul>
                <p>For predictive maintenance, a balance between precision and recall is often critical. High precision reduces unnecessary maintenance checks, while high recall ensures fewer missed failures.</p>
            </div>
            
            <div class="footer">
                <p>Analysis performed using Python with scikit-learn, TensorFlow, and custom machine learning models</p>
                <p>&copy; """ + datetime.now().strftime("%Y") + """ UDOO Lab - Predictive Maintenance & LLMs</p>
            </div>
        </body>
        </html>
        """)
    
    print(f"Enhanced HTML summary table saved to {html_path}")

def plot_confusion_matrices(all_metrics, prediction_files, output_dir):
    """Create a figure showing all confusion matrices side by side"""
    # Only proceed if we have prediction files
    valid_models = [m for m, p in zip(all_metrics, prediction_files) if p is not None]
    valid_files = [p for p in prediction_files if p is not None]
    
    if not valid_files:
        print("No valid prediction files found for confusion matrix comparison")
        return
    
    # Create a 2x2 grid of confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metrics, file, ax) in enumerate(zip(valid_models, valid_files, axes)):
        model_name = metrics['model_name']
        
        # Extract confusion matrix values
        tn = metrics.get('tn', 0)
        fp = metrics.get('fp', 0)
        fn = metrics.get('fn', 0)
        tp = metrics.get('tp', 0)
        
        # Create confusion matrix
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot on the corresponding axis
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} Confusion Matrix')
    
    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), dpi=300)
    plt.close()

def plot_anomaly_detection_comparison(prediction_files, output_dir):
    """Compare how different models detect anomalies on a timeline"""
    valid_files = [p for p in prediction_files if p is not None]
    
    if not valid_files:
        print("No valid prediction files found for anomaly detection comparison")
        return
    
    # Load each prediction file
    dfs = []
    model_names = []
    
    for file in valid_files:
        try:
            df = pd.read_csv(file)
            
            # Determine model name from file path
            if 'qwen' in file.lower():
                model_name = 'QWEN2.5'
            elif 'lstm' in file.lower():
                model_name = 'LSTM'
            elif 'ann' in file.lower():
                model_name = 'ANN'
            elif 'cnn' in file.lower():
                model_name = 'CNN'
            else:
                model_name = 'Unknown'
            
            # Ensure datetime column is properly formatted
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'DateTime' in df.columns:
                df['datetime'] = pd.to_datetime(df['DateTime'])
            
            # Standardize anomaly column names
            if 'predicted_anomaly' in df.columns:
                pass
            elif 'predicted_binary' in df.columns:
                df['predicted_anomaly'] = df['predicted_binary']
            elif 'prediction' in df.columns:
                df['predicted_anomaly'] = df['prediction'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
            
            if 'actual_anomaly' in df.columns:
                pass
            elif 'actual_binary' in df.columns:
                df['actual_anomaly'] = df['actual_binary']
            elif 'actual_class' in df.columns:
                df['actual_anomaly'] = df['actual_class'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
            
            dfs.append(df)
            model_names.append(model_name)
            
        except Exception as e:
            print(f"Error loading prediction file {file}: {e}")
    
    if not dfs:
        print("No valid prediction dataframes could be loaded")
        return
    
    # Create a timeline plot
    plt.figure(figsize=(15, 10))
    
    # First, plot the actual anomalies (should be the same in all files)
    reference_df = dfs[0]
    if 'actual_anomaly' in reference_df.columns and 'datetime' in reference_df.columns:
        anomaly_dates = reference_df[reference_df['actual_anomaly'] == 1]['datetime']
        for date in anomaly_dates:
            plt.axvline(x=date, color='grey', alpha=0.3, linestyle='--')
    
    # Plot predicted anomalies for each model
    for i, (df, model_name) in enumerate(zip(dfs, model_names)):
        if 'predicted_anomaly' in df.columns and 'datetime' in df.columns:
            # Plot points where anomalies were predicted
            anomaly_points = df[df['predicted_anomaly'] == 1]
            plt.scatter(
                anomaly_points['datetime'], 
                [i+1] * len(anomaly_points), 
                label=f"{model_name} Predicted Anomalies",
                marker='o',
                s=50,
                color=colors[i % len(colors)]
            )
    
    plt.yticks(range(1, len(dfs)+1), model_names)
    plt.title('Anomaly Detection Timeline Comparison', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Model')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis date labels
    plt.gcf().autofmt_xdate()
    
    # Add a legend for actual anomalies
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='grey', lw=1, linestyle='--', label='Actual Anomaly')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_detection_timeline.png'), dpi=300)
    plt.close()

def standardize_prediction_dataframe(df):
    """Standardize column names and formats in prediction dataframes"""
    # Standardize datetime column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'DateTime' in df.columns:
        df['datetime'] = pd.to_datetime(df['DateTime'])
        
    # Standardize actual class columns
    if 'actual_class' in df.columns:
        df['actual_binary'] = df['actual_class'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
    elif 'actual_classification' in df.columns:
        df['actual_binary'] = df['actual_classification'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
    elif 'actual_anomaly' in df.columns:
        df['actual_binary'] = df['actual_anomaly']
        
    # Standardize predicted class columns
    if 'prediction' in df.columns:
        df['predicted_binary'] = df['prediction'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
    elif 'predicted_classification' in df.columns:
        df['predicted_binary'] = df['predicted_classification'].apply(lambda x: 1 if x == 'ANOMALY' else 0)
    elif 'predicted_anomaly' in df.columns:
        df['predicted_binary'] = df['predicted_anomaly']
        
    return df

def calculate_common_subset_metrics(prediction_files):
    """Calculate metrics for a common subset of records across all models"""
    if not prediction_files or None in prediction_files:
        print("Cannot calculate common subset metrics: missing prediction files")
        return None, None
    
    # Load all prediction dataframes
    dfs = []
    for file in prediction_files:
        try:
            df = pd.read_csv(file)
            df = standardize_prediction_dataframe(df)
            
            # Store model type in the dataframe
            if 'qwen' in file.lower():
                df['model'] = 'QWEN2.5'
            elif 'lstm' in file.lower():
                df['model'] = 'LSTM'
            elif 'ann' in file.lower():
                df['model'] = 'ANN'
            elif 'cnn' in file.lower():
                df['model'] = 'CNN'
            else:
                df['model'] = 'Unknown'
                
            dfs.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return None, None
    
    # Find the smallest dataframe (Qwen) and its datetime records
    qwen_df = next((df for df in dfs if df['model'].iloc[0] == 'QWEN2.5'), None)
    if qwen_df is None:
        print("Could not find Qwen dataframe")
        return None, None
    
    print(f"Qwen analyzed {len(qwen_df)} records - using this as reference subset")
    qwen_dates = set(qwen_df['datetime'])
    
    # Filter other dataframes to match the Qwen subset
    filtered_dfs = []
    for df in dfs:
        if df['model'].iloc[0] == 'QWEN2.5':
            filtered_dfs.append(df)  # Keep Qwen as is
        else:
            # Filter to only keep records with matching dates
            filtered_df = df[df['datetime'].isin(qwen_dates)].copy()
            print(f"Filtered {df['model'].iloc[0]} from {len(df)} to {len(filtered_df)} records")
            if len(filtered_df) < len(qwen_dates):
                print(f"Warning: {df['model'].iloc[0]} is missing {len(qwen_dates) - len(filtered_df)} records from Qwen's subset")
            filtered_dfs.append(filtered_df)
    
    # Calculate metrics for each filtered dataframe
    common_metrics = []
    filtered_prediction_files = []
    
    for i, df in enumerate(filtered_dfs):
        model_name = df['model'].iloc[0]
        
        # Skip if not enough data after filtering
        if len(df) < 10:
            print(f"Not enough data for {model_name} after filtering")
            continue
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, 
            roc_auc_score, confusion_matrix
        )
        
        y_true = df['actual_binary'].values
        y_pred = df['predicted_binary'].values
        
        # For ROC AUC we need probabilities, if they exist
        if 'confidence_score' in df.columns:
            y_scores = df['confidence_score'].values
        elif 'prediction_probability' in df.columns:
            y_scores = df['prediction_probability'].values
        else:
            y_scores = y_pred  # Use binary predictions if no probabilities
        
        # Calculate confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred) * 100,  # Convert to percentage
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': specificity,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'model_name': model_name,
            'file': prediction_files[i]
        }
        
        # Try to calculate ROC AUC if we have meaningful scores
        try:
            if len(set(y_scores)) > 1:  # Only calculate if we have more than one value
                metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
            else:
                metrics['auc_roc'] = 0.5  # Default value when prediction is constant
        except Exception as e:
            print(f"Could not calculate ROC AUC for {model_name}: {e}")
            metrics['auc_roc'] = 0.5
        
        common_metrics.append(metrics)
        
        # Save filtered dataframe as a temporary file for visualizations
        temp_file = os.path.join(os.path.dirname(prediction_files[i]), f"filtered_{model_name.lower()}_common_subset.csv")
        df.to_csv(temp_file, index=False)
        filtered_prediction_files.append(temp_file)
    
    return common_metrics, filtered_prediction_files

def main():
    # Base directory for all model results
    base_dir = "test_predictions"
    output_dir = os.path.join(base_dir, "model_comparison")
    common_subset_dir = os.path.join(output_dir, "common_subset")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(common_subset_dir, exist_ok=True)
    
    print("===== Comparing All Predictive Maintenance Models =====")
    
    # Load metrics for each model
    all_metrics = []
    all_prediction_files = []
    
    # QWEN2.5 metrics (use the correct path)
    qwen_dir = os.path.join(base_dir, "llm", "results_qwen-2.5.32b")
    print(f"Loading QWEN2.5 metrics from: {qwen_dir}")
    
    # First find the prediction file to make sure it exists
    qwen_pred_file = find_qwen_prediction_file(qwen_dir)
    
    # If we found a prediction file, load or calculate the metrics
    if qwen_pred_file:
        qwen_metrics = load_qwen_metrics(qwen_dir)
        if not qwen_metrics:
            print("Attempting to calculate metrics directly from the prediction file...")
            qwen_metrics = calculate_metrics_from_csv(qwen_pred_file)
            
        if qwen_metrics:
            all_metrics.append(qwen_metrics)
            all_prediction_files.append(qwen_pred_file)
            print(f"Successfully loaded QWEN2.5 metrics and predictions")
        else:
            print("Warning: Could not load or calculate QWEN2.5 metrics.")
    else:
        print(f"Warning: Could not find QWEN2.5 prediction file in {qwen_dir}")
    
    # Neural network models (LSTM, ANN, CNN)
    models = ["lstm", "ann", "cnn"]
    
    for model in models:
        metrics_file = find_latest_metrics(base_dir, model)
        if metrics_file:
            metrics = load_metrics_file(metrics_file)
            if metrics:
                # Add model name
                metrics['model_name'] = model.upper()
                all_metrics.append(metrics)
                
                # Find corresponding prediction file
                pred_file = find_latest_prediction_file(base_dir, model)
                all_prediction_files.append(pred_file)
        else:
            all_prediction_files.append(None)
    
    # Check if we have any metrics to compare
    if not all_metrics:
        print("No metrics found for any model. Exiting.")
        return
    
    # Generate regular comparison visualizations
    print(f"Generating comparison visualizations in {output_dir}")
    plot_metrics_comparison(all_metrics, output_dir)
    plot_confusion_matrices(all_metrics, all_prediction_files, output_dir)
    plot_anomaly_detection_comparison(all_prediction_files, output_dir)
    
    # Generate comparison for common subset of records
    print(f"\n===== Comparing Models on Common Subset (Qwen's 1133 records) =====")
    common_metrics, filtered_prediction_files = calculate_common_subset_metrics(all_prediction_files)
    
    if common_metrics:
        # Generate comparison visualizations for common subset
        print(f"Generating common subset comparison visualizations in {common_subset_dir}")
        plot_metrics_comparison(common_metrics, common_subset_dir)
        plot_confusion_matrices(common_metrics, filtered_prediction_files, common_subset_dir)
        plot_anomaly_detection_comparison(filtered_prediction_files, common_subset_dir)
        
        print("\nCommon subset analysis complete!")
    else:
        print("Could not perform common subset analysis due to errors.")
    
    print("\nComparison complete! Results saved to:", output_dir)
    print("\nSummary of models compared (on full dataset):")
    for i, metrics in enumerate(all_metrics):
        print(f"- {metrics['model_name']}")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            value = metrics.get(metric, 0)
            # Normalize all values to percentage for display
            if value <= 1:
                value = value * 100
            print(f"  {metric.upper()}: {value:.2f}%")
    
    if common_metrics:
        print("\nSummary of models compared (on common subset of 1133 records):")
        for i, metrics in enumerate(common_metrics):
            print(f"- {metrics['model_name']}")
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                value = metrics.get(metric, 0)
                # Normalize all values to percentage for display
                if value <= 1:
                    value = value * 100
                print(f"  {metric.upper()}: {value:.2f}%")
    
    # Clean up temporary files
    if filtered_prediction_files:
        for file in filtered_prediction_files:
            try:
                os.remove(file)
            except:
                pass

if __name__ == "__main__":
    main()
