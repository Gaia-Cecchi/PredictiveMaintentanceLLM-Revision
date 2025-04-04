import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, 
    mean_absolute_error, mean_squared_error,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import os
import datetime
import matplotlib.dates as mdates
from pathlib import Path
import re

def read_data(csv_path):
    """Read and preprocess the CSV data"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print column names for debugging
    print(f"Columns in the CSV file: {list(df.columns)}")
    
    # Convert datetime to proper format if it exists
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'DateTime' in df.columns:
        df['datetime'] = pd.to_datetime(df['DateTime'])
        
    # Check for is_correct column or create it if needed
    if 'is_correct' not in df.columns and 'actual_anomaly' in df.columns and 'predicted_anomaly' in df.columns:
        df['is_correct'] = df['actual_anomaly'] == df['predicted_anomaly']
        
    # Ensure boolean is_correct
    if 'is_correct' in df.columns and df['is_correct'].dtype != bool:
        df['is_correct'] = df['is_correct'].map({'True': True, 'False': False, True: True, False: False})
    
    # Handle confidence score mapping
    if 'confidence' in df.columns:
        confidence_map = {'alta': 1.0, 'high': 1.0, 'media': 0.5, 'medium': 0.5, 'bassa': 0.0, 'low': 0.0}
        df['confidence_score'] = df['confidence'].apply(lambda x: confidence_map.get(str(x).lower(), 0.5) if pd.notnull(x) else 0.5)
    elif 'prediction_probability' in df.columns:
        df['confidence_score'] = df['prediction_probability']
        df['confidence'] = pd.cut(df['prediction_probability'], 
                                bins=[0, 0.33, 0.67, 1.0], 
                                labels=['low', 'medium', 'high'])
    
    # Create binary targets for classification metrics
    if 'actual_classification' in df.columns:
        df['actual_binary'] = (df['actual_classification'] == 'ANOMALY').astype(int)
    elif 'actual_anomaly' in df.columns:
        df['actual_binary'] = df['actual_anomaly']
        df['actual_classification'] = np.where(df['actual_anomaly'] == 1, 'ANOMALY', 'NORMAL VALUE')
        
    if 'predicted_classification' in df.columns:
        df['predicted_binary'] = (df['predicted_classification'] == 'ANOMALY').astype(int)
    elif 'predicted_anomaly' in df.columns:
        df['predicted_binary'] = df['predicted_anomaly']
        df['predicted_classification'] = np.where(df['predicted_anomaly'] == 1, 'ANOMALY', 'NORMAL VALUE')
    
    # Add missing columns with defaults
    required_columns = [
        'datetime', 'actual_type', 'actual_classification', 
        'predicted_classification', 'predicted_type', 'confidence', 
        'key_indicators', 'recommendation', 'is_correct'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in the dataset. Adding with default values.")
            if col == 'actual_type':
                df[col] = np.where(df['actual_binary'] == 1, 'anomaly', 'normal')
            elif col == 'predicted_type':
                df[col] = np.where(df['predicted_binary'] == 1, 'anomaly', '')
            elif col == 'key_indicators':
                df[col] = 'No indicators available'
            elif col == 'recommendation':
                df[col] = np.where(df['predicted_binary'] == 1,
                                 'Monitor system for potential issues.',
                                 'Continue standard monitoring.')
    
    return df

def calculate_metrics(df):
    """Calculate all performance metrics"""
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = (df['is_correct'].sum() / len(df)) * 100
    
    # For binary classification metrics
    metrics['precision'] = precision_score(df['actual_binary'], df['predicted_binary'], zero_division=0)
    metrics['recall'] = recall_score(df['actual_binary'], df['predicted_binary'], zero_division=0)
    metrics['f1'] = f1_score(df['actual_binary'], df['predicted_binary'], zero_division=0)
    
    # Calculate MAE and MSE on binary classification as error metrics
    metrics['mae'] = mean_absolute_error(df['actual_binary'], df['predicted_binary'])
    metrics['mse'] = mean_squared_error(df['actual_binary'], df['predicted_binary'])
    
    # ROC AUC - use confidence scores as probability estimates
    fpr, tpr, _ = roc_curve(df['actual_binary'], df['confidence_score'])
    metrics['auc_roc'] = auc(fpr, tpr)
    metrics['roc_curve'] = (fpr, tpr)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(df['actual_binary'], df['confidence_score'])
    metrics['avg_precision'] = average_precision_score(df['actual_binary'], df['confidence_score'], average='macro')
    metrics['pr_curve'] = (precision, recall)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(df['actual_binary'], df['predicted_binary'])
    
    # Calculate by type statistics
    type_stats = df.groupby(['actual_type', 'is_correct']).size().unstack(fill_value=0)
    if True in type_stats.columns and False in type_stats.columns:
        for idx in type_stats.index:
            total = type_stats.loc[idx, True] + type_stats.loc[idx, False]
            metrics[f'{idx}_accuracy'] = (type_stats.loc[idx, True] / total) * 100 if total > 0 else 0
    
    return metrics

def plot_confusion_matrix(metrics, output_dir, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/confusion_matrix_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_roc_curve(metrics, output_dir, model_name):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc_roc"]:.3f}')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/roc_curve_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_pr_curve(metrics, output_dir, model_name):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(8, 6))
    
    precision, recall = metrics['pr_curve']
    plt.step(recall, precision, color='b', where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}\nAverage Precision: {metrics["avg_precision"]:.3f}')
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/pr_curve_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_prediction_distribution(df, output_dir, model_name):
    """Plot distribution of predictions by type"""
    plt.figure(figsize=(10, 6))
    
    # Create a DataFrame for plotting
    plot_data = df.groupby(['actual_type', 'is_correct']).size().reset_index(name='count')
    plot_data['result'] = plot_data['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Create the plot
    ax = sns.barplot(x='actual_type', y='count', hue='result', data=plot_data, palette={'Correct': 'green', 'Incorrect': 'red'})
    
    plt.title(f'Prediction Results by Data Type - {model_name}')
    plt.xlabel('Actual Data Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add numbers on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/prediction_distribution_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_confidence_accuracy(df, output_dir, model_name):
    """Plot accuracy by confidence level"""
    plt.figure(figsize=(10, 6))
    
    # Create plot data
    confidence_counts = df.groupby('confidence')['is_correct'].agg(['count', 'sum']).reset_index()
    confidence_counts['accuracy'] = (confidence_counts['sum'] / confidence_counts['count']) * 100
    
    # Sort the confidence levels appropriately (low, medium, high)
    confidence_order = {'bassa': 0, 'low': 0, 'media': 1, 'medium': 1, 'alta': 2, 'high': 2}
    confidence_counts['order'] = confidence_counts['confidence'].map(lambda x: confidence_order.get(x.lower(), 1))
    confidence_counts = confidence_counts.sort_values('order')
    
    # Create the plot
    bars = plt.bar(confidence_counts['confidence'], confidence_counts['accuracy'], color='skyblue')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, confidence_counts['count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom')
    
    plt.title(f'Prediction Accuracy by Confidence Level - {model_name}')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # Ensure there's space for the count labels
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/confidence_accuracy_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_parameter_distribution(df, output_dir, model_name):
    """Plot distribution of key parameters for correct vs incorrect predictions"""
    parameters = ['temperature', 'vibration', 'pressure', 'current']
    filenames = {}
    
    for param in parameters:
        if param in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Create violin plots
            sns.violinplot(x='is_correct', y=param, data=df, palette=['red', 'green'],
                          order=[False, True])
            
            plt.title(f'{param.capitalize()} Distribution by Prediction Result - {model_name}')
            plt.xlabel('Prediction Correct')
            plt.ylabel(param.capitalize())
            plt.xticks([0, 1], ['Incorrect', 'Correct'])
            
            # Save figure
            plt.tight_layout()
            filename = f"{output_dir}/param_{param}_{model_name}.png"
            plt.savefig(filename, dpi=300)
            filenames[param] = filename
    
    return filenames

def plot_timeline(df, output_dir, model_name):
    """Plot timeline of predictions over time"""
    plt.figure(figsize=(12, 6))
    
    # Sort by datetime
    df_sorted = df.sort_values('datetime')
    
    # Create a plot with markers for correct/incorrect predictions
    correct = df_sorted[df_sorted['is_correct']]
    incorrect = df_sorted[~df_sorted['is_correct']]
    
    plt.scatter(correct['datetime'], [1] * len(correct), color='green', marker='o', label='Correct')
    plt.scatter(incorrect['datetime'], [1] * len(incorrect), color='red', marker='x', label='Incorrect')
    
    # Add highlights for actual anomalies
    anomalies = df_sorted[df_sorted['actual_classification'] == 'ANOMALY']
    for idx, row in anomalies.iterrows():
        plt.axvline(x=row['datetime'], color='orange', alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.title(f'Timeline of Predictions - {model_name}')
    plt.ylabel('Predictions')
    plt.yticks([])  # Hide y-ticks as they're not meaningful
    plt.legend()
    
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    
    # Save figure
    plt.tight_layout()
    filename = f"{output_dir}/timeline_{model_name}.png"
    plt.savefig(filename, dpi=300)
    return filename

def plot_anomaly_distribution(df, output_dir, model_name):
    """Plot distribution of anomaly types"""
    plt.figure(figsize=(10, 6))
    
    # Count anomalies by type
    anomalies = df[df['actual_classification'] == 'ANOMALY']
    if len(anomalies) > 0 and 'predicted_type' in anomalies.columns:
        # Get counts of each anomaly type
        anomaly_counts = anomalies['predicted_type'].value_counts()
        
        # Create the plot
        anomaly_counts.plot(kind='bar', color='darkred')
        plt.title(f'Distribution of Anomaly Types - {model_name}')
        plt.xlabel('Anomaly Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, v in enumerate(anomaly_counts):
            plt.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        filename = f"{output_dir}/anomaly_distribution_{model_name}.png"
        plt.savefig(filename, dpi=300)
        return filename
    return None

def generate_html_report(df, metrics, image_files, model_name, output_dir):
    """Generate an HTML report with all metrics and plots"""
    
    # Print available columns for debugging
    print(f"Available columns in the dataset: {list(df.columns)}")
    
    # Check for required columns and provide fallbacks
    required_params = ['temperature', 'vibration', 'pressure', 'current']
    available_params = [col for col in required_params if col in df.columns]
    
    # Map potential alternative column names
    column_mapping = {
        'temperature': ['temperature', 'Temperature', 'temp'],
        'vibration': ['vibration', 'Vibration', 'vib'],
        'pressure': ['pressure', 'Pressure', 'press'],
        'current': ['current', 'Current', 'curr']
    }
    
    # Try to find alternative column names
    param_column_map = {}
    for param, alternatives in column_mapping.items():
        found = False
        for alt in alternatives:
            if alt in df.columns:
                param_column_map[param] = alt
                found = True
                break
        if not found:
            param_column_map[param] = None
    
    print(f"Parameter column mapping: {param_column_map}")

    # Function to safely access parameter values
    def get_param_value(row, param):
        col_name = param_column_map[param]
        if col_name and col_name in row:
            return f"{row[col_name]}"
        return "N/A"
    
    # Function to create parameter string for display
    def get_param_string(row):
        params = []
        if param_column_map['temperature']:
            params.append(f"Temp: {get_param_value(row, 'temperature')}°C")
        if param_column_map['vibration']:
            params.append(f"Vib: {get_param_value(row, 'vibration')} mm/s")
        if param_column_map['pressure']:
            params.append(f"Press: {get_param_value(row, 'pressure')} bar")
        if param_column_map['current']:
            params.append(f"Curr: {get_param_value(row, 'current')} A")
        
        return ", ".join(params) if params else "Parameters not available"

    # Calculate some additional statistics for the report
    total_predictions = len(df)
    correct_predictions = df['is_correct'].sum()
    accuracy = (correct_predictions / total_predictions) * 100
    
    anomaly_count = (df['actual_classification'] == 'ANOMALY').sum()
    detected_anomalies = ((df['actual_classification'] == 'ANOMALY') & 
                         (df['predicted_classification'] == 'ANOMALY')).sum()
    
    false_positives = ((df['actual_classification'] == 'NORMAL VALUE') & 
                      (df['predicted_classification'] == 'ANOMALY')).sum()
    
    # Sample correct and incorrect predictions for examples
    correct_examples = df[df['is_correct']].sample(min(5, int(correct_predictions)))
    incorrect_examples = df[~df['is_correct']].sample(min(5, len(df) - int(correct_predictions)))
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report - {model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .metrics-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 200px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .plot-container {{
            margin: 30px 0;
        }}
        .plot-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .plot-item {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .plot-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .plot-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #555;
        }}
        .timestamp {{
            margin-top: 40px;
            font-size: 12px;
            color: #999;
            text-align: right;
        }}
        .success {{
            color: green;
        }}
        .error {{
            color: red;
        }}
    </style>
</head>
<body>
    <h1>Model Evaluation Report: {model_name}</h1>
    <p>Analysis of prediction results for compressor anomaly detection.</p>
    
    <h2>Summary</h2>
    <div class="metrics-container">
        <div class="metric-card">
            <div class="metric-value">{accuracy:.1f}%</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['precision']:.3f}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['recall']:.3f}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['f1']:.3f}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['auc_roc']:.3f}</div>
            <div class="metric-label">AUC-ROC</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['mae']:.3f}</div>
            <div class="metric-label">MAE</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['mse']:.3f}</div>
            <div class="metric-label">MSE</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{detected_anomalies}/{anomaly_count}</div>
            <div class="metric-label">Anomalies Detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{false_positives}</div>
            <div class="metric-label">False Positives</div>
        </div>
    </div>
    
    <h2>Accuracy by Data Type</h2>
    <table>
        <tr>
            <th>Type</th>
            <th>Accuracy</th>
            <th>Correctly Classified</th>
            <th>Total</th>
        </tr>
"""

    # Add rows for each data type
    for data_type in df['actual_type'].unique():
        type_data = df[df['actual_type'] == data_type]
        correct = type_data['is_correct'].sum()
        total = len(type_data)
        type_accuracy = (correct / total) * 100 if total > 0 else 0
        
        html_content += f"""
        <tr>
            <td>{data_type}</td>
            <td>{type_accuracy:.1f}%</td>
            <td>{correct}</td>
            <td>{total}</td>
        </tr>"""

    html_content += """
    </table>
    
    <h2>Classification Performance</h2>
    <div class="plot-row">
        <div class="plot-item">
            <img src="{}" alt="Confusion Matrix">
            <p class="plot-caption">Confusion Matrix showing True Positives, False Positives, True Negatives, and False Negatives</p>
        </div>
        <div class="plot-item">
            <img src="{}" alt="ROC Curve">
            <p class="plot-caption">Receiver Operating Characteristic (ROC) Curve - AUC: {:.3f}</p>
        </div>
    </div>
    
    <div class="plot-row">
        <div class="plot-item">
            <img src="{}" alt="Precision-Recall Curve">
            <p class="plot-caption">Precision-Recall Curve - Average Precision: {:.3f}</p>
        </div>
        <div class="plot-item">
            <img src="{}" alt="Prediction Distribution">
            <p class="plot-caption">Distribution of Correct and Incorrect Predictions by Data Type</p>
        </div>
    </div>
    
    <h2>Confidence Analysis</h2>
    <div class="plot-row">
        <div class="plot-item">
            <img src="{}" alt="Confidence vs Accuracy">
            <p class="plot-caption">Relationship between Confidence Level and Prediction Accuracy</p>
        </div>
    </div>
    
    <h2>Parameter Distributions</h2>
    <div class="plot-row">
""".format(
        os.path.basename(image_files.get('confusion_matrix', '')),
        os.path.basename(image_files.get('roc_curve', '')),
        metrics['auc_roc'],
        os.path.basename(image_files.get('pr_curve', '')),
        metrics['avg_precision'],
        os.path.basename(image_files.get('prediction_distribution', '')),
        os.path.basename(image_files.get('confidence_accuracy', ''))
    )

    # Add parameter distribution plots
    for param in ['temperature', 'vibration', 'pressure', 'current']:
        if param in image_files:
            html_content += f"""
        <div class="plot-item">
            <img src="{os.path.basename(image_files[param])}" alt="{param.capitalize()} Distribution">
            <p class="plot-caption">{param.capitalize()} Distribution for Correct vs Incorrect Predictions</p>
        </div>"""

    html_content += """
    </div>
    
    <h2>Timeline Analysis</h2>
    <div class="plot-row">
        <div class="plot-item">
            <img src="{}" alt="Prediction Timeline">
            <p class="plot-caption">Timeline of Predictions with Highlighted Anomalies</p>
        </div>
    </div>
    
    <h2>Sample Predictions</h2>
    <h3>Correct Predictions</h3>
    <table>
        <tr>
            <th>Datetime</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Key Parameters</th>
        </tr>
""".format(os.path.basename(image_files.get('timeline', '')))

    # Add rows for correct prediction examples
    for _, row in correct_examples.iterrows():
        params = get_param_string(row)
        html_content += f"""
        <tr>
            <td>{row['datetime']}</td>
            <td>{row['actual_classification']}</td>
            <td>{row['predicted_classification']}</td>
            <td>{row['predicted_type'] if row['predicted_classification'] == 'ANOMALY' else '-'}</td>
            <td>{row['confidence']}</td>
            <td>{params}</td>
        </tr>"""

    html_content += """
    </table>
    
    <h3>Incorrect Predictions</h3>
    <table>
        <tr>
            <th>Datetime</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Key Parameters</th>
        </tr>
"""

    # Add rows for incorrect prediction examples
    for _, row in incorrect_examples.iterrows():
        params = get_param_string(row)
        html_content += f"""
        <tr>
            <td>{row['datetime']}</td>
            <td class="{'success' if row['actual_classification'] == 'NORMAL VALUE' else 'error'}">{row['actual_classification']}</td>
            <td class="{'success' if row['predicted_classification'] == 'NORMAL VALUE' else 'error'}">{row['predicted_classification']}</td>
            <td>{row['predicted_type'] if row['predicted_classification'] == 'ANOMALY' else '-'}</td>
            <td>{row['confidence']}</td>
            <td>{params}</td>
        </tr>"""

    html_content += f"""
    </table>
    
    <div class="timestamp">
        <p>Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""

    # Write HTML to file
    html_path = f"{output_dir}/model_evaluation_{model_name}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

def main():
    # Ask for model name
    model_name = input("Enter the name of the model for this evaluation (default: CNN Model): ") or "CNN Model"
    model_name = re.sub(r'[^\w\s-]', '', model_name)  # Sanitize input
    
    # Set paths for CNN output
    csv_path = input("Enter the path to the CSV results file (default: test_predictions/cnn/results/cnn_predictions.csv): ") or "test_predictions/cnn/results/cnn_predictions.csv"
    output_dir = input("Enter the directory for output reports (default: test_predictions/cnn/results/report): ") or "test_predictions/cnn/results/report"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if the file exists before trying to read it
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        print("Possible locations to check:")
        for possible_path in [
            "test_predictions/cnn/results/cnn_predictions.csv",
            "test_predictions/cnn/results/full_analysis_results.csv"
        ]:
            if os.path.exists(possible_path):
                print(f" - {possible_path} (EXISTS)")
            else:
                print(f" - {possible_path} (NOT FOUND)")
        
        # Ask for an alternative path
        alt_path = input("Enter an alternative path or press Enter to exit: ")
        if not alt_path:
            print("Exiting program.")
            return
        csv_path = alt_path
    
    # Read data
    print(f"Reading data from {csv_path}...")
    df = read_data(csv_path)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(df)
    
    # Generate plots and save to files
    print("Generating plots...")
    image_files = {}
    
    # Confusion matrix
    image_files['confusion_matrix'] = plot_confusion_matrix(metrics, output_dir, model_name)
    
    # ROC curve
    image_files['roc_curve'] = plot_roc_curve(metrics, output_dir, model_name)
    
    # Precision-Recall curve
    image_files['pr_curve'] = plot_pr_curve(metrics, output_dir, model_name)
    
    # Prediction distribution
    image_files['prediction_distribution'] = plot_prediction_distribution(df, output_dir, model_name)
    
    # Confidence vs accuracy
    image_files['confidence_accuracy'] = plot_confidence_accuracy(df, output_dir, model_name)
    
    # Parameter distributions
    param_files = plot_parameter_distribution(df, output_dir, model_name)
    for param, filepath in param_files.items():
        image_files[param] = filepath
    
    # Timeline
    image_files['timeline'] = plot_timeline(df, output_dir, model_name)
    
    # Anomaly distribution
    anomaly_dist = plot_anomaly_distribution(df, output_dir, model_name)
    if anomaly_dist:
        image_files['anomaly_distribution'] = anomaly_dist
    
    # Generate HTML report
    print("Generating HTML report...")
    html_path = generate_html_report(df, metrics, image_files, model_name, output_dir)
    
    print(f"Evaluation complete! Report saved to: {html_path}")
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()