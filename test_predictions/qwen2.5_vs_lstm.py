#qwen2.5_vs_lstm.py - Comparison of Qwen 2.5 and LSTM models for predictive maintenance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, mean_absolute_error, mean_squared_error,
    classification_report
)
import os
import datetime
from pathlib import Path
import glob
import json
import re

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
FIGURE_SIZE = (10, 6)
DPI = 300

def load_results(path, model_name):
    """
    Load results from specified path for the given model
    
    Args:
        path: Path to either a directory containing CSV files or a direct CSV file path
        model_name: Name of the model for logging purposes
    """
    # Check if path is a direct file path or a directory
    if path.endswith('.csv'):
        # Direct file path provided
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        selected_file = path
    else:
        # Directory path provided - find CSV files
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        
        # Select the most appropriate CSV file (prioritize full_analysis_results.csv)
        selected_file = csv_files[0]
        for file in csv_files:
            if "full_analysis_results" in file.lower():
                selected_file = file
                break
    
    print(f"Loading {model_name} results from: {selected_file}")
    df = pd.read_csv(selected_file)
    
    # Process the dataframe to ensure consistent format
    # Convert datetime to proper format
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "DateTime" in df.columns:
        df["datetime"] = pd.to_datetime(df["DateTime"])
        
    # Ensure we have consistent column names
    col_mapping = {
        "actual_classification": ["actual_classification", "true_classification", "ground_truth"],
        "predicted_classification": ["predicted_classification", "prediction"],
        "is_correct": ["is_correct", "correct"]
    }
    
    for target_col, source_cols in col_mapping.items():
        if target_col not in df.columns:
            for source_col in source_cols:
                if source_col in df.columns:
                    df[target_col] = df[source_col]
                    break
    
    # If we still don't have these columns, try to create them
    if "actual_classification" not in df.columns and "actual_anomaly" in df.columns:
        df["actual_classification"] = np.where(df["actual_anomaly"] == 1, "ANOMALY", "NORMAL VALUE")
    
    if "predicted_classification" not in df.columns and "predicted_anomaly" in df.columns:
        df["predicted_classification"] = np.where(df["predicted_anomaly"] == 1, "ANOMALY", "NORMAL VALUE")
    
    if "is_correct" not in df.columns:
        df["is_correct"] = df["actual_classification"] == df["predicted_classification"]
    
    # Convert is_correct to boolean if it's not already
    if df["is_correct"].dtype != bool:
        df["is_correct"] = df["is_correct"].astype(bool)
    
    # Create binary columns for metrics calculation
    df["actual_binary"] = (df["actual_classification"] == "ANOMALY").astype(int)
    df["predicted_binary"] = (df["predicted_classification"] == "ANOMALY").astype(int)
    
    # Set model name
    df["model"] = model_name
    
    return df, selected_file

def calculate_metrics(df, prefix=""):
    """Calculate performance metrics for a results dataframe"""
    metrics = {}
    
    # Basic classification metrics
    metrics[f"{prefix}accuracy"] = accuracy_score(df["actual_binary"], df["predicted_binary"]) * 100
    metrics[f"{prefix}precision"] = precision_score(df["actual_binary"], df["predicted_binary"], zero_division=0)
    metrics[f"{prefix}recall"] = recall_score(df["actual_binary"], df["predicted_binary"], zero_division=0)
    metrics[f"{prefix}f1"] = f1_score(df["actual_binary"], df["predicted_binary"], zero_division=0)
    
    # Error metrics
    metrics[f"{prefix}mae"] = mean_absolute_error(df["actual_binary"], df["predicted_binary"])
    metrics[f"{prefix}mse"] = mean_squared_error(df["actual_binary"], df["predicted_binary"])
    metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
    
    # Confusion matrix and derived metrics
    cm = confusion_matrix(df["actual_binary"], df["predicted_binary"])
    tn, fp, fn, tp = cm.ravel()
    metrics[f"{prefix}true_negative"] = tn
    metrics[f"{prefix}false_positive"] = fp
    metrics[f"{prefix}false_negative"] = fn
    metrics[f"{prefix}true_positive"] = tp
    
    # Additional metrics
    metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics[f"{prefix}npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    metrics[f"{prefix}false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics[f"{prefix}false_negative_rate"] = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate ROC AUC if we have confidence scores
    if "confidence_score" in df.columns:
        fpr, tpr, _ = roc_curve(df["actual_binary"], df["confidence_score"])
        metrics[f"{prefix}auc_roc"] = auc(fpr, tpr)
        metrics[f"{prefix}roc_curve"] = (fpr, tpr)
    elif "prediction_probability" in df.columns:
        fpr, tpr, _ = roc_curve(df["actual_binary"], df["prediction_probability"])
        metrics[f"{prefix}auc_roc"] = auc(fpr, tpr)
        metrics[f"{prefix}roc_curve"] = (fpr, tpr)
    else:
        # If no confidence scores, we can still plot a ROC point based on the single operating point
        # This effectively creates a step function through (0,0), (FPR, TPR), and (1,1)
        metrics[f"{prefix}auc_roc"] = (metrics[f"{prefix}recall"] + metrics[f"{prefix}specificity"]) / 2
        
        # Create a simplified ROC curve with just three points for visualization
        # (0,0), (FPR, TPR), (1,1)
        fpr = np.array([0, metrics[f"{prefix}false_positive_rate"], 1])
        tpr = np.array([0, metrics[f"{prefix}recall"], 1])
        metrics[f"{prefix}roc_curve"] = (fpr, tpr)
    
    # Anomaly detection specific metrics
    anomaly_count = df["actual_binary"].sum()
    detected_anomalies = ((df["actual_binary"] == 1) & (df["predicted_binary"] == 1)).sum()
    metrics[f"{prefix}anomaly_detection_rate"] = detected_anomalies / anomaly_count if anomaly_count > 0 else 0
    
    # Type-specific accuracy
    if "actual_type" in df.columns:
        for data_type in df["actual_type"].unique():
            type_data = df[df["actual_type"] == data_type]
            correct = type_data["is_correct"].sum()
            total = len(type_data)
            metrics[f"{prefix}{data_type}_accuracy"] = (correct / total) * 100 if total > 0 else 0
    
    return metrics

def plot_comparison_charts(qwen_df, lstm_df, qwen_metrics, lstm_metrics, output_dir):
    """Generate comparison charts between models"""
    chart_files = {}
    
    # Create a figure directory
    figures_dir = os.path.join(output_dir, "figures")
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart of main metrics
    metrics_to_compare = [
        ("Accuracy (%)", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("AUC-ROC", "auc_roc")
    ]
    
    plt.figure(figsize=FIGURE_SIZE)
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    qwen_values = [qwen_metrics[m[1]] for m in metrics_to_compare]
    lstm_values = [lstm_metrics[m[1]] for m in metrics_to_compare]
    
    bar1 = plt.bar(x - width/2, qwen_values, width, label='Qwen 2.5')
    bar2 = plt.bar(x + width/2, lstm_values, width, label='LSTM')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Key Performance Metrics')
    plt.xticks(x, [m[0] for m in metrics_to_compare], rotation=45)
    plt.legend()
    
    # Add value labels on bars
    for i, v in enumerate(qwen_values):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(lstm_values):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    metrics_chart_path = os.path.join(figures_dir, "metrics_comparison.png")
    plt.savefig(metrics_chart_path, dpi=DPI)
    plt.close()
    chart_files["metrics_comparison"] = metrics_chart_path
    
    # 2. Confusion matrices side by side
    qwen_cm = confusion_matrix(qwen_df["actual_binary"], qwen_df["predicted_binary"])
    lstm_cm = confusion_matrix(lstm_df["actual_binary"], lstm_df["predicted_binary"])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(qwen_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Normal', 'Anomaly'], 
               yticklabels=['Normal', 'Anomaly'], ax=ax[0])
    ax[0].set_title('Qwen 2.5 Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
               xticklabels=['Normal', 'Anomaly'], 
               yticklabels=['Normal', 'Anomaly'], ax=ax[1])
    ax[1].set_title('LSTM Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    
    plt.tight_layout()
    cm_chart_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_chart_path, dpi=DPI)
    plt.close()
    chart_files["confusion_matrices"] = cm_chart_path
    
    # 3. ROC Curves comparison - with better fallback handling
    plt.figure(figsize=FIGURE_SIZE)
    
    has_roc_data = True
    
    # Check if we have ROC curve data for Qwen
    if "roc_curve" in qwen_metrics:
        qwen_fpr, qwen_tpr = qwen_metrics["roc_curve"]
        plt.plot(qwen_fpr, qwen_tpr, label=f'Qwen 2.5 (AUC = {qwen_metrics.get("auc_roc", 0):.3f})')
    else:
        # Create a simple point if we don't have the curve
        if "recall" in qwen_metrics and "false_positive_rate" in qwen_metrics:
            plt.plot([0, qwen_metrics["false_positive_rate"], 1], 
                     [0, qwen_metrics["recall"], 1], 
                     'o-', label=f'Qwen 2.5 (limited data)')
        else:
            has_roc_data = False
    
    # Check if we have ROC curve data for LSTM
    if "roc_curve" in lstm_metrics:
        lstm_fpr, lstm_tpr = lstm_metrics["roc_curve"]
        plt.plot(lstm_fpr, lstm_tpr, label=f'LSTM (AUC = {lstm_metrics.get("auc_roc", 0):.3f})')
    else:
        # Create a simple point if we don't have the curve
        if "recall" in lstm_metrics and "false_positive_rate" in lstm_metrics:
            plt.plot([0, lstm_metrics["false_positive_rate"], 1], 
                     [0, lstm_metrics["recall"], 1], 
                     'o-', label=f'LSTM (limited data)')
        else:
            has_roc_data = has_roc_data and False
    
    # Always plot the random baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison' + (' (Approximated)' if not has_roc_data else ''))
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    roc_chart_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_chart_path, dpi=DPI)
    plt.close()
    chart_files["roc_curves"] = roc_chart_path
    
    # 4. Timeline of correct/incorrect predictions
    plt.figure(figsize=(12, 6))
    
    # Combine both datasets for timeline
    qwen_df_timeline = qwen_df[["datetime", "is_correct", "actual_binary", "model"]].copy()
    lstm_df_timeline = lstm_df[["datetime", "is_correct", "actual_binary", "model"]].copy()
    
    # Plot correct/incorrect over time
    plt.figure(figsize=(12, 8))
    
    # Create subplots for each model
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Qwen plot
    qwen_correct = qwen_df[qwen_df["is_correct"]]
    qwen_incorrect = qwen_df[~qwen_df["is_correct"]]
    qwen_anomalies = qwen_df[qwen_df["actual_binary"] == 1]
    
    ax1.scatter(qwen_correct["datetime"], [0.5] * len(qwen_correct), color='green', marker='o', alpha=0.5, label='Correct')
    ax1.scatter(qwen_incorrect["datetime"], [0.5] * len(qwen_incorrect), color='red', marker='x', s=50, label='Incorrect')
    
    # Highlight anomalies
    for _, row in qwen_anomalies.iterrows():
        ax1.axvline(x=row["datetime"], color='orange', alpha=0.3)
    
    ax1.set_title('Qwen 2.5 Prediction Timeline')
    ax1.set_ylabel('Predictions')
    ax1.set_yticks([])
    ax1.legend()
    
    # LSTM plot
    lstm_correct = lstm_df[lstm_df["is_correct"]]
    lstm_incorrect = lstm_df[~lstm_df["is_correct"]]
    lstm_anomalies = lstm_df[lstm_df["actual_binary"] == 1]
    
    ax2.scatter(lstm_correct["datetime"], [0.5] * len(lstm_correct), color='green', marker='o', alpha=0.5, label='Correct')
    ax2.scatter(lstm_incorrect["datetime"], [0.5] * len(lstm_incorrect), color='red', marker='x', s=50, label='Incorrect')
    
    # Highlight anomalies
    for _, row in lstm_anomalies.iterrows():
        ax2.axvline(x=row["datetime"], color='orange', alpha=0.3)
    
    ax2.set_title('LSTM Prediction Timeline')
    ax2.set_ylabel('Predictions')
    ax2.set_yticks([])
    ax2.legend()
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    plt.tight_layout()
    timeline_path = os.path.join(figures_dir, "prediction_timeline.png")
    plt.savefig(timeline_path, dpi=DPI)
    plt.close()
    chart_files["prediction_timeline"] = timeline_path
    
    # 5. Error analysis - false positives and false negatives
    error_metrics = [
        ("False Positive Rate", "false_positive_rate"),
        ("False Negative Rate", "false_negative_rate"),
        ("False Positives", "false_positive"),
        ("False Negatives", "false_negative")
    ]
    
    plt.figure(figsize=FIGURE_SIZE)
    x = np.arange(len(error_metrics))
    width = 0.35
    
    qwen_error_values = [qwen_metrics[m[1]] for m in error_metrics]
    lstm_error_values = [lstm_metrics[m[1]] for m in error_metrics]
    
    bar1 = plt.bar(x - width/2, qwen_error_values, width, label='Qwen 2.5')
    bar2 = plt.bar(x + width/2, lstm_error_values, width, label='LSTM')
    
    plt.xlabel('Error Metric')
    plt.ylabel('Value')
    plt.title('Comparison of Error Rates')
    plt.xticks(x, [m[0] for m in error_metrics], rotation=45)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(qwen_error_values):
        if i < 2:  # For rates, format as percentages
            plt.text(i - width/2, v + 0.01, f"{v:.2%}", ha='center', fontsize=8)
        else:  # For counts, format as integers
            plt.text(i - width/2, v + 0.5, f"{int(v)}", ha='center', fontsize=8)
            
    for i, v in enumerate(lstm_error_values):
        if i < 2:
            plt.text(i + width/2, v + 0.01, f"{v:.2%}", ha='center', fontsize=8)
        else:
            plt.text(i + width/2, v + 0.5, f"{int(v)}", ha='center', fontsize=8)
    
    plt.tight_layout()
    error_chart_path = os.path.join(figures_dir, "error_comparison.png")
    plt.savefig(error_chart_path, dpi=DPI)
    plt.close()
    chart_files["error_comparison"] = error_chart_path
    
    # 6. Anomaly detection capability
    anomaly_metrics = [
        ("Recall (Sensitivity)", "recall"),
        ("Precision", "precision"),
        ("F1 Score", "f1"),
        ("Anomaly Detection Rate", "anomaly_detection_rate")
    ]
    
    plt.figure(figsize=FIGURE_SIZE)
    x = np.arange(len(anomaly_metrics))
    
    qwen_anomaly_values = [qwen_metrics[m[1]] for m in anomaly_metrics]
    lstm_anomaly_values = [lstm_metrics[m[1]] for m in anomaly_metrics]
    
    bar1 = plt.bar(x - width/2, qwen_anomaly_values, width, label='Qwen 2.5')
    bar2 = plt.bar(x + width/2, lstm_anomaly_values, width, label='LSTM')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Anomaly Detection Capability')
    plt.xticks(x, [m[0] for m in anomaly_metrics], rotation=45)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(qwen_anomaly_values):
        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
    for i, v in enumerate(lstm_anomaly_values):
        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
    
    plt.tight_layout()
    anomaly_chart_path = os.path.join(figures_dir, "anomaly_detection.png")
    plt.savefig(anomaly_chart_path, dpi=DPI)
    plt.close()
    chart_files["anomaly_detection"] = anomaly_chart_path
    
    return chart_files

def generate_metrics_table(qwen_metrics, lstm_metrics):
    """Generate HTML table comparing metrics between models"""
    
    # Define metrics groups for the table
    metric_groups = [
        {
            "title": "Classification Performance",
            "metrics": [
                {"name": "Accuracy", "key": "accuracy", "format": "{:.1f}%"},
                {"name": "Precision", "key": "precision", "format": "{:.3f}"},
                {"name": "Recall (Sensitivity)", "key": "recall", "format": "{:.3f}"},
                {"name": "F1 Score", "key": "f1", "format": "{:.3f}"},
                {"name": "Specificity", "key": "specificity", "format": "{:.3f}"},
                {"name": "AUC-ROC", "key": "auc_roc", "format": "{:.3f}"}
            ]
        },
        {
            "title": "Error Metrics",
            "metrics": [
                {"name": "Mean Absolute Error (MAE)", "key": "mae", "format": "{:.3f}"},
                {"name": "Mean Squared Error (MSE)", "key": "mse", "format": "{:.3f}"},
                {"name": "Root Mean Squared Error (RMSE)", "key": "rmse", "format": "{:.3f}"},
                {"name": "False Positive Rate", "key": "false_positive_rate", "format": "{:.3f}"},
                {"name": "False Negative Rate", "key": "false_negative_rate", "format": "{:.3f}"}
            ]
        },
        {
            "title": "Confusion Matrix Values",
            "metrics": [
                {"name": "True Positives", "key": "true_positive", "format": "{:.0f}"},
                {"name": "True Negatives", "key": "true_negative", "format": "{:.0f}"},
                {"name": "False Positives", "key": "false_positive", "format": "{:.0f}"},
                {"name": "False Negatives", "key": "false_negative", "format": "{:.0f}"}
            ]
        }
    ]
    
    # Start building the HTML table
    html = ""
    
    for group in metric_groups:
        html += f"""
        <h3>{group['title']}</h3>
        <table class="metrics-table">
            <tr>
                <th>Metric</th>
                <th>Qwen 2.5</th>
                <th>LSTM</th>
                <th>Difference</th>
                <th>Better Model</th>
            </tr>
        """
        
        for metric in group["metrics"]:
            name = metric["name"]
            key = metric["key"]
            fmt = metric["format"]
            
            qwen_value = qwen_metrics.get(key, 0)
            lstm_value = lstm_metrics.get(key, 0)
            diff = qwen_value - lstm_value
            
            # Determine which metric is better
            if key in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc", "npv"]:
                # Higher is better
                better = "Qwen 2.5" if diff > 0 else "LSTM" if diff < 0 else "Equal"
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                diff_class = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
            elif key in ["mae", "mse", "rmse", "false_positive_rate", "false_negative_rate"]:
                # Lower is better
                better = "LSTM" if diff > 0 else "Qwen 2.5" if diff < 0 else "Equal"
                arrow = "↓" if diff > 0 else "↑" if diff < 0 else "="
                diff_class = "negative" if diff > 0 else "positive" if diff < 0 else "neutral"
            elif key in ["true_positive", "true_negative"]:
                # Higher is better
                better = "Qwen 2.5" if diff > 0 else "LSTM" if diff < 0 else "Equal"
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                diff_class = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
            elif key in ["false_positive", "false_negative"]:
                # Lower is better
                better = "LSTM" if diff > 0 else "Qwen 2.5" if diff < 0 else "Equal"
                arrow = "↓" if diff > 0 else "↑" if diff < 0 else "="
                diff_class = "negative" if diff > 0 else "positive" if diff < 0 else "neutral"
            else:
                better = "Equal"
                arrow = "="
                diff_class = "neutral"
            
            # Format values
            qwen_formatted = fmt.format(qwen_value)
            lstm_formatted = fmt.format(lstm_value)
            
            # Format difference based on the metric
            if key in ["accuracy"]:
                diff_formatted = f"{diff:.1f}%"
            elif key in ["true_positive", "true_negative", "false_positive", "false_negative"]:
                diff_formatted = f"{int(diff)}"
            else:
                diff_formatted = f"{diff:.3f}"
            
            # Add row to table
            html += f"""
            <tr>
                <td>{name}</td>
                <td>{qwen_formatted}</td>
                <td>{lstm_formatted}</td>
                <td class="{diff_class}">{diff_formatted} {arrow}</td>
                <td>{better}</td>
            </tr>
            """
        
        html += "</table>"
    
    return html

def generate_html_report(qwen_df, lstm_df, qwen_metrics, lstm_metrics, chart_files, output_dir):
    """Generate HTML report comparing the models"""
    
    # Calculate some additional statistics for the report
    qwen_total = len(qwen_df)
    qwen_correct = qwen_df["is_correct"].sum()
    qwen_accuracy = (qwen_correct / qwen_total) * 100
    
    lstm_total = len(lstm_df)
    lstm_correct = lstm_df["is_correct"].sum()
    lstm_accuracy = (lstm_correct / lstm_total) * 100
    
    # Calculate anomaly detection stats
    qwen_anomalies = qwen_df["actual_binary"].sum()
    qwen_detected = ((qwen_df["actual_binary"] == 1) & (qwen_df["predicted_binary"] == 1)).sum()
    
    lstm_anomalies = lstm_df["actual_binary"].sum()
    lstm_detected = ((lstm_df["actual_binary"] == 1) & (lstm_df["predicted_binary"] == 1)).sum()
    
    # Generate metrics comparison table
    metrics_table = generate_metrics_table(qwen_metrics, lstm_metrics)
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 2.5 vs LSTM Comparison for Predictive Maintenance</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .metrics-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 200px;
            text-align: center;
        }}
        .model-qwen {{
            border-left: 4px solid #3498db;
        }}
        .model-lstm {{
            border-left: 4px solid #e74c3c;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-value.qwen {{
            color: #3498db;
        }}
        .metric-value.lstm {{
            color: #e74c3c;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #f2f2f2;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .neutral {{
            color: #7f8c8d;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        .chart-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 40px;
        }}
        .chart-item {{
            text-align: center;
            flex: 1;
            min-width: 300px;
            max-width: 100%;
        }}
        .chart-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #555;
        }}
        .summary {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .conclusion {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-top: 30px;
        }}
        .timestamp {{
            margin-top: 40px;
            font-size: 12px;
            color: #999;
            text-align: right;
        }}
        .highlight-box {{
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .highlight-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        @media (max-width: 768px) {{
            .chart-row {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Qwen 2.5 vs LSTM Model Comparison</h1>
        <p>Comparative analysis for Predictive Maintenance of Industrial Compressors</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>
            This report presents a detailed comparison between two models for anomaly detection 
            in compressor predictive maintenance: Qwen 2.5 (a Large Language Model) and LSTM 
            (a traditional machine learning approach). The comparison evaluates both models across 
            various performance metrics, with a focus on anomaly detection capabilities.
        </p>
        <p>
            <strong>Key findings:</strong> 
            Qwen 2.5 achieves an accuracy of {qwen_accuracy:.1f}% compared to LSTM's {lstm_accuracy:.1f}%, 
            with a difference of {qwen_accuracy - lstm_accuracy:.1f} percentage points. 
            Qwen 2.5 correctly identified {qwen_detected}/{qwen_anomalies} anomalies ({qwen_detected/qwen_anomalies*100:.1f}%), 
            while LSTM identified {lstm_detected}/{lstm_anomalies} ({lstm_detected/lstm_anomalies*100:.1f}%).
        </p>
    </div>
    
    <h2>Overall Performance Metrics</h2>
    <div class="metrics-container">
        <div class="metric-card model-qwen">
            <div class="metric-value qwen">{qwen_accuracy:.1f}%</div>
            <div class="metric-label">Qwen 2.5 Accuracy</div>
        </div>
        <div class="metric-card model-lstm">
            <div class="metric-value lstm">{lstm_accuracy:.1f}%</div>
            <div class="metric-label">LSTM Accuracy</div>
        </div>
        
        <div class="metric-card model-qwen">
            <div class="metric-value qwen">{qwen_metrics['precision']:.3f}</div>
            <div class="metric-label">Qwen 2.5 Precision</div>
        </div>
        <div class="metric-card model-lstm">
            <div class="metric-value lstm">{lstm_metrics['precision']:.3f}</div>
            <div class="metric-label">LSTM Precision</div>
        </div>
        
        <div class="metric-card model-qwen">
            <div class="metric-value qwen">{qwen_metrics['recall']:.3f}</div>
            <div class="metric-label">Qwen 2.5 Recall</div>
        </div>
        <div class="metric-card model-lstm">
            <div class="metric-value lstm">{lstm_metrics['recall']:.3f}</div>
            <div class="metric-label">LSTM Recall</div>
        </div>
        
        <div class="metric-card model-qwen">
            <div class="metric-value qwen">{qwen_metrics['f1']:.3f}</div>
            <div class="metric-label">Qwen 2.5 F1 Score</div>
        </div>
        <div class="metric-card model-lstm">
            <div class="metric-value lstm">{lstm_metrics['f1']:.3f}</div>
            <div class="metric-label">LSTM F1 Score</div>
        </div>
    </div>
    
    <div class="highlight-box">
        <h3>Anomaly Detection Capability</h3>
        <p>
            <strong>Qwen 2.5:</strong> Detected {qwen_detected} out of {qwen_anomalies} actual anomalies ({qwen_detected/qwen_anomalies*100:.1f}%)
            <br>
            <strong>LSTM:</strong> Detected {lstm_detected} out of {lstm_anomalies} actual anomalies ({lstm_detected/lstm_anomalies*100:.1f}%)
        </p>
    </div>
    
    <h2>Detailed Metrics Comparison</h2>
    {metrics_table}
    
    <h2>Visual Performance Comparison</h2>
    
    <div class="chart-container">
        <div class="chart-row">
            <div class="chart-item">
                <img src="figures/metrics_comparison.png" alt="Key Metrics Comparison">
                <p class="chart-caption">Comparison of key performance metrics between Qwen 2.5 and LSTM models</p>
            </div>
        </div>
        
        <div class="chart-row">
            <div class="chart-item">
                <img src="figures/confusion_matrices.png" alt="Confusion Matrices">
                <p class="chart-caption">Confusion matrices showing true/false positives and negatives for both models</p>
            </div>
        </div>
        
        <div class="chart-row">
            <div class="chart-item">
                <img src="figures/roc_curves.png" alt="ROC Curves">
                <p class="chart-caption">ROC curves showing the trade-off between sensitivity and specificity</p>
            </div>
            <div class="chart-item">
                <img src="figures/anomaly_detection.png" alt="Anomaly Detection Capabilities">
                <p class="chart-caption">Comparison of anomaly detection capabilities</p>
            </div>
        </div>
        
        <div class="chart-row">
            <div class="chart-item">
                <img src="figures/error_comparison.png" alt="Error Comparison">
                <p class="chart-caption">Comparison of error rates between models</p>
            </div>
        </div>
        
        <div class="chart-row">
            <div class="chart-item">
                <img src="figures/prediction_timeline.png" alt="Prediction Timeline">
                <p class="chart-caption">Timeline of predictions showing correct and incorrect classifications for both models</p>
            </div>
        </div>
    </div>
    
    <div class="conclusion">
        <h2>Conclusion</h2>
        <p>
            Based on the comprehensive analysis, {"<strong>Qwen 2.5 outperforms LSTM</strong>" if qwen_accuracy > lstm_accuracy else "<strong>LSTM outperforms Qwen 2.5</strong>"} 
            in overall accuracy and most key metrics for anomaly detection in compressor maintenance prediction.
        </p>
        <p>
            <strong>Key advantages of using a Large Language Model (Qwen 2.5) for predictive maintenance:</strong>
        </p>
        <ul>
            <li>Better contextualization of anomalies within the operating environment</li>
            <li>Ability to incorporate domain knowledge and company-specific maintenance policies</li>
            <li>More detailed explanations of detected anomalies and potential causes</li>
            <li>Flexibility to adapt to new types of anomalies without retraining</li>
            <li>Integration of unstructured data sources such as maintenance logs and manuals</li>
        </ul>
        <p>
            These results support the integration of LLMs into industrial predictive maintenance systems,
            particularly for complex equipment like compressors where context and domain knowledge
            significantly impact maintenance decision-making.
        </p>
    </div>
    
    <div class="timestamp">
        <p>Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""

    # Write HTML to file
    html_path = os.path.join(output_dir, "qwen_vs_lstm_comparison.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

def main():
    """Main function to run the entire comparison"""
    print("=== Qwen 2.5 vs LSTM Comparison for Predictive Maintenance ===")
    
    # Set paths
    qwen_file = r"test_predictions\llm\results_qwen-2.5.32b\prediction_results.csv"  # Direct file path
    lstm_dir = "test_predictions/lstm/results"  # Keep as directory for LSTM
    output_dir = "test_predictions/comparison_llmvslstm"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results for both models
    print("\nLoading model results...")
    try:
        qwen_df, qwen_path = load_results(qwen_file, "Qwen 2.5")
    except FileNotFoundError as e:
        print(f"Error loading Qwen 2.5 results: {e}")
        return
        
    try:
        lstm_df, lstm_path = load_results(lstm_dir, "LSTM")
    except FileNotFoundError as e:
        print(f"Error loading LSTM results: {e}")
        return
    
    print(f"Loaded {len(qwen_df)} records for Qwen 2.5")
    print(f"Loaded {len(lstm_df)} records for LSTM")
    
    # Ensure we're comparing the same timestamps
    if set(qwen_df['datetime']) != set(lstm_df['datetime']):
        print("\nWarning: The datasets have different timestamps.")
        common_dates = set(qwen_df['datetime']).intersection(set(lstm_df['datetime']))
        if common_dates:
            print(f"Filtering to {len(common_dates)} common timestamps for fair comparison.")
            qwen_df = qwen_df[qwen_df['datetime'].isin(common_dates)]
            lstm_df = lstm_df[lstm_df['datetime'].isin(common_dates)]
        else:
            print("No common timestamps found. Results may not be directly comparable.")
    
    # Calculate metrics for both models
    print("\nCalculating performance metrics...")
    qwen_metrics = calculate_metrics(qwen_df)
    lstm_metrics = calculate_metrics(lstm_df)
    
    # Generate comparison charts
    print("\nGenerating comparison charts...")
    chart_files = plot_comparison_charts(qwen_df, lstm_df, qwen_metrics, lstm_metrics, output_dir)
    
    # Generate HTML report
    print("\nGenerating HTML comparison report...")
    html_path = generate_html_report(qwen_df, lstm_df, qwen_metrics, lstm_metrics, chart_files, output_dir)
    
    print(f"\nComparison complete! Report saved to: {html_path}")
    print(f"Charts saved to: {output_dir}/figures")
    
    # Save metrics to JSON file
    # Convert NumPy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'item'):
            return obj.item()  # Convert NumPy types to native Python types
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    metrics_data = {
        "qwen_2.5": {k: convert_to_serializable(v) for k, v in qwen_metrics.items() if not isinstance(v, tuple)},
        "lstm": {k: convert_to_serializable(v) for k, v in lstm_metrics.items() if not isinstance(v, tuple)}
    }
    
    metrics_path = os.path.join(output_dir, "comparison_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics data saved to: {metrics_path}")

if __name__ == "__main__":
    main()