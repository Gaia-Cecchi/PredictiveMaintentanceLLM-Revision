import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

def load_prediction_results(results_dir):
    """
    Load prediction results from CSV file and metrics from JSON
    """
    # Check for extended validation results first
    extended_csv = os.path.join(results_dir, 'prediction_results.csv')
    if os.path.exists(extended_csv):
        df = pd.read_csv(extended_csv)
        print(f"Loaded extended validation results: {len(df)} predictions")
        analysis_type = 'extended'
    else:
        # Try quick analysis results
        quick_csv = os.path.join(results_dir, 'quick_analysis_results.csv')
        if os.path.exists(quick_csv):
            df = pd.read_csv(quick_csv)
            print(f"Loaded quick analysis results: {len(df)} predictions")
            analysis_type = 'quick'
        else:
            raise FileNotFoundError("No prediction results found. Run the analysis first.")
    
    # Convert datetime to proper format
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Standardize the classification labels for binary prompt
    if 'actual_classification' in df.columns:
        # Change 'NORMAL' to 'NORMAL VALUE'
        df['actual_classification'] = df['actual_classification'].replace('NORMAL', 'NORMAL VALUE')
        # Also change 'FALSE POSITIVE' to 'NORMAL VALUE' for binary classification
        df['actual_classification'] = df['actual_classification'].replace('FALSE POSITIVE', 'NORMAL VALUE')
    
    # Standardize event types
    if 'actual_type' in df.columns:
        # Map both 'normal' and 'false_positive' to 'normal_value'
        df['actual_type'] = df['actual_type'].replace(['normal', 'false_positive'], 'normal_value')
    
    # Update the is_correct field for binary classification
    if 'predicted_classification' in df.columns and 'is_correct' in df.columns:
        # For binary classification: ANOMALY vs NORMAL VALUE
        df['is_correct'] = (df['actual_classification'] == df['predicted_classification'])
        
        # Fix any NORMAL VALUE vs FALSE POSITIVE discrepancies
        normal_mask = df['actual_classification'] == 'NORMAL VALUE'
        fp_pred_mask = df['predicted_classification'] == 'FALSE POSITIVE'
        # If actual is NORMAL VALUE and predicted is FALSE POSITIVE, mark as correct
        df.loc[normal_mask & fp_pred_mask, 'is_correct'] = True
        
        # Also handle the reverse - if predicted is NORMAL VALUE and actual is FALSE POSITIVE
        pred_normal_mask = df['predicted_classification'] == 'NORMAL VALUE'
        actual_fp_mask = df['actual_type'] == 'false_positive'
        df.loc[pred_normal_mask & actual_fp_mask, 'is_correct'] = True
    
    # Load metrics if available
    metrics_file = os.path.join(results_dir, 'prediction_metrics.json')
    metrics = None
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
            # Standardize labels for binary classification
            if 'classes' in metrics:
                # Combine all non-ANOMALY classes into NORMAL VALUE
                classes_to_combine = ['NORMAL', 'FALSE POSITIVE', 'NOT ANOMALY', 'NORMAL VALUE']
                
                # Check which classes need to be combined
                present_classes = [c for c in classes_to_combine if c in metrics['classes']]
                
                if len(present_classes) > 1 or (len(present_classes) == 1 and present_classes[0] != 'NORMAL VALUE'):
                    # Calculate combined metrics
                    combined_support = 0
                    weighted_precision = 0
                    weighted_recall = 0
                    weighted_f1 = 0
                    
                    for cls in present_classes:
                        if cls in metrics['classes']:
                            cls_support = metrics['classes'][cls]['support']
                            combined_support += cls_support
                            weighted_precision += metrics['classes'][cls]['precision'] * cls_support
                            weighted_recall += metrics['classes'][cls]['recall'] * cls_support
                            weighted_f1 += metrics['classes'][cls]['f1-score'] * cls_support
                    
                    # Create the NORMAL VALUE metrics
                    if combined_support > 0:
                        metrics['classes']['NORMAL VALUE'] = {
                            'precision': weighted_precision / combined_support,
                            'recall': weighted_recall / combined_support,
                            'f1-score': weighted_f1 / combined_support,
                            'support': combined_support
                        }
                        
                        # Remove the original classes
                        for cls in present_classes:
                            if cls != 'NORMAL VALUE' and cls in metrics['classes']:
                                metrics['classes'].pop(cls)
                    
                    # Update labels in confusion matrix to use binary classification
                    if 'confusion_matrix' in metrics and 'labels' in metrics['confusion_matrix']:
                        metrics['confusion_matrix']['labels'] = [
                            'NORMAL VALUE' if label in classes_to_combine else label 
                            for label in metrics['confusion_matrix']['labels']
                        ]
    
    return df, metrics, analysis_type

def calculate_evaluation_metrics(df):
    """
    Calculate comprehensive evaluation metrics from the prediction results
    """
    # Map classifications to binary (ANOMALY = 1, NORMAL VALUE = 0)
    y_true = (df['actual_classification'] == 'ANOMALY').astype(int)
    y_pred = (df['predicted_classification'] == 'ANOMALY').astype(int)
    
    # Calculate basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_anomaly = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_anomaly = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_anomaly = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    precision_fp = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_fp = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fp = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Additional metrics
    # ROC AUC (only if we have both classes)
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            pass
    
    # Error metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Count anomalies and normal values
    anomaly_count = sum(y_true)
    normal_count = len(y_true) - anomaly_count
    
    evaluation = {
        'accuracy': accuracy,
        'precision_anomaly': precision_anomaly,
        'recall_anomaly': recall_anomaly,
        'f1_anomaly': f1_anomaly,
        'precision_normal': precision_fp,
        'recall_normal': recall_fp,
        'f1_normal': f1_fp,
        'roc_auc': roc_auc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'confusion_matrix': cm.tolist(),
        'class_counts': {
            'anomaly': int(anomaly_count),
            'normal': int(normal_count),
            'total': len(y_true)
        }
    }
    
    return evaluation

# ===== Interactive Plotly Visualizations =====

def create_interactive_confusion_matrix(df):
    """
    Create an interactive confusion matrix plot with Plotly
    """
    # Create confusion matrix
    actual = df['actual_classification'].values
    predicted = df['predicted_classification'].values
    
    # Standardize to binary classification
    actual = np.array(['ANOMALY' if a == 'ANOMALY' else 'NORMAL VALUE' for a in actual])
    predicted = np.array(['ANOMALY' if p == 'ANOMALY' else 'NORMAL VALUE' for p in predicted])
    
    # Get unique classes in sorted order
    classes = sorted(list(set(actual) | set(predicted)))
    
    # Initialize the confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Fill the confusion matrix
    for a, p in zip(actual, predicted):
        i = classes.index(a)
        j = classes.index(p)
        cm[i, j] += 1
    
    # Create a heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=classes,
        y=classes,
        text_auto=True,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title="Confusion Matrix (Binary Classification: ANOMALY vs NORMAL VALUE)",
        xaxis_title="Predicted Classification",
        yaxis_title="Actual Classification",
        width=700,
        height=600
    )
    
    return fig

def create_accuracy_by_type(df):
    """
    Create bar chart showing accuracy by event type
    """
    # For binary classification, we simplify to just 'anomaly' and 'normal_value'
    # First, standardize types
    if 'actual_type' not in df.columns:
        df['actual_type'] = df['actual_classification'].apply(
            lambda x: 'anomaly' if x == 'ANOMALY' else 'normal_value'
        )
    
    # Calculate accuracy for each type
    type_accuracy = df.groupby('actual_type')['is_correct'].mean().reset_index()
    type_accuracy['accuracy'] = type_accuracy['is_correct'] * 100
    
    # Count occurrences
    type_counts = df.groupby('actual_type').size().reset_index(name='count')
    
    # Merge accuracy and counts
    type_stats = pd.merge(type_accuracy, type_counts, on='actual_type')
    
    # Create the figure
    fig = px.bar(
        type_stats,
        x='actual_type',
        y='accuracy',
        text=type_stats['count'].apply(lambda x: f"n={x}"),
        color='actual_type',
        labels={'actual_type': 'Event Type', 'accuracy': 'Accuracy (%)'},
        title='Prediction Accuracy by Event Type (Binary Classification)',
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Event Type",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100]
    )
    
    return fig

def create_confidence_vs_accuracy(df):
    """
    Create plot showing relationship between confidence level and accuracy
    """
    # Map confidence levels to numeric values
    confidence_map = {'high': 3, 'medium': 2, 'low': 1}
    df['confidence_num'] = df['confidence'].apply(lambda x: confidence_map.get(x.lower(), 0) if isinstance(x, str) else 0)
    
    # Calculate accuracy by confidence level
    conf_accuracy = df.groupby('confidence')['is_correct'].agg(
        accuracy='mean',
        count='size'
    ).reset_index()
    
    # Create the figure
    fig = px.bar(
        conf_accuracy,
        x='confidence',
        y='accuracy',
        text=conf_accuracy['count'].apply(lambda x: f"n={x}"),
        color='confidence',
        labels={'confidence': 'Confidence Level', 'accuracy': 'Accuracy'},
        title='Accuracy by Confidence Level',
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Confidence Level",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1]
    )
    
    # Add text labels showing accuracy percentage
    for i, row in enumerate(conf_accuracy.itertuples()):
        fig.add_annotation(
            x=i,
            y=row.accuracy + 0.05,
            text=f"{row.accuracy:.1%}",
            showarrow=False
        )
    
    return fig

def create_time_series_accuracy(df):
    """
    Create time series plot showing prediction performance over time
    """
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Calculate cumulative accuracy
    df['cumulative_correct'] = df['is_correct'].cumsum()
    df['cumulative_total'] = range(1, len(df) + 1)
    df['cumulative_accuracy'] = df['cumulative_correct'] / df['cumulative_total']
    
    # Create the figure
    fig = go.Figure()
    
    # Add cumulative accuracy line
    fig.add_trace(go.Scatter(
        x=df['datetime'],
        y=df['cumulative_accuracy'],
        mode='lines+markers',
        name='Cumulative Accuracy',
        line=dict(color='blue', width=3)
    ))
    
    # Add individual predictions as markers
    for actual_type in df['actual_type'].unique():
        subset = df[df['actual_type'] == actual_type]
        fig.add_trace(go.Scatter(
            x=subset['datetime'],
            y=[1 if correct else 0 for correct in subset['is_correct']],
            mode='markers',
            marker=dict(
                size=12,
                symbol='circle',
                color=['green' if correct else 'red' for correct in subset['is_correct']]
            ),
            name=f'{actual_type} predictions',
            text=[f"Actual: {a}<br>Predicted: {p}" for a, p in zip(subset['actual_classification'], subset['predicted_classification'])]
        ))
    
    fig.update_layout(
        title="Prediction Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis_range=[-0.1, 1.1],
        legend_title="Legend"
    )
    
    return fig

def create_parameter_threshold_analysis(df):
    """
    Create visualization showing predictions in relation to critical thresholds
    """
    # Create subplots for different parameters
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            "Temperature vs. Classification (115°C Threshold)",
            "Vibration vs. Classification (4.0 mm/s Threshold)",
            "Pressure vs. Classification (5.5 bar Threshold)",
            "Current vs. Classification"
        )
    )
    
    # Define marker colors based on predicted classification - use binary colors
    marker_colors = df['predicted_classification'].apply(lambda x: 'red' if x == 'ANOMALY' else 'green')
    
    # Temperature subplot
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['temperature'],
            mode='markers',
            marker=dict(
                size=10,
                color=marker_colors,
                symbol=df['is_correct'].apply(lambda x: 'circle' if x else 'x')
            ),
            name='Temperature',
            text=[f"Actual: {row['actual_classification']}<br>Predicted: {row['predicted_classification']}<br>Temp: {row['temperature']}" 
                  for _, row in df.iterrows()]
        ),
        row=1, col=1
    )
    
    # Add temperature threshold line
    fig.add_hline(y=115, row=1, col=1, line_dash="dash", line_color="red",
                  annotation_text="Critical threshold (115°C)")
    
    # Vibration subplot
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['vibration'],
            mode='markers',
            marker=dict(
                size=10,
                color=marker_colors,
                symbol=df['is_correct'].apply(lambda x: 'circle' if x else 'x')
            ),
            name='Vibration',
            text=[f"Actual: {row['actual_classification']}<br>Predicted: {row['predicted_classification']}<br>Vibration: {row['vibration']}" 
                  for _, row in df.iterrows()]
        ),
        row=1, col=2
    )
    
    # Add vibration threshold line
    fig.add_hline(y=4.0, row=1, col=2, line_dash="dash", line_color="red",
                  annotation_text="Critical threshold (4.0 mm/s)")
    
    # Pressure subplot
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['pressure'],
            mode='markers',
            marker=dict(
                size=10,
                color=marker_colors,
                symbol=df['is_correct'].apply(lambda x: 'circle' if x else 'x')
            ),
            name='Pressure',
            text=[f"Actual: {row['actual_classification']}<br>Predicted: {row['predicted_classification']}<br>Pressure: {row['pressure']}" 
                  for _, row in df.iterrows()]
        ),
        row=2, col=1
    )
    
    # Add pressure threshold line
    fig.add_hline(y=5.5, row=2, col=1, line_dash="dash", line_color="red",
                  annotation_text="Critical threshold (5.5 bar)")
    
    # Current subplot
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['current'],
            mode='markers',
            marker=dict(
                size=10,
                color=marker_colors,
                symbol=df['is_correct'].apply(lambda x: 'circle' if x else 'x')
            ),
            name='Current',
            text=[f"Actual: {row['actual_classification']}<br>Predicted: {row['predicted_classification']}<br>Current: {row['current']}" 
                  for _, row in df.iterrows()]
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Parameter Values in Relation to Critical Thresholds",
        height=800,
        showlegend=False
    )
    
    # Add legend for marker symbols and colors - adjust for binary classification
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='green', symbol='circle'),
            name='Correct NORMAL VALUE prediction'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name='Correct ANOMALY prediction'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='green', symbol='x'),
            name='Incorrect NORMAL VALUE prediction'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Incorrect ANOMALY prediction'
        )
    )
    
    fig.update_layout(showlegend=True)
    
    return fig

# ===== Static PNG Visualizations =====

def create_static_confusion_matrix(evaluation, output_dir):
    """
    Create a detailed confusion matrix visualization (static PNG)
    """
    cm = np.array(evaluation['confusion_matrix'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["NORMAL VALUE", "ANOMALY"],
               yticklabels=["NORMAL VALUE", "ANOMALY"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Accuracy: {evaluation["accuracy"]:.2%}')
    
    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    
    return cm_file

def create_static_classification_plots(df, evaluation, output_dir):
    """
    Create static PNG visualizations for classification results
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prediction distribution plot
    cm = np.array(evaluation['confusion_matrix'])
    
    # Calculate percentages
    total_actual_normal = cm[0].sum()
    total_actual_anomaly = cm[1].sum()
    
    normal_correct_pct = 100 * cm[0, 0] / total_actual_normal if total_actual_normal > 0 else 0
    normal_wrong_pct = 100 * cm[0, 1] / total_actual_normal if total_actual_normal > 0 else 0
    anomaly_correct_pct = 100 * cm[1, 1] / total_actual_anomaly if total_actual_anomaly > 0 else 0
    anomaly_wrong_pct = 100 * cm[1, 0] / total_actual_anomaly if total_actual_anomaly > 0 else 0
    
    # Create a DataFrame for the prediction distribution plot
    data = {
        'ANOMALY': [anomaly_correct_pct/100, anomaly_wrong_pct/100],
        'NORMAL VALUE': [normal_wrong_pct/100, normal_correct_pct/100]
    }
    
    ct_df = pd.DataFrame(data, index=['ANOMALY', 'NORMAL VALUE'])
    
    # Plot as a grouped bar chart
    plt.figure(figsize=(10, 6))
    ct_df.plot(kind='bar', ax=plt.gca())
    plt.title('Predicted vs Actual Types')
    plt.ylabel('Percentage')
    plt.xlabel('Actual Type')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Predicted Type')
    
    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Add value labels
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', padding=3, label_type='edge')
    
    # Save the distribution plot
    dist_file = os.path.join(output_dir, "prediction_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_file)
    plt.close()
    
    # 2. Classification pie chart
    total_predicted_anomaly = cm[0, 1] + cm[1, 1]
    total_predicted_normal = cm[0, 0] + cm[1, 0]
    
    plt.figure(figsize=(8, 8))
    final_counts = [total_predicted_anomaly, total_predicted_normal]
    labels = [f"ANOMALY ({total_predicted_anomaly})", f"NORMAL VALUE ({total_predicted_normal})"]
    
    # Create a dynamic explode based on number of categories
    explode = [0.05, 0.05]
    
    # Set a list of colors
    colors = ['#e74c3c', '#3498db']
    
    plt.pie(final_counts, labels=labels, autopct='%1.1f%%', 
            colors=colors, explode=explode,
            shadow=True, startangle=90)
    
    plt.title('Distribution of Predicted Classifications')
    plt.axis('equal')
    
    pie_file = os.path.join(output_dir, "classification_pie.png")
    plt.tight_layout()
    plt.savefig(pie_file)
    plt.close()
    
    # 3. Confidence distribution by classification
    if 'confidence' in df.columns:
        plt.figure(figsize=(10, 6))
        confidence_order = ['high', 'medium', 'low']
        
        # Create a pivot table of confidence counts
        conf_data = pd.crosstab(
            df['confidence'], 
            df['predicted_classification'],
            normalize='columns'
        ).reindex(confidence_order)
        
        # Plot stacked bar chart
        conf_data.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
        plt.title('Confidence Distribution by Prediction Type')
        plt.xlabel('Confidence Level')
        plt.ylabel('Proportion')
        plt.legend(title='Prediction')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add total count annotations
        for i, level in enumerate(confidence_order):
            if level in df['confidence'].values:
                count = len(df[df['confidence'] == level])
                plt.annotate(f'n={count}', (i, 0.05), ha='center', fontweight='bold')
        
        conf_file = os.path.join(output_dir, "confidence_distribution.png")
        plt.tight_layout()
        plt.savefig(conf_file)
        plt.close()
    else:
        conf_file = None
    
    # 4. Accuracy by event type
    type_accuracy = df.groupby('actual_type')['is_correct'].mean().reset_index()
    type_accuracy['accuracy'] = type_accuracy['is_correct'] * 100
    
    # Count occurrences
    type_counts = df.groupby('actual_type').size().reset_index(name='count')
    
    # Merge accuracy and counts
    type_stats = pd.merge(type_accuracy, type_counts, on='actual_type')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(type_stats['actual_type'], type_stats['accuracy'], color=['#3498db', '#e74c3c'])
    
    plt.title('Prediction Accuracy by Event Type')
    plt.xlabel('Event Type')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels
    for i, bar in enumerate(bars):
        plt.annotate(f"n={type_stats['count'].iloc[i]}", 
                    xy=(bar.get_x() + bar.get_width()/2, 5),
                    ha='center', va='bottom',
                    fontweight='bold')
        
    # Add percentage labels
    for i, bar in enumerate(bars):
        plt.annotate(f"{type_stats['accuracy'].iloc[i]:.1f}%", 
                    xy=(bar.get_x() + bar.get_width()/2, type_stats['accuracy'].iloc[i] + 2),
                    ha='center', va='bottom')
    
    type_file = os.path.join(output_dir, "accuracy_by_type.png")
    plt.tight_layout()
    plt.savefig(type_file)
    plt.close()
    
    return {
        "distribution_file": dist_file,
        "pie_file": pie_file,
        "confidence_file": conf_file,
        "type_accuracy_file": type_file
    }

def find_interesting_cases(df, limit=5):
    """
    Find interesting cases for detailed examination
    """
    interesting_cases = []
    
    # 1. Find disagreements (incorrect predictions)
    disagreements = df[~df['is_correct']].copy()
    
    if len(disagreements) > 0:
        # Sort by confidence to find high-confidence errors first
        confidence_map = {'high': 3, 'medium': 2, 'low': 1, '': 0}
        if 'confidence' in disagreements.columns:
            disagreements['confidence_num'] = disagreements['confidence'].apply(lambda x: confidence_map.get(x, 0))
            disagreements = disagreements.sort_values('confidence_num', ascending=False)
        
        for i, (_, row) in enumerate(disagreements.iloc[:limit].iterrows()):
            # Extract key values
            case_data = {
                'index': i+1,
                'actual': row['actual_classification'],
                'predicted': row['predicted_classification'],
                'confidence': row.get('confidence', 'unknown'),
                'datetime': row['datetime'] if 'datetime' in row else None,
            }
            
            # Add available sensor readings
            sensor_readings = {}
            for sensor in ['temperature', 'vibration', 'pressure', 'current', 'voltage', 'speed']:
                if sensor in row:
                    sensor_readings[sensor] = row[sensor]
            
            case_data['sensor_readings'] = sensor_readings
            
            # Add recommendation if available
            if 'parsed_response' in row and isinstance(row['parsed_response'], dict):
                case_data['recommendation'] = row['parsed_response'].get('recommendation', '')
                case_data['key_indicators'] = row['parsed_response'].get('key_indicators', '')
            
            interesting_cases.append(case_data)
    
    return interesting_cases

def calculate_execution_stats(df, metrics=None):
    """
    Calculate execution statistics 
    """
    # Use metrics if provided
    if metrics and 'execution_stats' in metrics:
        return metrics['execution_stats']
    
    # Otherwise estimate based on available data
    return {
        'total_requests': len(df),
        'successful_requests': len(df),
        'execution_time_seconds': 0,
        'avg_time_per_prediction': 0,
        'api_success_rate': 100.0
    }

def generate_html_report(df, static_output_dir, interactive_output_dir, model_name=None):
    """
    Generate comprehensive HTML report with results and visualizations
    """
    # Create output directories
    os.makedirs(static_output_dir, exist_ok=True)
    os.makedirs(interactive_output_dir, exist_ok=True)
    
    # Calculate evaluation metrics
    evaluation = calculate_evaluation_metrics(df)
    
    # Create static visualizations
    cm_file = create_static_confusion_matrix(evaluation, static_output_dir)
    visualization_files = create_static_classification_plots(df, evaluation, static_output_dir)

    # Create interactive visualizations
    print("Creating interactive visualizations...")

    # Create and save interactive confusion matrix
    cm_interactive = create_interactive_confusion_matrix(df)
    cm_interactive.write_html(os.path.join(interactive_output_dir, 'confusion_matrix.html'))

    # Create and save accuracy by type chart
    type_fig = create_accuracy_by_type(df)
    type_fig.write_html(os.path.join(interactive_output_dir, 'accuracy_by_type.html'))

    # Create and save confidence vs accuracy chart
    conf_fig = create_confidence_vs_accuracy(df)
    conf_fig.write_html(os.path.join(interactive_output_dir, 'confidence_vs_accuracy.html'))

    # Create and save time series accuracy plot
    time_fig = create_time_series_accuracy(df)
    time_fig.write_html(os.path.join(interactive_output_dir, 'time_series_accuracy.html'))

    # Create and save parameter threshold analysis
    param_fig = create_parameter_threshold_analysis(df)
    param_fig.write_html(os.path.join(interactive_output_dir, 'parameter_thresholds.html'))

    # Find interesting cases
    interesting_cases = find_interesting_cases(df, limit=5)

    # Calculate execution statistics
    execution_stats = calculate_execution_stats(df)

    # Target accuracy threshold
    target_accuracy = 0.95
    accuracy_met = evaluation['accuracy'] >= target_accuracy

    # Create additional metrics HTML
    additional_metrics_html = ""
    if evaluation['roc_auc'] is not None:
        additional_metrics_html = f"""
        <h2>Advanced Performance Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>ROC AUC</td>
                <td>{evaluation['roc_auc']:.4f}</td>
                <td>Area under ROC curve (higher is better)</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>{evaluation['mae']:.4f}</td>
                <td>Mean Absolute Error</td>
            </tr>
            <tr>
                <td>MSE</td>
                <td>{evaluation['mse']:.4f}</td>
                <td>Mean Squared Error</td>
            </tr>
            <tr>
                <td>RMSE</td>
                <td>{evaluation['rmse']:.4f}</td>
                <td>Root Mean Squared Error</td>
            </tr>
        </table>
        """

    # Create interesting cases HTML
    cases_html = ""
    if interesting_cases:
        for case in interesting_cases:
            # Format sensor readings
            sensor_html = ""
            for sensor, value in case['sensor_readings'].items():
                if sensor == 'temperature':
                    sensor_html += f"Temperature: {value:.1f}°C, "
                elif sensor == 'vibration':
                    sensor_html += f"Vibration: {value:.2f} mm/s, "
                elif sensor == 'pressure':
                    sensor_html += f"Pressure: {value:.2f} bar, "
                elif sensor == 'current':
                    sensor_html += f"Current: {value:.2f}A, "
                elif sensor == 'voltage':
                    sensor_html += f"Voltage: {value:.1f}V, "
                elif sensor == 'speed':
                    sensor_html += f"Speed: {value:.0f} RPM, "
            
            sensor_html = sensor_html.rstrip(', ')
            
            # Format date if available
            date_str = f"at {case['datetime']}" if case['datetime'] else ""
            
            # Create case HTML
            cases_html += f"""
            <div class="case">
                <h3>Case {case['index']}: {case['actual']} classified as {case['predicted']}</h3>
                <p><strong>Parameters:</strong> {sensor_html}</p>
                <p><strong>Confidence:</strong> {case['confidence']}</p>
                """
                
            if 'key_indicators' in case and case['key_indicators']:
                cases_html += f"<p><strong>Key Indicators:</strong> {case['key_indicators']}</p>"
                
            if 'recommendation' in case and case['recommendation']:
                cases_html += f"<p><strong>Recommendation:</strong> {case['recommendation']}</p>"
                
            cases_html += "</div>"
    else:
        cases_html = "<p>No disagreement cases found in the predictions.</p>"

    # Model name display
    model_display = model_name if model_name else "LLM Model"

    # Create the main HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM-based Anomaly Detection Results</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ background-color: #34495e; color: white; padding: 20px; margin-bottom: 30px; border-radius: 5px; }}
            .header h1 {{ margin: 0; color: white; }}
            .header p {{ margin: 5px 0 0 0; opacity: 0.8; }}
            .metric {{ margin-bottom: 10px; }}
            .metric span {{ font-weight: bold; color: #3498db; }}
            .status-good {{ color: #27ae60; font-weight: bold; }}
            .status-bad {{ color: #e74c3c; font-weight: bold; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .image-container {{ margin-right: 20px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .case {{ background-color: #f8f9fa; padding: 15px; margin-bottom: 15px; border-left: 4px solid #3498db; }}
            .summary-box {{ background-color: #f0f7fb; border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; }}
            .metrics-container {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .metrics-card {{ background-color: #fff; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .metrics-card h4 {{ margin-top: 0; color: #3498db; }}
            .metrics-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px 0; }}
            .metrics-label {{ font-size: 14px; color: #7f8c8d; }}
            footer {{ margin-top: 50px; border-top: 1px solid #eee; padding-top: 20px; color: #7f8c8d; font-size: 0.9em; }}
            iframe {{ width: 100%; height: 500px; border: none; }}
            .interactive-link {{ background-color: #3498db; color: white; padding: 10px; text-decoration: none; border-radius: 5px; display: inline-block; margin-top: 10px; }}
            .interactive-link:hover {{ background-color: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>LLM-based Anomaly Detection Evaluation</h1>
        <p>Model: {model_display}</p>
        <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary-box">
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive evaluation of an LLM-based anomaly detection system for industrial compressors. 
        The system analyzes sensor data to distinguish between true anomalies and normal values.</p>
        <div class="metrics-container">
            <div class="metrics-card">
                <h4>Accuracy</h4>
                <div class="metrics-value">{evaluation['accuracy']:.2%}</div>
                <div class="metrics-label">Overall classification accuracy</div>
            </div>
            <div class="metrics-card">
                <h4>Anomaly F1</h4>
                <div class="metrics-value">{evaluation['f1_anomaly']:.2%}</div>
                <div class="metrics-label">F1 score for anomaly detection</div>
            </div>
            <div class="metrics-card">
                <h4>Normal Value F1</div>
                <div class="metrics-value">{evaluation['f1_normal']:.2%}</div>
                <div class="metrics-label">F1 score for normal value detection</div>
            </div>
            <div class="metrics-card">
                <h4>Total Predictions</h4>
                <div class="metrics-value">{evaluation['class_counts']['total']}</div>
                <div class="metrics-label">Number of analyzed cases</div>
            </div>
        </div>
    </div>
    """
    
    return html_content

def generate_metrics_html(results):
    """Generate HTML for metrics cards"""
    total_cases = len(results)
    correct_predictions = sum(1 for r in results if r['is_correct'])
    accuracy = correct_predictions / total_cases if total_cases > 0 else 0
    
    metrics_html = f"""
        <div class="metric-card">
            <div class="metric-value">{accuracy:.1%}</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_cases}</div>
            <div class="metric-label">Total Cases</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{correct_predictions}</div>
            <div class="metric-label">Correct Predictions</div>
        </div>
    """
    
    return metrics_html

def main():
    """Main function to generate the report"""
    print("Starting report generation...")
    
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "llm", "results")
    report_dir = os.path.join(current_dir, "llm", "report")
    static_dir = os.path.join(report_dir, "static")
    interactive_dir = os.path.join(report_dir, "interactive")
    
    # Create directories if they don't exist
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(interactive_dir, exist_ok=True)
    
    try:
        # Load prediction results
        print("Loading prediction results...")
        df, metrics, analysis_type = load_prediction_results(results_dir)
        
        print(f"Loaded {len(df)} predictions from {analysis_type} analysis")
        
        # Generate report
        print("Generating report...")
        html_content = generate_html_report(
            df=df,
            static_output_dir=static_dir,
            interactive_output_dir=interactive_dir,
            model_name="Groq LLM (qwen-2.5-32b)"
        )
        
        # Save HTML report
        report_file = os.path.join(report_dir, "anomaly_detection_report.html")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"\nReport generation complete!")
        print(f"HTML report saved to: {report_file}")
        print(f"Static visualizations saved to: {static_dir}")
        print(f"Interactive visualizations saved to: {interactive_dir}")
        
    except FileNotFoundError:
        print("Error: No prediction results found. Please run the analysis first.")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()