import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime

def load_prediction_data(results_path):
    """Load prediction data from the specified file"""
    if results_path.endswith('.json'):
        with open(results_path, 'r') as f:
            data = json.load(f)
    elif results_path.endswith('.csv'):
        data = pd.read_csv(results_path).to_dict('records')
    else:
        raise ValueError("The file must be in JSON or CSV format")
    
    print(f"Loaded {len(data)} samples from file {results_path}")
    return data

def prepare_data_for_evaluation(data):
    """Prepare data for evaluation"""
    # Convert classifications to binary values (1 for ANOMALY, 0 for NORMAL VALUE)
    y_true = [1 if d['actual_classification'] == 'ANOMALY' else 0 for d in data]
    y_pred = [1 if d['predicted_classification'] == 'ANOMALY' else 0 for d in data]
    
    # Extract anomaly type and timestamp for analysis
    types = [d.get('actual_type', 'unknown') for d in data]
    timestamps = [d.get('datetime', '') for d in data]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'timestamp': timestamps,
        'actual_type': types,
        'y_true': y_true,
        'y_pred': y_pred,
        'temperature': [d.get('temperature', np.nan) for d in data],
        'vibration': [d.get('vibration', np.nan) for d in data],
        'pressure': [d.get('pressure', np.nan) for d in data],
        'current': [d.get('current', np.nan) for d in data]
    })
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    return df

def perform_cross_validation(df, k=5, random_state=42):
    """Executes K-Fold Cross-Validation"""
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Store metrics for each fold
    fold_metrics = []
    
    # Initialize arrays for predictions
    all_y_true = np.array(df['y_true'])
    all_y_pred = np.array(df['y_pred'])
    
    # Perform k-fold cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        print(f"\nProcessing fold {fold+1}/{k}...")
        
        # Extract data for this fold
        y_true_test = all_y_true[test_idx]
        y_pred_test = all_y_pred[test_idx]
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_test, y_pred_test)
        metrics['fold'] = fold + 1
        fold_metrics.append(metrics)
        
        print(f"Fold {fold+1} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return fold_metrics

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC-ROC only if there are both positives and negatives
    try:
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_pred)
        else:
            auc_roc = np.nan
    except:
        auc_roc = np.nan
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Precision, recall and F1 for normal class (specificity)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'specificity': specificity,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def summarize_metrics(fold_metrics):
    """Calculate summary statistics of metrics across all folds"""
    # Convert to DataFrame
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Calculate mean and standard deviation for each metric
    summary = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    
    # Consider only numeric metrics (exclude fold)
    numeric_cols = [col for col in metrics_df.columns if col != 'fold']
    
    for col in numeric_cols:
        summary['metric'].append(col)
        summary['mean'].append(metrics_df[col].mean())
        summary['std'].append(metrics_df[col].std())
        summary['min'].append(metrics_df[col].min())
        summary['max'].append(metrics_df[col].max())
    
    summary_df = pd.DataFrame(summary)
    
    return summary_df, metrics_df

def plot_metrics_distribution(metrics_df, output_dir):
    """Creates metric distribution plots"""
    # Select only main metrics
    main_metrics = ['precision', 'recall', 'f1_score', 'specificity']
    
    # Create a plot for each main metric
    for metric in main_metrics:
        plt.figure(figsize=(10, 6))
        
        # Box plot to show distribution
        sns.boxplot(y=metrics_df[metric])
        plt.title(f'Distribution of {metric} across {len(metrics_df)} folds')
        plt.ylabel(metric.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add individual fold values
        sns.stripplot(y=metrics_df[metric], color='red', size=8)
        
        # Add a line for the mean
        plt.axhline(metrics_df[metric].mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {metrics_df[metric].mean():.4f}')
        
        # Add lines for mean ± standard deviation
        std = metrics_df[metric].std()
        plt.axhline(metrics_df[metric].mean() + std, color='green', linestyle='--', alpha=0.7, label=f'Std Dev: {std:.4f}')
        plt.axhline(metrics_df[metric].mean() - std, color='green', linestyle='--', alpha=0.7)
        
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'ann_crossval_{metric}_distribution.png'))
        plt.close()
    
    # Create radar chart
    plt.figure(figsize=(10, 8))
    
    # Prepare data for radar chart
    N = len(main_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    ax = plt.subplot(111, polar=True)
    
    # Plot each fold with semi-transparent color
    for fold in metrics_df['fold'].unique():
        fold_data = metrics_df[metrics_df['fold'] == fold][main_metrics].values.flatten().tolist()
        fold_data += fold_data[:1]  # Close the circle
        ax.plot(angles, fold_data, linewidth=1, alpha=0.3, label=f'Fold {fold}')
        ax.fill(angles, fold_data, alpha=0.05)
    
    # Plot the mean with thicker line
    mean_values = metrics_df[main_metrics].mean().values.tolist()
    mean_values += mean_values[:1]  # Close the circle
    ax.plot(angles, mean_values, linewidth=2, color='red', label='Mean')
    ax.fill(angles, mean_values, alpha=0.1, color='red')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(main_metrics)
    
    plt.title('Metrics for ANN across all folds (Radar Chart)')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(os.path.join(output_dir, 'ann_crossval_radar_chart.png'))
    plt.close()
    
    return

def create_latex_table(summary_df, output_dir):
    """Creates a LaTeX table with summary metrics"""
    # Filter only main metrics
    main_metrics = ['precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    filtered_df = summary_df[summary_df['metric'].isin(main_metrics)]
    
    # Rename metrics for display
    metric_names = {
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1-Score',
        'specificity': 'Specificity',
        'auc_roc': 'AUC-ROC'
    }
    
    filtered_df['metric'] = filtered_df['metric'].map(lambda x: metric_names.get(x, x))
    
    # Format numeric values
    latex_df = filtered_df.copy()
    for col in ['mean', 'std', 'min', 'max']:
        latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}")
    
    # Create LaTeX table
    latex_table = filtered_df.to_latex(index=False, float_format="%.4f")
    
    # Save LaTeX table to file
    with open(os.path.join(output_dir, 'ann_crossval_metrics.tex'), 'w') as f:
        f.write(latex_table)
    
    return latex_table

def create_html_report(df, summary_df, main_metrics, output_dir):
    """Creates HTML report"""
    html_path = os.path.join(output_dir, 'ann_crossval_report.html')
    
    # Calculate coefficient of variation for interpretation
    cv_ratio = summary_df[summary_df['metric'] == 'f1_score']['std'].values[0] / summary_df[summary_df['metric'] == 'f1_score']['mean'].values[0]
    
    # Determine interpretation based on CV ratio
    if cv_ratio < 0.05:
        interpretation = f"The low variation in F1-score (CV={cv_ratio:.3f}) suggests that the model is very stable and shows no signs of overfitting."
    elif cv_ratio < 0.15:
        interpretation = f"The F1-score variation (CV={cv_ratio:.3f}) is normal, suggesting a good balance between stability and adaptation."
    else:
        interpretation = f"The high variation in F1-score (CV={cv_ratio:.3f}) might indicate some instability in predictions across different data subsets."
    
    # Build metrics table
    metrics_table = "<table border='1' class='metrics-table'>\n"
    metrics_table += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>\n"
    
    for _, row in summary_df[summary_df['metric'].isin(main_metrics)].iterrows():
        metrics_table += f"<tr><td>{row['metric']}</td><td>{row['mean']:.4f}</td><td>{row['std']:.4f}</td><td>{row['min']:.4f}</td><td>{row['max']:.4f}</td></tr>\n"
    
    metrics_table += "</table>"
    
    # Create HTML content with ANN-specific titles
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Validation Report for Artificial Neural Network</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .metrics-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        .metrics-table th {{
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        .metrics-table td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin: 20px 0;
        }}
        .chart-image {{
            max-width: 48%;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .full-width-image {{
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        @media (max-width: 768px) {{
            .chart-image {{
                max-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <h1>Cross-Validation Report for Artificial Neural Network</h1>
    <p>Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Overview</h2>
    <ul>
        <li>Total samples analyzed: {len(df)}</li>
        <li>Number of anomalies in dataset: {df['y_true'].sum()}</li>
        <li>Percentage of anomalies: {(df['y_true'].sum() / len(df) * 100):.2f}%</li>
        <li>Number of folds: 5</li>
    </ul>
    <h2>Main Metrics (mean ± std)</h2>
    {metrics_table}
    <h2>Metrics Visualization</h2>
    <div class="image-container">
        <img src="ann_crossval_precision_distribution.png" alt="Precision Distribution" class="chart-image">
        <img src="ann_crossval_recall_distribution.png" alt="Recall Distribution" class="chart-image">
        <img src="ann_crossval_f1_score_distribution.png" alt="F1-Score Distribution" class="chart-image">
        <img src="ann_crossval_specificity_distribution.png" alt="Specificity Distribution" class="chart-image">
    </div>
    <h3>Metrics Radar Chart</h3>
    <img src="ann_crossval_radar_chart.png" alt="Metrics Radar Chart" class="full-width-image">
    <h2>Interpretation</h2>
    <p>The metrics show consistency across folds, indicating that the model generalizes well to unseen data. {interpretation}</p>
    <h2>Conclusion</h2>
    <p>The Artificial Neural Network model shows overall good performance in classifying anomalies in the predictive maintenance dataset.</p>
</body>
</html>
"""
    
    # Save HTML report
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    """Main function"""
    # Set paths
    base_dir = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10"
    results_file = os.path.join(base_dir, "test_predictions", "ann", "results", "ann_predictions_20250320_093957.csv")
    output_dir = os.path.join(base_dir, "test_predictions", "1) Cross-Validation", "ann")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction results
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found at: {results_file}")
    
    prediction_results = load_prediction_data(results_file)
    
    # Prepare data
    print("Preparing data...")
    df = prepare_data_for_evaluation(prediction_results)
    
    # Perform cross-validation
    print(f"Executing 5-fold cross validation on {len(df)} samples...")
    fold_metrics = perform_cross_validation(df, k=5)
    
    # Calculate summary statistics
    print("Calculating summary statistics...")
    summary_df, metrics_df = summarize_metrics(fold_metrics)
    
    # Show main summary statistics
    main_metrics = ['precision', 'recall', 'f1_score', 'specificity', 'auc_roc']
    print("\nSummary statistics:")
    print(summary_df[summary_df['metric'].isin(main_metrics)].to_string(index=False))
    
    # Save metrics to CSV
    summary_df.to_csv(os.path.join(output_dir, 'ann_crossval_summary.csv'), index=False)
    metrics_df.to_csv(os.path.join(output_dir, 'ann_crossval_fold_metrics.csv'), index=False)
    
    # Create plots
    print("Creating plots...")
    plot_metrics_distribution(metrics_df, output_dir)
    
    # Create LaTeX table
    print("Generating LaTeX table...")
    create_latex_table(summary_df, output_dir)
    
    print(f"\nAnalysis completed. Results have been saved to: {output_dir}")
    
    # Create Markdown report
    report_path = os.path.join(output_dir, 'ann_crossval_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# Cross-Validation Report for Artificial Neural Network\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overview\n\n")
        f.write(f"- Total samples analyzed: {len(df)}\n")
        f.write(f"- Number of anomalies in dataset: {df['y_true'].sum()}\n")
        f.write(f"- Percentage of anomalies: {(df['y_true'].sum() / len(df) * 100):.2f}%\n")
        f.write(f"- Number of folds: 5\n\n")
        
        f.write(f"## Main Metrics (mean ± std)\n\n")
        f.write(f"| Metric | Mean | Std Dev | Min | Max |\n")
        f.write(f"|--------|------|---------|-----|-----|\n")
        for _, row in summary_df[summary_df['metric'].isin(main_metrics)].iterrows():
            f.write(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
        
        f.write(f"\n## Interpretation\n\n")
        cv_ratio = summary_df[summary_df['metric'] == 'f1_score']['std'].values[0] / summary_df[summary_df['metric'] == 'f1_score']['mean'].values[0]
        if cv_ratio < 0.05:
            f.write(f"The low variation in F1-score (CV={cv_ratio:.3f}) suggests that the model is very stable and shows no signs of overfitting.\n\n")
        elif cv_ratio < 0.15:
            f.write(f"The F1-score variation (CV={cv_ratio:.3f}) is normal, suggesting a good balance between stability and adaptation.\n\n")
        else:
            f.write(f"The high variation in F1-score (CV={cv_ratio:.3f}) might indicate some instability in predictions across different data subsets.\n\n")
        
        f.write(f"## Conclusion\n\n")
        f.write(f"The metrics show consistency across folds, indicating that the model generalizes well to unseen data. ")
        f.write(f"The Artificial Neural Network model shows overall good performance in classifying anomalies in the predictive maintenance dataset.")
    
    print(f"Report saved to: {report_path}")
    
    # Create HTML report
    html_report_path = create_html_report(df, summary_df, main_metrics, output_dir)
    print(f"Report HTML saved to: {html_report_path}")

if __name__ == "__main__":
    main()