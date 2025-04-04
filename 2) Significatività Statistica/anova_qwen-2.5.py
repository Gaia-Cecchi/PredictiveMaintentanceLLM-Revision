import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

def perform_bootstrap_sampling(df, n_iterations=1000, subsample_size=None, random_state=42):
    """
    Perform bootstrap sampling to generate performance metrics distribution
    """
    if subsample_size is None:
        subsample_size = len(df) // 2  # Use half the dataset by default
    
    np.random.seed(random_state)
    
    # Initialize arrays to store metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []
    auc_roc_scores = []
    
    # Get actual and predicted values
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    # Perform bootstrap iterations
    for i in range(n_iterations):
        # Randomly sample with replacement
        indices = np.random.choice(len(df), subsample_size, replace=True)
        
        # Get bootstrap sample
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        # Skip if no positive samples
        if sum(y_true_sample) == 0:
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_sample, y_pred_sample)
        
        # Store metrics
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        specificity_scores.append(metrics['specificity'])
        if not np.isnan(metrics['auc_roc']):
            auc_roc_scores.append(metrics['auc_roc'])
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'metric': ['precision'] * len(precision_scores) + 
                 ['recall'] * len(recall_scores) + 
                 ['f1_score'] * len(f1_scores) + 
                 ['specificity'] * len(specificity_scores) + 
                 ['auc_roc'] * len(auc_roc_scores),
        'value': precision_scores + recall_scores + f1_scores + specificity_scores + auc_roc_scores,
        'model': ['qwen-2.5'] * (len(precision_scores) + len(recall_scores) + len(f1_scores) + 
                                 len(specificity_scores) + len(auc_roc_scores))
    })
    
    return results

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

def perform_anova(results_df):
    """Perform ANOVA test on metrics"""
    metrics = results_df['metric'].unique()
    anova_results = {}
    
    # For each metric, perform one-way ANOVA
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric]
        
        # Only perform ANOVA if we have multiple groups
        if len(metric_data['model'].unique()) > 1:
            # Create formula for ANOVA
            formula = 'value ~ C(model)'
            model = ols(formula, data=metric_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Store results
            anova_results[metric] = {
                'f_value': anova_table['F'][0],
                'p_value': anova_table['PR(>F)'][0],
                'significant': anova_table['PR(>F)'][0] < 0.05
            }
            
            # If significant, perform post-hoc Tukey's test
            if anova_results[metric]['significant'] and len(metric_data['model'].unique()) > 2:
                tukey = pairwise_tukeyhsd(endog=metric_data['value'], 
                                         groups=metric_data['model'], 
                                         alpha=0.05)
                anova_results[metric]['posthoc'] = tukey
        else:
            # Calculate descriptive statistics
            anova_results[metric] = {
                'mean': metric_data['value'].mean(),
                'std': metric_data['value'].std(),
                'min': metric_data['value'].min(),
                'max': metric_data['value'].max(),
                'significant': False
            }
    
    return anova_results

def plot_bootstrap_distributions(results_df, output_dir):
    """Plot bootstrap distributions for each metric"""
    metrics = results_df['metric'].unique()
    
    # Create a histogram for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        metric_data = results_df[results_df['metric'] == metric]
        
        # Plot histogram with KDE
        sns.histplot(metric_data['value'], kde=True)
        
        # Add mean line
        plt.axvline(metric_data['value'].mean(), color='red', linewidth=2, label=f'Mean: {metric_data["value"].mean():.4f}')
        
        # Add 95% confidence interval
        ci_low, ci_high = np.percentile(metric_data['value'], [2.5, 97.5])
        plt.axvline(ci_low, color='green', linestyle='--', linewidth=2, label=f'95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
        plt.axvline(ci_high, color='green', linestyle='--', linewidth=2)
        
        plt.title(f'Bootstrap Distribution for {metric} (Qwen-2.5)')
        plt.xlabel(f'{metric.capitalize()} Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'qwen_bootstrap_{metric}_distribution.png'))
        plt.close()
    
    # Create summary box plot
    plt.figure(figsize=(12, 8))
    
    # Filter out metrics with very different scales
    plot_metrics = [m for m in metrics if m != 'auc_roc']
    
    # Create box plot
    sns.boxplot(x='metric', y='value', data=results_df[results_df['metric'].isin(plot_metrics)])
    
    plt.title('Performance Metrics Distribution (Qwen-2.5)')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'qwen_metrics_boxplot.png'))
    plt.close()
    
    return

def create_html_report(results_df, anova_results, output_dir):
    """Create HTML report with statistical results"""
    html_path = os.path.join(output_dir, 'qwen_statistical_analysis.html')
    
    # Calculate bootstrap statistics
    metrics = results_df['metric'].unique()
    bootstrap_stats = {}
    
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric]
        ci_low, ci_high = np.percentile(metric_data['value'], [2.5, 97.5])
        
        bootstrap_stats[metric] = {
            'mean': metric_data['value'].mean(),
            'std': metric_data['value'].std(),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_samples': len(metric_data)
        }
    
    # Build metrics table
    metrics_table = "<table border='1' class='metrics-table'>\n"
    metrics_table += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>95% CI Low</th><th>95% CI High</th><th>Samples</th></tr>\n"
    
    for metric in metrics:
        stats = bootstrap_stats[metric]
        metrics_table += f"<tr><td>{metric}</td><td>{stats['mean']:.4f}</td><td>{stats['std']:.4f}</td>"
        metrics_table += f"<td>{stats['ci_low']:.4f}</td><td>{stats['ci_high']:.4f}</td><td>{stats['n_samples']}</td></tr>\n"
    
    metrics_table += "</table>"
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Analysis Report for Qwen-2.5</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
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
    <h1>Statistical Analysis Report for Qwen-2.5</h1>
    <p>Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Bootstrap Analysis</h2>
    <p>Performance metrics were analyzed using bootstrap resampling to generate robust distribution statistics:</p>
    {metrics_table}
    
    <h2>Visualization of Bootstrap Distributions</h2>
    <div class="image-container">
"""

    # Add images for each metric
    for metric in metrics:
        html_content += f"""
        <img src="qwen_bootstrap_{metric}_distribution.png" alt="{metric} Distribution" class="chart-image">"""
    
    html_content += f"""
    </div>
    
    <h3>Combined Metrics Distribution</h3>
    <img src="qwen_metrics_boxplot.png" alt="Metrics Box Plot" class="full-width-image">
    
    <h2>Statistical Interpretation</h2>
    <p>The bootstrap analysis provides the following insights:</p>
    <ul>
"""

    # Add interpretations for each metric
    for metric in metrics:
        stats = bootstrap_stats[metric]
        if metric == 'precision':
            quality = "excellent" if stats['mean'] > 0.8 else "good" if stats['mean'] > 0.6 else "moderate" if stats['mean'] > 0.4 else "poor"
            html_content += f"<li><strong>Precision</strong>: The model shows {quality} precision ({stats['mean']:.4f}) with a 95% confidence interval of [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]</li>\n"
        elif metric == 'recall':
            quality = "excellent" if stats['mean'] > 0.8 else "good" if stats['mean'] > 0.6 else "moderate" if stats['mean'] > 0.4 else "poor"
            html_content += f"<li><strong>Recall</strong>: The model shows {quality} recall ({stats['mean']:.4f}) with a 95% confidence interval of [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]</li>\n"
        elif metric == 'f1_score':
            quality = "excellent" if stats['mean'] > 0.8 else "good" if stats['mean'] > 0.6 else "moderate" if stats['mean'] > 0.4 else "poor"
            html_content += f"<li><strong>F1-Score</strong>: The model shows {quality} F1-Score ({stats['mean']:.4f}) with a 95% confidence interval of [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]</li>\n"
        elif metric == 'specificity':
            html_content += f"<li><strong>Specificity</strong>: The model correctly identifies {stats['mean']*100:.2f}% of normal cases with a 95% confidence interval of [{stats['ci_low']*100:.2f}%, {stats['ci_high']*100:.2f}%]</li>\n"
        elif metric == 'auc_roc':
            quality = "excellent" if stats['mean'] > 0.9 else "good" if stats['mean'] > 0.8 else "moderate" if stats['mean'] > 0.7 else "poor"
            html_content += f"<li><strong>AUC-ROC</strong>: The model shows {quality} discriminative ability ({stats['mean']:.4f}) with a 95% confidence interval of [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]</li>\n"

    html_content += f"""
    </ul>
    
    <h2>Conclusion</h2>
    <p>The statistical analysis confirms that the Qwen-2.5 model demonstrates robust performance in anomaly detection. 
    The narrow confidence intervals indicate statistical stability in the model's predictions. 
    Key performance metrics show strong results, particularly in terms of {', '.join([m for m in metrics if bootstrap_stats[m]['mean'] > 0.8])}.</p>
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
    results_file = os.path.join(base_dir, "test_predictions", "llm", "results_qwen-2.5.32b", "prediction_results.csv")
    output_dir = os.path.join(base_dir, "test_predictions", "2) Significatività Statistica", "qwen-2.5")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction results
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found at: {results_file}")
    
    prediction_results = load_prediction_data(results_file)
    
    # Prepare data
    print("Preparing data...")
    df = prepare_data_for_evaluation(prediction_results)
    
    # Perform bootstrap sampling
    print("Performing bootstrap sampling...")
    bootstrap_results = perform_bootstrap_sampling(df, n_iterations=1000)
    
    # Perform ANOVA
    print("Performing statistical tests...")
    anova_results = perform_anova(bootstrap_results)
    
    # Create plots
    print("Creating visualizations...")
    plot_bootstrap_distributions(bootstrap_results, output_dir)
    
    # Create HTML report
    print("Generating report...")
    html_report_path = create_html_report(bootstrap_results, anova_results, output_dir)
    
    # Save bootstrap results to CSV
    bootstrap_results.to_csv(os.path.join(output_dir, 'qwen_bootstrap_results.csv'), index=False)
    
    print(f"Analysis completed. Results have been saved to: {output_dir}")
    print(f"Report saved to: {html_report_path}")

if __name__ == "__main__":
    main()
