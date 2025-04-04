import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def load_bootstrap_results():
    """Load bootstrap results for all models"""
    base_dir = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10\test_predictions\2) Significatività Statistica"
    
    # Load results for each model
    qwen_results = pd.read_csv(os.path.join(base_dir, "qwen-2.5", "qwen_bootstrap_results.csv"))
    ann_results = pd.read_csv(os.path.join(base_dir, "ann", "ann_bootstrap_results.csv"))
    cnn_results = pd.read_csv(os.path.join(base_dir, "cnn", "cnn_bootstrap_results.csv"))
    lstm_results = pd.read_csv(os.path.join(base_dir, "lstm", "lstm_bootstrap_results.csv"))
    
    # Combine all results
    all_results = pd.concat([qwen_results, ann_results, cnn_results, lstm_results])
    
    return all_results

def perform_statistical_tests(results_df):
    """Perform ANOVA and post-hoc tests on metrics"""
    metrics = results_df['metric'].unique()
    statistical_results = {}
    
    for metric in metrics:
        metric_data = results_df[results_df['metric'] == metric]
        
        # Perform one-way ANOVA
        models = metric_data['model'].unique()
        model_groups = [metric_data[metric_data['model'] == model]['value'] for model in models]
        
        f_stat, p_value = stats.f_oneway(*model_groups)
        
        # Perform Tukey's HSD test
        tukey = pairwise_tukeyhsd(endog=metric_data['value'], 
                                 groups=metric_data['model'],
                                 alpha=0.05)
        
        statistical_results[metric] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'tukey_results': tukey
        }
    
    return statistical_results

def create_comparison_plots(results_df, output_dir):
    """Create comparison plots between all models"""
    metrics = results_df['metric'].unique()
    models = results_df['model'].unique()
    
    # Violin plots for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=results_df[results_df['metric'] == metric], 
                      x='model', y='value')
        plt.title(f'Distribution of {metric} across Models')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'model_comparison_{metric}_violin.png'))
        plt.close()
    
    # Box plots for all metrics
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=results_df, x='metric', y='value', hue='model')
    plt.title('Performance Metrics Distribution by Model')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'model_comparison_boxplot.png'))
    plt.close()

def create_html_report(results_df, statistical_results, output_dir):
    """Create HTML report with statistical comparison results"""
    metrics = results_df['metric'].unique()
    models = results_df['model'].unique()
    
    # Calculate summary statistics
    summary_stats = results_df.groupby(['metric', 'model'])['value'].agg(['mean', 'std']).reset_index()
    
    # Build summary table
    summary_table = "<table border='1' class='metrics-table'>\n"
    summary_table += "<tr><th>Metric</th><th>Model</th><th>Mean &plusmn; Std</th></tr>\n"
    
    for _, row in summary_stats.iterrows():
        summary_table += f"<tr><td>{row['metric']}</td><td>{row['model']}</td>"
        summary_table += f"<td>{row['mean']:.4f} &plusmn; {row['std']:.4f}</td></tr>\n"
    
    summary_table += "</table>"
    
    # Build statistical results table
    stats_table = "<table border='1' class='stats-table'>\n"
    stats_table += "<tr><th>Metric</th><th>F-statistic</th><th>p-value</th><th>Significant</th></tr>\n"
    
    for metric, results in statistical_results.items():
        stats_table += f"<tr><td>{metric}</td>"
        stats_table += f"<td>{results['f_statistic']:.4f}</td>"
        stats_table += f"<td>{results['p_value']:.4f}</td>"
        stats_table += f"<td>{'Yes' if results['significant'] else 'No'}</td></tr>\n"
    
    stats_table += "</table>"
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistical Model Comparison Report</title>
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
        .metrics-table, .stats-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .image-container {{
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>Statistical Model Comparison Report</h1>
    <p>Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Summary</h2>
    {summary_table}
    
    <h2>Statistical Test Results</h2>
    <p>One-way ANOVA results:</p>
    {stats_table}
    
    <h2>Post-hoc Analysis</h2>
    <p>Tukey's HSD test results for each metric:</p>
"""

    # Add Tukey's test results
    for metric, results in statistical_results.items():
        html_content += f"<h3>{metric}</h3>"
        html_content += f"<pre>{results['tukey_results']}</pre>"

    html_content += """
    <h2>Visualization</h2>
    <div class="image-container">
"""

    # Add violin plots
    for metric in metrics:
        html_content += f"""
        <h3>Distribution of {metric}</h3>
        <img src="model_comparison_{metric}_violin.png" alt="{metric} Distribution">
"""

    html_content += """
        <h3>Combined Metrics Distribution</h3>
        <img src="model_comparison_boxplot.png" alt="Combined Metrics Distribution">
    </div>
    
    <h2>Conclusion</h2>
"""

    # Add conclusion based on statistical results
    significant_metrics = [m for m, r in statistical_results.items() if r['significant']]
    if significant_metrics:
        html_content += f"<p>Statistically significant differences were found between models for the following metrics: {', '.join(significant_metrics)}. "
        html_content += "The post-hoc analysis reveals which specific models differ significantly from each other.</p>"
    else:
        html_content += "<p>No statistically significant differences were found between models for any metrics.</p>"

    html_content += """
</body>
</html>
"""

    # Save HTML report
    html_path = os.path.join(output_dir, 'model_comparison_statistical.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    """Main function"""
    base_dir = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10"
    output_dir = os.path.join(base_dir, "test_predictions", "2) Significatività Statistica")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load bootstrap results from all models
    print("Loading bootstrap results...")
    all_results = load_bootstrap_results()
    
    # Perform statistical tests
    print("Performing statistical tests...")
    statistical_results = perform_statistical_tests(all_results)
    
    # Create comparison plots
    print("Creating visualizations...")
    create_comparison_plots(all_results, output_dir)
    
    # Create comparison report
    print("Generating report...")
    html_report_path = create_html_report(all_results, statistical_results, output_dir)
    
    print("Comparison analysis completed. Check the output directory for results.")
    print(f"Report saved to: {html_report_path}")

if __name__ == "__main__":
    main()
