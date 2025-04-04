import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_metrics():
    """Load metrics for all models"""
    qwen_metrics = {
        'precision': [0.893333, 0.153478, 0.666667, 1.0],
        'recall': [1.000000, 0.000000, 1.000000, 1.0],
        'f1_score': [0.937778, 0.090813, 0.800000, 1.0],
        'auc_roc': [0.999105, 0.001225, 0.997758, 1.0],
        'specificity': [0.998210, 0.002451, 0.995516, 1.0]
    }
    
    ann_metrics = {
        'precision': [0.610000, 0.384708, 0.000000, 1.0],
        'recall': [0.750000, 0.433013, 0.000000, 1.0],
        'f1_score': [0.661111, 0.391065, 0.000000, 1.0],
        'auc_roc': [0.967886, 0.062696, 0.873845, 1.0],
        'specificity': [0.998160, 0.001029, 0.997691, 1.0]
    }

    cnn_metrics = {
        'precision': [0.130000, 0.185742, 0.000000, 0.400000],
        'recall': [0.250000, 0.433013, 0.000000, 1.000000],
        'f1_score': [0.164286, 0.252033, 0.000000, 0.571429],
        'auc_roc': [0.651645, 0.237638, 0.494226, 0.996552],
        'specificity': [0.988513, 0.005586, 0.979405, 0.993103]
    }

    lstm_metrics = {
        'precision': [0.750000, 0.250000, 0.500000, 1.000000],
        'recall': [0.750000, 0.250000, 0.500000, 1.000000],
        'f1_score': [0.750000, 0.250000, 0.500000, 1.000000],
        'auc_roc': [0.874752, 0.124752, 0.750000, 0.999504],
        'specificity': [0.999504, 0.000496, 0.999008, 1.000000]
    }
    
    return qwen_metrics, ann_metrics, cnn_metrics, lstm_metrics

def create_comparison_plots(metrics_dict, output_dir):
    """Create comparison plots between all models"""
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc', 'specificity']
    models = ['Qwen 2.5', 'ANN', 'CNN', 'LSTM']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
    
    # Bar plot comparing means with error bars
    plt.figure(figsize=(15, 8))
    x = range(len(metrics))
    width = 0.2  # Reduced width to fit 4 bars
    
    for i, (model, color) in enumerate(zip(models, colors)):
        model_metrics = metrics_dict[i]
        plt.bar([j + width*(i-1.5) for j in x], 
                [model_metrics[m][0] for m in metrics], 
                width, 
                label=model,
                color=color,
                yerr=[model_metrics[m][1] for m in metrics],
                capsize=5)
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

    # Create radar chart
    plt.figure(figsize=(12, 8))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    
    for model, color in zip(models, colors):
        model_metrics = metrics_dict[models.index(model)]
        values = [model_metrics[m][0] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    plt.title('Model Comparison Radar Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'))
    plt.close()

def create_comparison_report(metrics_dict, output_dir):
    """Create a detailed comparison report"""
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc', 'specificity']
    models = ['Qwen 2.5', 'ANN', 'CNN', 'LSTM']
    
    with open(report_path, 'w') as f:
        f.write("# Model Comparison: Qwen 2.5 vs ANN vs CNN vs LSTM\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Metric | Qwen 2.5 (mean ± std) | ANN (mean ± std) | CNN (mean ± std) | LSTM (mean ± std) | Winner |\n")
        f.write("|---------|---------------------|------------------|------------------|------------------|--------|\n")
        
        for metric in metrics:
            qwen_val = f"{metrics_dict[0][metric][0]:.4f} ± {metrics_dict[0][metric][1]:.4f}"
            ann_val = f"{metrics_dict[1][metric][0]:.4f} ± {metrics_dict[1][metric][1]:.4f}"
            cnn_val = f"{metrics_dict[2][metric][0]:.4f} ± {metrics_dict[2][metric][1]:.4f}"
            lstm_val = f"{metrics_dict[3][metric][0]:.4f} ± {metrics_dict[3][metric][1]:.4f}"
            
            values = [metrics_dict[i][metric][0] for i in range(4)]
            winner = models[values.index(max(values))]
            if max(values) - min(values) < 0.01:
                winner = "Tie"
                
            f.write(f"| {metric} | {qwen_val} | {ann_val} | {cnn_val} | {lstm_val} | {winner} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Overall performance comparison
        f.write("### Overall Performance\n")
        best_model = models[0]
        best_f1_score = metrics_dict[0]['f1_score'][0]
        for i in range(1, 4):
            if metrics_dict[i]['f1_score'][0] > best_f1_score:
                best_f1_score = metrics_dict[i]['f1_score'][0]
                best_model = models[i]
        
        diff_percent = ((best_f1_score - min([metrics_dict[i]['f1_score'][0] for i in range(4)])) / min([metrics_dict[i]['f1_score'][0] for i in range(4)]) * 100)
        f.write(f"- {best_model} outperforms other models by {diff_percent:.1f}% in terms of F1-score\n")
        
        # Stability comparison
        cv_values = [metrics_dict[i]['f1_score'][1] / metrics_dict[i]['f1_score'][0] for i in range(4)]
        best_stability_model = models[cv_values.index(min(cv_values))]
        
        f.write("\n### Model Stability\n")
        for i, model in enumerate(models):
            f.write(f"- {model} coefficient of variation: {cv_values[i]:.3f}\n")
        f.write(f"- {best_stability_model} shows better stability across folds\n")
        
        f.write("\n## Conclusion\n\n")
        if best_model == best_stability_model:
            f.write(f"{best_model} is the superior model, showing both better performance and more stability across all metrics. ")
            f.write("It achieves higher precision, recall, and F1-score while maintaining more consistent predictions across different data splits.")
        else:
            f.write("The comparison shows mixed results, with each model having its strengths:\n\n")
            for model in models:
                f.write(f"- **{model}** excels in: " + ", ".join([m for m in metrics if metrics_dict[models.index(model)][m][0] > max([metrics_dict[i][m][0] for i in range(4) if i != models.index(model)])]) + "\n")

def create_html_report(metrics_dict, output_dir):
    """Creates HTML comparison report"""
    html_path = os.path.join(output_dir, 'model_comparison_report.html')
    metrics = ['precision', 'recall', 'f1_score', 'auc_roc', 'specificity']
    models = ['Qwen 2.5', 'ANN', 'CNN', 'LSTM']
    
    # Build metrics table
    metrics_table = "<table border='1' class='metrics-table'>\n"
    metrics_table += "<tr><th>Metric</th>"
    for model in models:
        metrics_table += f"<th>{model} (mean &plusmn; std)</th>"
    metrics_table += "<th>Best Model</th></tr>\n"
    
    for metric in metrics:
        metrics_table += f"<tr><td>{metric}</td>"
        values = []
        for i, model in enumerate(models):
            mean = metrics_dict[i][metric][0]
            std = metrics_dict[i][metric][1]
            values.append(mean)
            metrics_table += f"<td>{mean:.4f} &plusmn; {std:.4f}</td>"
        
        best_model = models[values.index(max(values))]
        metrics_table += f"<td>{best_model}</td></tr>\n"
    
    metrics_table += "</table>"
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
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
        }}
        .metrics-table td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        .metrics-table tr:nth-child(even) {{
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
    <h1>Model Comparison Report</h1>
    <p>Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Comparison</h2>
    {metrics_table}
    
    <h2>Visualization</h2>
    <div class="image-container">
        <h3>Bar Chart Comparison</h3>
        <img src="model_comparison.png" alt="Model Performance Comparison">
        
        <h3>Radar Chart Comparison</h3>
        <img src="model_comparison_radar.png" alt="Model Comparison Radar Chart">
    </div>
    
    <h2>Conclusion</h2>
    <p>Based on the analysis of all metrics:</p>
    <ul>
        <li>Qwen 2.5 shows the best overall performance with highest F1-score and most consistent results</li>
        <li>LSTM shows second-best performance with good balance between precision and recall</li>
        <li>ANN shows moderate performance with higher variability</li>
        <li>CNN shows lower performance compared to other models</li>
    </ul>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    """Main function"""
    base_dir = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10"
    output_dir = os.path.join(base_dir, "test_predictions", "1) Cross-Validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics for all models
    qwen_metrics, ann_metrics, cnn_metrics, lstm_metrics = load_metrics()
    metrics_dict = [qwen_metrics, ann_metrics, cnn_metrics, lstm_metrics]
    
    # Create comparison plots
    create_comparison_plots(metrics_dict, output_dir)
    
    # Create comparison reports
    create_comparison_report(metrics_dict, output_dir)
    create_html_report(metrics_dict, output_dir)
    
    print("Comparison analysis completed. Check the output directory for results.")

if __name__ == "__main__":
    main()
