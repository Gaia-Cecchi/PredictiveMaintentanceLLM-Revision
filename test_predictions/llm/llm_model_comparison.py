import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from datetime import datetime

def load_and_process_data(file_path):
    """Load and process prediction results from CSV file"""
    df = pd.read_csv(file_path)
    # Ensure all columns we need are present
    required_cols = ['actual_binary', 'predicted_binary']
    
    # Check if required columns exist
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Required columns missing in {file_path}")
        return None
    
    # Convert to numeric if needed
    for col in required_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in required columns
    df = df.dropna(subset=required_cols)
    
    return df

def calculate_metrics(df):
    """Calculate all metrics for a dataframe"""
    if df is None or len(df) == 0:
        return None
    
    y_true = df['actual_binary'].values
    y_pred = df['predicted_binary'].values
    
    # Handle case where all true values are the same (AUC-ROC undefined)
    auc_roc = 0
    if len(np.unique(y_true)) > 1:
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except:
            auc_roc = 0
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': auc_roc
    }
    
    # Calculate MAE and MSE if confidence_score is available
    if 'confidence_score' in df.columns:
        # Convert confidence score to reflect error when prediction is wrong
        df['error'] = np.where(df['actual_binary'] != df['predicted_binary'], 
                              df['confidence_score'], 0)
        metrics['mae'] = mean_absolute_error(np.ones(len(df)), 1 - df['error'])
        metrics['mse'] = mean_squared_error(np.ones(len(df)), 1 - df['error'])
    else:
        metrics['mae'] = 0
        metrics['mse'] = 0
    
    return metrics

def create_comparison_plots(metrics_dict, model_names, output_dir):
    """Create comparison plots between all models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mae', 'mse']
    # Use a color palette that can accommodate 6 models
    colors = sns.color_palette("husl", len(model_names))
    
    # Bar plot comparing metrics
    plt.figure(figsize=(15, 8))
    x = range(len(metrics))
    width = 0.15  # Adjust width to fit all bars
    
    for i, (model, color) in enumerate(zip(model_names, colors)):
        plt.bar([j + width*(i-(len(model_names)-1)/2) for j in x], 
                [metrics_dict[model][m] for m in metrics], 
                width, 
                label=model,
                color=color)
    
    plt.ylabel('Score')
    plt.title('LLM Model Performance Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'llm_model_comparison.png'), bbox_inches='tight')
    plt.close()

    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    # We'll exclude MAE and MSE from radar chart as they're error metrics
    radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    
    for i, (model, color) in enumerate(zip(model_names, colors)):
        values = [metrics_dict[model][m] for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    
    plt.title('LLM Model Comparison Radar Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(os.path.join(output_dir, 'llm_model_comparison_radar.png'))
    plt.close()

def main():
    """Main function"""
    base_dir = r"c:\Users\gaia1\Desktop\UDOO Lab\Predictive Maintenance & LLMs\code 10"
    output_dir = os.path.join(base_dir, "test_predictions", "llm")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model files and their display names
    model_files = {
        'Qwen 2.5': os.path.join(base_dir, 'test_predictions', 'llm', 'results_qwen-2.5.32b', 'prediction_results.csv'),
        'Qwen QWQ': os.path.join(base_dir, 'test_predictions', 'llm', 'results_qwen-qwq-32b', 'prediction_results_correct.csv'),
        'Llama 3.2 90B': os.path.join(base_dir, 'test_predictions', 'llm', 'results_llama-3.2-90b-vision-preview - 332 casi', 'prediction_results_correct.csv'),
        'Llama 3.2 11B': os.path.join(base_dir, 'test_predictions', 'llm', 'results_llama-3.2-11b-vision-preview - 222 casi', 'prediction_results_correct.csv'),
        'Llama 3.3 70B': os.path.join(base_dir, 'test_predictions', 'llm', 'results_llama-3.3-70b-versatile - 107 casi', 'prediction_results_correct.csv'),
        'DeepSeek R1': os.path.join(base_dir, 'test_predictions', 'llm', 'results_deepseek-r1-distill-qwen-32b', 'prediction_results_correct.csv')
    }
    
    # Load and process data
    metrics_dict = {}
    for model_name, file_path in model_files.items():
        df = load_and_process_data(file_path)
        if df is not None:
            metrics = calculate_metrics(df)
            metrics_dict[model_name] = metrics
            print(f"Processed {model_name}: {len(df)} records")
        else:
            print(f"Failed to process {model_name}")
    
    # Create comparison plots
    if metrics_dict:
        create_comparison_plots(metrics_dict, list(metrics_dict.keys()), output_dir)
        print(f"Comparison charts created in {output_dir}")
    else:
        print("No valid metrics to compare")

if __name__ == "__main__":
    main()
