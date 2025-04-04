import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path


class EnglishReportGenerator:
    """Generate scalability analysis reports in English from the Qwen model results"""
    
    def __init__(self, results_csv_path, output_dir="qwen_english_reports"):
        """Initialize the report generator
        
        Args:
            results_csv_path: Path to the CSV file with test results
            output_dir: Directory where reports will be saved
        """
        self.results_csv_path = results_csv_path
        self.output_dir = output_dir
        
        # Create output directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        
        # Load results data
        self.results_df = pd.read_csv(results_csv_path)
        print(f"Loaded {len(self.results_df)} test results from {results_csv_path}")
        
        # Convert reset_frequency to numeric for correlation analysis
        # Handle NaN values properly
        self.results_df['reset_freq_num'] = self.results_df['reset_frequency'].apply(
            lambda x: 9999 if pd.isna(x) or x == "None" else int(float(x)))
    
    def create_visualization_plots(self):
        """Create visualization plots in English"""
        figures_dir = f"{self.output_dir}/figures"
        
        # 1. Accuracy vs Batch Size for different reset frequencies
        plt.figure(figsize=(10, 6))
        for reset_freq in self.results_df['reset_frequency'].unique():
            if pd.isna(reset_freq):
                # Handle NaN values in the plot legend
                label = "Reset Freq: None"
            else:
                label = f"Reset Freq: {reset_freq}"
                
            subset = self.results_df[self.results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['accuracy'], 'o-', label=label)
        
        plt.title('Accuracy vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/accuracy_vs_batchsize.png", dpi=300)
        plt.close()
        
        # 2. Processing Speed vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in self.results_df['reset_frequency'].unique():
            if pd.isna(reset_freq):
                label = "Reset Freq: None"
            else:
                label = f"Reset Freq: {reset_freq}"
                
            subset = self.results_df[self.results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['samples_per_second'], 'o-', label=label)
        
        plt.title('Processing Speed vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/speed_vs_batchsize.png", dpi=300)
        plt.close()
        
        # 3. Memory Usage vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in self.results_df['reset_frequency'].unique():
            if pd.isna(reset_freq):
                label = "Reset Freq: None"
            else:
                label = f"Reset Freq: {reset_freq}"
                
            subset = self.results_df[self.results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['avg_memory_usage'], 'o-', label=label)
        
        plt.title('Memory Usage vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Memory Usage (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/memory_vs_batchsize.png", dpi=300)
        plt.close()
        
        # 4. Accuracy Heatmap for Batch Size and Reset Frequency
        plt.figure(figsize=(12, 8))
        # Use reset_freq_num which properly handles NaN values
        pivot_table = self.results_df.pivot_table(
            values='accuracy',
            index='batch_size',
            columns='reset_freq_num'
        )
        
        # Rename columns for clarity
        pivot_table.columns = [
            'None' if c == 9999 else str(c) for c in pivot_table.columns
        ]
        
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu",
                   linewidths=.5, cbar_kws={"label": "Accuracy"})
        
        plt.title('Accuracy Heatmap: Batch Size vs. Reset Frequency')
        plt.xlabel('Reset Frequency')
        plt.ylabel('Batch Size')
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/accuracy_heatmap.png", dpi=300)
        plt.close()
        
        # 5. Processing Time per Sample vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in self.results_df['reset_frequency'].unique():
            if pd.isna(reset_freq):
                label = "Reset Freq: None"
            else:
                label = f"Reset Freq: {reset_freq}"
                
            subset = self.results_df[self.results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['processing_time_per_sample'], 'o-', label=label)
        
        plt.title('Processing Time per Sample vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Sample (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/time_per_sample_vs_batchsize.png", dpi=300)
        plt.close()
        
        print(f"Created 5 visualization plots in {figures_dir}")
    
    def generate_results_table(self):
        """Generate an HTML table with the results"""
        html_table = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Batch Size</th>
                    <th>Reset Frequency</th>
                    <th>Accuracy</th>
                    <th>Samples/Second</th>
                    <th>Avg Memory (MB)</th>
                    <th>Time per Sample (s)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Sort by batch_size and reset_frequency for a more readable table
        sorted_results = self.results_df.sort_values(['batch_size', 'reset_freq_num'])
        
        for _, row in sorted_results.iterrows():
            # Handle NaN in reset_frequency for display
            reset_freq_display = "None" if pd.isna(row['reset_frequency']) else row['reset_frequency']
            
            html_table += f"""
                <tr>
                    <td>{row['batch_size']}</td>
                    <td>{reset_freq_display}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['samples_per_second']:.2f}</td>
                    <td>{row['avg_memory_usage']:.2f}</td>
                    <td>{row['processing_time_per_sample']:.4f}</td>
                </tr>
            """
        
        html_table += """
            </tbody>
        </table>
        """
        
        return html_table
    
    def generate_html_report(self):
        """Generate a complete HTML report in English"""
        # Find optimal configurations
        best_accuracy_idx = self.results_df['accuracy'].idxmax()
        best_accuracy_config = self.results_df.loc[best_accuracy_idx]
        
        best_speed_idx = self.results_df['samples_per_second'].idxmax()
        best_speed_config = self.results_df.loc[best_speed_idx]
        
        # Find the equilibrium point (best compromise between accuracy and speed)
        self.results_df['efficiency'] = self.results_df['accuracy'] * self.results_df['samples_per_second']
        best_efficiency_idx = self.results_df['efficiency'].idxmax()
        best_efficiency_config = self.results_df.loc[best_efficiency_idx]
        
        # Calculate correlation between batch_size and avg_memory_usage
        numeric_df = self.results_df.select_dtypes(include=['number'])
        correlation = 'linear' if abs(numeric_df['batch_size'].corr(numeric_df['avg_memory_usage'])) > 0.9 else 'sub-linear'
        
        # Format display values, handling NaN
        best_acc_reset = "None" if pd.isna(best_accuracy_config['reset_frequency']) else best_accuracy_config['reset_frequency']
        best_speed_reset = "None" if pd.isna(best_speed_config['reset_frequency']) else best_speed_config['reset_frequency']
        best_eff_reset = "None" if pd.isna(best_efficiency_config['reset_frequency']) else best_efficiency_config['reset_frequency']
        
        # Get the results table
        html_table = self.generate_results_table()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 2.5 Scalability Analysis</title>
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
        .figure-container {{
            margin: 30px 0;
        }}
        .figure-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .figure {{
            max-width: 100%;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .optimal-config {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .highlight {{
            font-weight: bold;
            color: #2980b9;
        }}
    </style>
</head>
<body>
    <h1>Qwen 2.5 Scalability Analysis</h1>
    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary-box">
        <h2>Summary</h2>
        <p>This analysis evaluates how the Qwen 2.5 model's performance is influenced by two key parameters:</p>
        <ul>
            <li><strong>Batch Size</strong>: The number of samples to process before updating the model's internal state</li>
            <li><strong>Reset Frequency</strong>: How often the conversation context is cleared (in batches)</li>
        </ul>
        <p>The analysis was performed on a dataset with normal operation cases and anomalies.</p>
    </div>
    
    <div class="optimal-config">
        <h2>Optimal Configurations</h2>
        
        <h3>Maximum Accuracy</h3>
        <p>Batch Size: <span class="highlight">{best_accuracy_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_acc_reset}</span></p>
        <p>Accuracy: {best_accuracy_config['accuracy']:.4f},
           Speed: {best_accuracy_config['samples_per_second']:.2f} samples/second</p>
        
        <h3>Maximum Speed</h3>
        <p>Batch Size: <span class="highlight">{best_speed_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_speed_reset}</span></p>
        <p>Speed: {best_speed_config['samples_per_second']:.2f} samples/second,
           Accuracy: {best_speed_config['accuracy']:.4f}</p>
        
        <h3>Best Balance (Efficiency)</h3>
        <p>Batch Size: <span class="highlight">{best_efficiency_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_eff_reset}</span></p>
        <p>Accuracy: {best_efficiency_config['accuracy']:.4f},
           Speed: {best_efficiency_config['samples_per_second']:.2f} samples/second</p>
    </div>
    
    <h2>Detailed Results</h2>
    {html_table}
    
    <h2>Visualizations</h2>
    
    <div class="figure-container">
        <div class="figure">
            <img src="figures/accuracy_vs_batchsize.png" alt="Accuracy vs Batch Size">
            <p>The model's accuracy at varying batch sizes and reset frequencies</p>
        </div>
        
        <div class="figure">
            <img src="figures/speed_vs_batchsize.png" alt="Processing Speed vs Batch Size">
            <p>Processing speed (samples per second) at varying batch sizes</p>
        </div>
        
        <div class="figure">
            <img src="figures/memory_vs_batchsize.png" alt="Memory Usage vs Batch Size">
            <p>Memory usage at varying batch sizes</p>
        </div>
        
        <div class="figure">
            <img src="figures/time_per_sample_vs_batchsize.png" alt="Time per Sample vs Batch Size">
            <p>Processing time per sample at varying batch sizes</p>
        </div>
        
        <div class="figure">
            <img src="figures/accuracy_heatmap.png" alt="Accuracy Heatmap">
            <p>Accuracy heatmap for different combinations of batch size and reset frequency</p>
        </div>
    </div>
    
    <h2>Conclusions</h2>
    <p>The following conclusions emerge from the analysis:</p>
    <ul>
        <li><strong>Impact of batch size</strong>: Larger batches tend to {
        'improve' if self.results_df.groupby('batch_size')['accuracy'].mean().iloc[-1] > self.results_df.groupby('batch_size')['accuracy'].mean().iloc[0]
        else 'worsen'} accuracy but increase memory usage.</li>
        
        <li><strong>Impact of reset frequency</strong>: Less frequent resets allow the model to better leverage context,
        but can lead to performance degradation if the context becomes too large.</li>
        
        <li><strong>Optimal compromise</strong>: The configuration with batch size {best_efficiency_config['batch_size']} and
        reset frequency {best_eff_reset} offers the best compromise between accuracy and speed.</li>
        
        <li><strong>Memory considerations</strong>: Memory usage grows {correlation}ly with batch size,
        suggesting that the model {
        'efficiently handles' if correlation == 'sub-linear' else 'scales with'} larger batches.</li>
    </ul>
    
    <h3>Recommendations</h3>
    <p>Based on the analysis results, we recommend:</p>
    <ul>
        <li>Using a batch size of <strong>{best_efficiency_config['batch_size']}</strong> for optimal balance.</li>
        <li>Setting the reset frequency to <strong>{best_eff_reset}</strong> to maintain consistent performance over time.</li>
        <li>For applications requiring maximum accuracy, consider batch size <strong>{best_accuracy_config['batch_size']}</strong>
            with reset frequency <strong>{best_acc_reset}</strong>.</li>
        <li>For applications requiring maximum speed, consider batch size <strong>{best_speed_config['batch_size']}</strong>
            with reset frequency <strong>{best_speed_reset}</strong>.</li>
    </ul>
</body>
</html>
"""
        
        # Save the HTML report
        report_path = f"{self.output_dir}/qwen_scalability_report_english.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"Generated English HTML report: {report_path}")
        return report_path
    
    def run(self):
        """Run the complete report generation process"""
        print(f"Starting English report generation from {self.results_csv_path}")
        self.create_visualization_plots()
        report_path = self.generate_html_report()
        print(f"English report generation completed. Report available at {report_path}")
        return report_path


def main():
    """Main function to run the English report generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate English reports from Qwen scalability test results")
    parser.add_argument("--input", "-i", required=False, help="Path to the CSV file with test results")
    parser.add_argument("--output", "-o", default="qwen_english_reports", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # If input is not provided, look for the most recent results file
    if not args.input:
        # Look in the default output directory with complete path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(script_dir, "qwen_scalability_analysis_datisintetici")
        
        if os.path.exists(default_dir):
            csv_files = [f for f in os.listdir(default_dir) if f.endswith('.csv') and 'results' in f]
            if csv_files:
                # Sort by modification time (most recent first)
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(default_dir, x)), reverse=True)
                args.input = os.path.join(default_dir, csv_files[0])
                print(f"Using most recent results file: {args.input}")
            else:
                print("No results files found in the default directory. Please specify an input file.")
                return
        else:
            print(f"Default directory not found: {default_dir}")
            print("Please specify an input file using the --input argument.")
            return
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found.")
        return
    
    try:
        # Create the report generator and run it
        generator = EnglishReportGenerator(args.input, args.output)
        report_path = generator.run()
        
        # Open the report in the default browser
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        
    except Exception as e:
        print(f"Error generating English report: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
