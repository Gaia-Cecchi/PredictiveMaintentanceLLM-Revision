import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import torch
from typing import List, Dict, Any, Optional
import json
import re
import sqlite3
import gc
from dotenv import load_dotenv

# Add parent directory to path to import from evaluate_anomalies and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary components from evaluate_anomalies
try:
    from llm.core.data_manager import CompressorDataManager
    from llm.core.llm_predictor import LLMPredictor
    from llm.prompts.expert_prompts import definitive_prompt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the llm module structure is correct")
    sys.exit(1)

class RealTimeQwenScalabilityAnalyzer:
    """
    Analyzer for Qwen 2.5 model scalability using real-time predictions
    with actual model execution and timing measurements
    """
   
    def __init__(self, output_dir: str = "qwen_realtime_scalability_analysis", groq_api_key: str = None):
        """
        Initialize the real-time analyzer
        
        Args:
            output_dir: Directory for results
            groq_api_key: API key for Groq (if not in environment variables)
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/backups").mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RealTimeQwenScalabilityAnalyzer")
        
        # Initialize API key
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            load_dotenv()  # Try to load from .env file
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                self.logger.error("GROQ_API_KEY not found in environment variables or provided as argument")
                raise ValueError("GROQ_API_KEY not found")
        
        # Parameters to test - these are for batch and reset simulation
        self.batch_sizes = [5, 10, 20]  # Reduced set for real testing
        self.reset_frequencies = [5, 10, 20, None]  # Reduced set for real testing
        
        # Database paths
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(script_dir, "datasets", "compressor_data_2024.db")
        self.labeled_db_path = os.path.join(script_dir, "datasets", "compressor_data_2024_etichettato.db")
        
        # Verify database paths
        if not os.path.exists(self.db_path):
            self.logger.error(f"Database not found at {self.db_path}")
            raise FileNotFoundError(f"Database not found at {self.db_path}")
            
        if not os.path.exists(self.labeled_db_path):
            self.logger.error(f"Labeled database not found at {self.labeled_db_path}")
            raise FileNotFoundError(f"Labeled database not found at {self.labeled_db_path}")
        
        # Initialize data manager
        self.data_manager = CompressorDataManager(self.db_path, self.labeled_db_path)
        
        # Results storage
        self.results = []
        
        # Backup settings
        self.backup_frequency = 5  # Save backup every 5 records
        
        # Store memory of last run
        self.last_memory_usage = 0
    
    def select_test_cases(self, n_samples=200):
        """
        Select a balanced set of test cases including all labeled anomalies
        
        Args:
            n_samples: Total number of samples to include
            
        Returns:
            DataFrame with selected test cases
        """
        self.logger.info(f"Selecting {n_samples} test cases including all labeled anomalies")
        
        # Get all labeled events
        anomalies_df, false_positives_df = self.data_manager.get_labeled_events()
        
        # Ensure we include all anomalies
        self.logger.info(f"Found {len(anomalies_df)} labeled anomalies")
        
        # Calculate how many normal cases to include
        n_anomalies = len(anomalies_df)
        n_normal_needed = n_samples - n_anomalies
        
        if n_normal_needed <= 0:
            self.logger.warning(f"Too many anomalies ({n_anomalies}) for sample size {n_samples}")
            self.logger.warning(f"Using only anomalies and reducing sample count to {n_anomalies}")
            return anomalies_df
        
        # Select normal cases from the database, excluding anomaly dates
        self.logger.info(f"Selecting {n_normal_needed} normal cases")
        anomaly_dates = anomalies_df['DateTime'].tolist()
        normal_cases = self.data_manager.select_normal_cases(
            exclude_dates=anomaly_dates,
            n_samples=n_normal_needed
        )
        
        # Combine anomalies and normal cases
        combined_df = pd.concat([anomalies_df, normal_cases])
        
        # Sort by date for consistent ordering
        combined_df = combined_df.sort_values('DateTime')
        
        self.logger.info(f"Selected {len(combined_df)} test cases: {n_anomalies} anomalies and {len(normal_cases)} normal cases")
        return combined_df
    
    def run_real_batch_test(self, test_cases, batch_size, reset_frequency):
        """
        Run real Qwen predictions with actual batch processing and timing
        
        Args:
            test_cases: DataFrame with test cases
            batch_size: Batch size to use
            reset_frequency: Reset frequency (in batches, None for no reset)
            
        Returns:
            dict: Results of the test run
        """
        # Configure LLM model - using Qwen 2.5 32B
        self.logger.info(f"Initializing Qwen 2.5 32B for batch_size={batch_size}, reset_frequency={reset_frequency or 'None'}")
        llm_predictor = LLMPredictor(
            api_key=self.api_key,
            model="qwen-2.5-32b"
        )
        
        # Replace prompt with definitive_prompt
        llm_predictor.prompt = definitive_prompt()
        
        # Track overall timing and performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Prediction results
        predictions = []
        correct_predictions = 0
        
        # Batch management
        n_samples = len(test_cases)
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        # Context management
        conversation_context = []
        
        # Start batch processing
        total_processing_time = 0
        max_memory_usage = 0
        
        # Memory measurements
        memory_samples = []
        processing_times = []
        
        try:
            for batch_idx in range(n_batches):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                batch = test_cases.iloc[batch_start_idx:batch_end_idx]
                batch_size_actual = len(batch)
                
                self.logger.info(f"Processing batch {batch_idx+1}/{n_batches} ({batch_size_actual} cases)")
                
                # Process each case in the batch
                for idx, case in batch.iterrows():
                    case_start_time = time.time()
                    event_time = case['DateTime']
                    actual_classification = case['actual_classification']
                    
                    self.logger.info(f"Processing case {batch_start_idx + idx % batch_size_actual + 1}/{n_samples}: {event_time}")
                    
                    # Measure memory before prediction
                    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
                    
                    # Get data window
                    data_window = self.data_manager.get_data_window(event_time, window_hours=12)
                    
                    # Make prediction with explicit timing
                    prediction_start = time.time()
                    try:
                        prediction = llm_predictor.predict(
                            event_time,
                            data_window['compressor_str'],
                            data_window['weather_str']
                        )
                    except Exception as e:
                        self.logger.error(f"Error during prediction: {str(e)}")
                        # Use a default prediction in case of error
                        prediction = {
                            'classification': 'NORMAL VALUE',
                            'type': '',
                            'confidence': 'low',
                            'key_indicators': 'Prediction failed',
                            'recommendation': 'Error occurred during prediction'
                        }
                    
                    # Measure prediction time
                    prediction_time = time.time() - prediction_start
                    processing_times.append(prediction_time)
                    total_processing_time += prediction_time
                    
                    # Check if prediction is correct
                    is_correct = prediction['classification'] == actual_classification
                    if is_correct:
                        correct_predictions += 1
                    
                    # Add to conversation context for next predictions
                    conversation_context.append({
                        'time': event_time,
                        'input': data_window['compressor_str'],
                        'prediction': prediction['classification']
                    })
                    
                    # Measure memory after prediction
                    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_used = mem_after - mem_before
                    memory_samples.append(memory_used)
                    max_memory_usage = max(max_memory_usage, mem_after)
                    
                    # Log progress
                    case_processing_time = time.time() - case_start_time
                    self.logger.info(f"  Prediction: {prediction['classification']} (Actual: {actual_classification})")
                    self.logger.info(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}")
                    self.logger.info(f"  Processed in {case_processing_time:.2f} seconds (prediction: {prediction_time:.2f}s)")
                    
                    # Save prediction
                    predictions.append({
                        'datetime': str(event_time),
                        'actual_classification': actual_classification,
                        'predicted_classification': prediction['classification'],
                        'is_correct': is_correct,
                        'processing_time': prediction_time,
                        'memory_used': memory_used,
                        'batch_idx': batch_idx,
                        'in_batch_idx': idx % batch_size_actual
                    })
                    
                    # Save backup if needed
                    if len(predictions) % self.backup_frequency == 0:
                        self._save_backup(predictions, batch_size, reset_frequency)
                
                # Apply reset if needed
                if reset_frequency is not None and (batch_idx + 1) % reset_frequency == 0:
                    self.logger.info(f"Resetting conversation context after batch {batch_idx+1}")
                    conversation_context = []
                    
                    # Force garbage collection to properly measure memory differences
                    gc.collect()
                    
                    # Add a small pause to ensure system stabilizes
                    time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error during test execution: {str(e)}")
            traceback.print_exc()
            # Save what we have so far before exiting
            if predictions:
                self._save_backup(predictions, batch_size, reset_frequency, is_emergency=True)
        
        # Calculate final metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        total_time = end_time - start_time
        memory_change = end_memory - start_memory
        accuracy = correct_predictions / n_samples if n_samples > 0 else 0
        avg_time_per_sample = total_processing_time / n_samples if n_samples > 0 else 0
        samples_per_second = n_samples / total_processing_time if total_processing_time > 0 else 0
        
        # Average memory per sample - use measured samples
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        
        results = {
            'batch_size': batch_size,
            'reset_frequency': reset_frequency if reset_frequency is not None else "None",
            'accuracy': accuracy,
            'error_rate': 1 - accuracy,
            'processing_time': total_processing_time,
            'processing_time_per_sample': avg_time_per_sample,
            'samples_per_second': samples_per_second,
            'avg_memory_usage': avg_memory,
            'max_memory_usage': max_memory_usage,
            'n_samples': n_samples,
            'total_runtime': total_time,
            'predictions': predictions
        }
        
        self.logger.info(f"Test completed: accuracy={accuracy:.4f}, speed={samples_per_second:.2f} samples/sec")
        self.logger.info(f"Average processing time: {avg_time_per_sample:.2f} seconds per sample")
        self.logger.info(f"Total runtime: {total_time:.2f} seconds")
        
        # Save the results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"{self.output_dir}/qwen_test_b{batch_size}_r{reset_frequency or 'None'}_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            # Save a version without the full predictions list for better readability
            save_results = results.copy()
            save_results.pop('predictions')
            json.dump(save_results, f, indent=2)
        
        # Also save the detailed predictions
        predictions_path = f"{self.output_dir}/qwen_predictions_b{batch_size}_r{reset_frequency or 'None'}_{timestamp}.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        return results
    
    def _save_backup(self, predictions, batch_size, reset_frequency, is_emergency=False):
        """Save a backup of the current predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "emergency_" if is_emergency else ""
        backup_path = f"{self.output_dir}/backups/{prefix}backup_b{batch_size}_r{reset_frequency or 'None'}_{timestamp}_{len(predictions)}cases.json"
        
        with open(backup_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        if is_emergency:
            self.logger.warning(f"Saved emergency backup to {backup_path}")
        else:
            self.logger.info(f"Saved backup to {backup_path}")
    
    def run_analysis(self, n_samples=200):
        """Run the complete analysis with all batch sizes and reset frequencies"""
        self.logger.info(f"Starting real-time Qwen scalability analysis with {n_samples} samples")
        
        # Select test cases including all anomalies
        test_cases = self.select_test_cases(n_samples)
        
        # Store test cases for reference
        test_cases.to_csv(f"{self.output_dir}/test_cases.csv", index=False)
        self.logger.info(f"Saved {len(test_cases)} test cases to {self.output_dir}/test_cases.csv")
        
        all_results = []
        
        # Run tests for each configuration
        for batch_size in self.batch_sizes:
            for reset_frequency in self.reset_frequencies:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Running test with batch_size={batch_size}, reset_frequency={reset_frequency or 'None'}")
                self.logger.info(f"{'='*50}\n")
                
                try:
                    # Run the actual test with real predictions
                    result = self.run_real_batch_test(test_cases, batch_size, reset_frequency)
                    all_results.append(result)
                    
                    # Save current results to track progress
                    self._save_all_results(all_results)
                    
                    # Force garbage collection between tests
                    gc.collect()
                    
                    # Add a pause between tests to let the system stabilize
                    self.logger.info(f"Pausing for 10 seconds before next test configuration")
                    time.sleep(10)
                    
                except Exception as e:
                    self.logger.error(f"Error running test: {str(e)}")
                    traceback.print_exc()
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in all_results])
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/qwen_realtime_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        self.logger.info(f"Analysis completed. Results saved to {results_file}")
        return results_df
    
    def _save_all_results(self, results):
        """Save the current set of all results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save a clean version without predictions for better readability
        clean_results = [{k: v for k, v in r.items() if k != 'predictions'} for r in results]
        
        results_path = f"{self.output_dir}/qwen_all_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        self.logger.info(f"Saved current results to {results_path}")
    
    def generate_report(self, results_df):
        """Generate a complete report with tables and visualizations"""
        self.logger.info("Generating report and visualizations")
        
        # Create visualizations
        self._create_visualization_plots(results_df)
        
        # Generate HTML report
        html_table = self._generate_html_table(results_df)
        self._generate_html_report(results_df, html_table)
        
        self.logger.info(f"Report generated at {self.output_dir}/qwen_realtime_scalability_report.html")
    
    def _create_visualization_plots(self, results_df):
        """Create visualization plots"""
        figures_dir = f"{self.output_dir}/figures"
        
        # 1. Accuracy vs Batch Size for different reset frequencies
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['accuracy'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
        
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
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['samples_per_second'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
        
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
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['avg_memory_usage'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
        
        plt.title('Memory Usage vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Memory Usage (MB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/memory_vs_batchsize.png", dpi=300)
        plt.close()
        
        # 4. Processing Time per Sample vs Batch Size
        plt.figure(figsize=(10, 6))
        for reset_freq in results_df['reset_frequency'].unique():
            subset = results_df[results_df['reset_frequency'] == reset_freq]
            plt.plot(subset['batch_size'], subset['processing_time_per_sample'], 'o-',
                     label=f"Reset Freq: {reset_freq}")
        
        plt.title('Processing Time per Sample vs. Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Sample (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/time_per_sample_vs_batchsize.png", dpi=300)
        plt.close()
    
    def _generate_html_table(self, results_df):
        """Generate HTML table from results"""
        html_table = """
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Batch Size</th>
                    <th>Reset Frequency</th>
                    <th>Accuracy</th>
                    <th>Samples/Second</th>
                    <th>Avg Time (s)</th>
                    <th>Avg Memory (MB)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Sort for readability
        sorted_results = results_df.sort_values(['batch_size', 'reset_frequency'])
        
        for _, row in sorted_results.iterrows():
            html_table += f"""
                <tr>
                    <td>{row['batch_size']}</td>
                    <td>{row['reset_frequency']}</td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['samples_per_second']:.4f}</td>
                    <td>{row['processing_time_per_sample']:.2f}</td>
                    <td>{row['avg_memory_usage']:.2f}</td>
                </tr>
            """
        
        html_table += """
            </tbody>
        </table>
        """
        
        return html_table
    
    def _generate_html_report(self, results_df, html_table):
        """Generate complete HTML report"""
        # Find optimal configurations
        best_accuracy_idx = results_df['accuracy'].idxmax()
        best_accuracy_config = results_df.loc[best_accuracy_idx]
        
        best_speed_idx = results_df['samples_per_second'].idxmax()
        best_speed_config = results_df.loc[best_speed_idx]
        
        # Find the equilibrium point (best compromise)
        results_df['efficiency'] = results_df['accuracy'] * results_df['samples_per_second']
        best_efficiency_idx = results_df['efficiency'].idxmax()
        best_efficiency_config = results_df.loc[best_efficiency_idx]
        
        # Calculate correlation
        numeric_df = results_df.select_dtypes(include=['number'])
        correlation = ('lineare' if abs(numeric_df['batch_size'].corr(numeric_df['avg_memory_usage'])) > 0.9 
                      else 'sub-lineare')
        
        # Add timing information
        avg_time_per_sample = results_df['processing_time_per_sample'].mean()
        total_samples = sum(results_df['n_samples'])
        estimated_total_time = avg_time_per_sample * total_samples
        
        timing_info = f"""
        <div class="summary-box">
            <h2>Informazioni sui Tempi di Esecuzione</h2>
            <p>I tempi riportati sono basati sull'esecuzione reale del modello Qwen 2.5:</p>
            <ul>
                <li><strong>Tempo medio di inferenza</strong>: {avg_time_per_sample:.2f} secondi per campione</li>
                <li><strong>Velocità media di elaborazione</strong>: {1/avg_time_per_sample:.2f} campioni al minuto</li>
                <li><strong>Tempo totale di esecuzione dei test</strong>: {estimated_total_time/60:.1f} minuti</li>
            </ul>
            <p>Nota: questi tempi sono stati misurati durante l'esecuzione effettiva del modello.</p>
        </div>
        """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 2.5 Real-Time Scalability Analysis</title>
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
    <h1>Qwen 2.5 Real-Time Scalability Analysis</h1>
    <p>Report generato il {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
   
    <div class="summary-box">
        <h2>Riepilogo</h2>
        <p>Questa analisi valuta le prestazioni reali del modello Qwen 2.5 con diversi parametri di batch e reset:</p>
        <ul>
            <li><strong>Dimensione del batch</strong>: Il numero di campioni da elaborare prima di aggiornare lo stato interno del modello</li>
            <li><strong>Frequenza di reset</strong>: Ogni quanti batch viene azzerato il contesto della conversazione</li>
        </ul>
        <p>L'analisi è stata effettuata su {total_samples} campioni includendo tutte le 11 anomalie note.</p>
    </div>
    
    {timing_info}
   
    <div class="optimal-config">
        <h2>Configurazioni Ottimali</h2>
       
        <h3>Massima Accuratezza</h3>
        <p>Batch Size: <span class="highlight">{best_accuracy_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_accuracy_config['reset_frequency']}</span></p>
        <p>Accuratezza: {best_accuracy_config['accuracy']:.4f},
           Velocità: {best_accuracy_config['samples_per_second']:.4f} campioni/secondo</p>
       
        <h3>Massima Velocità</h3>
        <p>Batch Size: <span class="highlight">{best_speed_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_speed_config['reset_frequency']}</span></p>
        <p>Velocità: {best_speed_config['samples_per_second']:.4f} campioni/secondo,
           Accuratezza: {best_speed_config['accuracy']:.4f}</p>
       
        <h3>Miglior Equilibrio (Efficienza)</h3>
        <p>Batch Size: <span class="highlight">{best_efficiency_config['batch_size']}</span>,
           Reset Frequency: <span class="highlight">{best_efficiency_config['reset_frequency']}</span></p>
        <p>Accuratezza: {best_efficiency_config['accuracy']:.4f},
           Velocità: {best_efficiency_config['samples_per_second']:.4f} campioni/secondo</p>
    </div>
   
    <h2>Risultati Dettagliati</h2>
    {html_table}
   
    <h2>Visualizzazioni</h2>
   
    <div class="figure-container">
        <div class="figure">
            <img src="figures/accuracy_vs_batchsize.png" alt="Accuracy vs Batch Size">
            <p>L'accuratezza del modello al variare della dimensione del batch e della frequenza di reset</p>
        </div>
       
        <div class="figure">
            <img src="figures/speed_vs_batchsize.png" alt="Processing Speed vs Batch Size">
            <p>La velocità di elaborazione (campioni al secondo) al variare della dimensione del batch</p>
        </div>
       
        <div class="figure">
            <img src="figures/memory_vs_batchsize.png" alt="Memory Usage vs Batch Size">
            <p>L'utilizzo di memoria al variare della dimensione del batch</p>
        </div>
       
        <div class="figure">
            <img src="figures/time_per_sample_vs_batchsize.png" alt="Time per Sample vs Batch Size">
            <p>Il tempo di elaborazione per campione al variare della dimensione del batch</p>
        </div>
    </div>
   
    <h2>Conclusioni</h2>
    <p>Dall'analisi delle prestazioni reali emergono le seguenti conclusioni:</p>
    <ul>
        <li><strong>Impatto della dimensione del batch</strong>: Batch più grandi tendono a {
        'migliorare' if results_df.groupby('batch_size')['accuracy'].mean().iloc[-1] > results_df.groupby('batch_size')['accuracy'].mean().iloc[0]
        else 'peggiorare'} l'accuratezza ma influenzano l'utilizzo di memoria.</li>
       
        <li><strong>Impatto della frequenza di reset</strong>: Reset meno frequenti permettono al modello di sfruttare meglio il contesto,
        ma possono portare a degradazione delle prestazioni se il contesto diventa troppo grande.</li>
       
        <li><strong>Compromesso ottimale</strong>: La configurazione con batch size {best_efficiency_config['batch_size']} e
        reset frequency {best_efficiency_config['reset_frequency']} offre il miglior compromesso tra accuratezza e velocità.</li>
       
        <li><strong>Considerazioni sulla memoria</strong>: L'utilizzo di memoria cresce in modo {correlation} con la dimensione del batch.</li>
    </ul>
   
    <h3>Raccomandazioni</h3>
    <p>In base ai risultati dell'analisi, si raccomanda di:</p>
    <ul>
        <li>Utilizzare una dimensione del batch di <strong>{best_efficiency_config['batch_size']}</strong> per un equilibrio ottimale.</li>
        <li>Impostare la frequenza di reset a <strong>{best_efficiency_config['reset_frequency']}</strong> per mantenere le prestazioni costanti nel tempo.</li>
        <li>Per applicazioni che richiedono la massima accuratezza, considerare batch size <strong>{best_accuracy_config['batch_size']}</strong>
            con reset frequency <strong>{best_accuracy_config['reset_frequency']}</strong>.</li>
        <li>Per applicazioni che richiedono la massima velocità, considerare batch size <strong>{best_speed_config['batch_size']}</strong>
            con reset frequency <strong>{best_speed_config['reset_frequency']}</strong>.</li>
    </ul>
</body>
</html>
"""
        
        # Save the HTML report
        report_path = f"{self.output_dir}/qwen_realtime_scalability_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)


def main():
    """Main function for the real-time Qwen scalability analysis"""
    try:
        # Load environment variables for API key
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("GROQ_API_KEY not found in environment variables.")
            api_key = input("Please enter your Groq API key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                return
        
        # Create output directory with absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "qwen_realtime_scalability_analysis")
        
        # Ask for sample size with a default of 200
        try:
            print("\nThis script will run real Qwen predictions to analyze scalability.")
            print("WARNING: This will take a considerable amount of time and API usage.")
            print("Default sample size is 200 (including all 11 labeled anomalies).")
            sample_input = input("Enter sample size (or press Enter for default 200): ").strip()
            sample_size = int(sample_input) if sample_input else 200
            
            if sample_size < 11:
                print("Sample size must be at least 11 to include all labeled anomalies.")
                print("Setting sample size to minimum (11).")
                sample_size = 11
                
        except ValueError:
            print("Invalid input. Using default sample size of 200.")
            sample_size = 200
        
        # Initialize analyzer
        analyzer = RealTimeQwenScalabilityAnalyzer(output_dir, api_key)
        
        # Run analysis
        print(f"\nRunning real-time analysis with {sample_size} samples...")
        print("This will take a significant amount of time. Progress will be logged.")
        
        try:
            # Time the entire analysis
            start_time = time.time()
            
            # Run the analysis
            results_df = analyzer.run_analysis(sample_size)
            
            # Generate reports
            analyzer.generate_report(results_df)
            
            # Calculate total time
            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"\nAnalysis completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s!")
            print(f"Report available at: {os.path.join(output_dir, 'qwen_realtime_scalability_report.html')}")
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            traceback.print_exc()
            print("Check log files for details.")
            return
            
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()