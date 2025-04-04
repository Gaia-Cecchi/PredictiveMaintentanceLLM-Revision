import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import logging
from datetime import datetime
from pathlib import Path
import traceback
import json
import gc
from dotenv import load_dotenv

# Add parent directory to path to import from evaluate_anomalies
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary components
try:
    from llm.core.data_manager import CompressorDataManager
    from llm.core.llm_predictor import LLMPredictor
    from llm.prompts.expert_prompts import definitive_prompt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the llm module structure is correct")
    sys.exit(1)

class QuickQwenScalabilityAnalyzer:
    """
    Quick Analyzer for Qwen 2.5 model scalability using minimal test configurations
    """
   
    def __init__(self, output_dir: str = "qwen_quicktest_scalability", groq_api_key: str = None):
        """
        Initialize the quick analyzer
        
        Args:
            output_dir: Directory for results
            groq_api_key: API key for Groq (if not in environment variables)
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/backups").mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{output_dir}/quick_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("QuickQwenScalabilityAnalyzer")
        
        # Initialize API key
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            load_dotenv()  # Try to load from .env file
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key:
                self.logger.error("GROQ_API_KEY not found in environment variables")
                raise ValueError("GROQ_API_KEY not found")
        
        # Minimal test parameters - just 1-2 of each for quick testing
        self.batch_sizes = [5]  # Just one batch size for quickest test
        self.reset_frequencies = [None]  # Just no reset for quickest test
        
        # Database paths
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(script_dir, "datasets", "compressor_data_2024.db")
        self.labeled_db_path = os.path.join(script_dir, "datasets", "compressor_data_2024_etichettato.db")
        
        # Verify database paths
        if not os.path.exists(self.db_path) or not os.path.exists(self.labeled_db_path):
            self.logger.error(f"Database files not found")
            raise FileNotFoundError(f"Database files not found")
        
        # Initialize data manager
        self.data_manager = CompressorDataManager(self.db_path, self.labeled_db_path)
        
        # Results storage
        self.results = []
        
        # Backup settings - save after each test
        self.backup_frequency = 5  # Save backup every 5 records
    
    def select_minimal_test_cases(self, n_samples=25):
        """
        Select a minimal balanced test set, prioritizing all labeled anomalies
        
        Args:
            n_samples: Minimum total number of samples to include (will include 
                       all anomalies even if more than n_samples)
            
        Returns:
            DataFrame with selected test cases
        """
        self.logger.info(f"Selecting minimal test set (~{n_samples} samples)")
        
        # Get all labeled anomalies - these MUST be included
        anomalies_df, _ = self.data_manager.get_labeled_events()
        n_anomalies = len(anomalies_df)
        
        self.logger.info(f"Found {n_anomalies} labeled anomalies - including all")
        
        # If we have fewer anomalies than requested samples, add some normal cases
        if n_anomalies < n_samples:
            n_normal_needed = n_samples - n_anomalies
            self.logger.info(f"Adding {n_normal_needed} normal cases to balance test set")
            
            # Get normal cases excluding anomaly dates
            anomaly_dates = anomalies_df['DateTime'].tolist()
            normal_cases = self.data_manager.select_normal_cases(
                exclude_dates=anomaly_dates,
                n_samples=n_normal_needed
            )
            
            # Combine and sort by date
            test_cases = pd.concat([anomalies_df, normal_cases])
            test_cases = test_cases.sort_values('DateTime')
        else:
            # Just use anomalies, potentially sampling down if there are too many
            if n_anomalies > n_samples * 2:
                self.logger.warning(f"Too many anomalies ({n_anomalies}), sampling down to {n_samples}")
                test_cases = anomalies_df.sample(n_samples, random_state=42)
            else:
                test_cases = anomalies_df
            
            test_cases = test_cases.sort_values('DateTime')
        
        self.logger.info(f"Final test set: {len(test_cases)} samples ({n_anomalies} anomalies)")
        return test_cases
    
    def run_batch_test(self, test_cases, batch_size, reset_frequency):
        """
        Run a single batch test with minimal configuration
        
        Args:
            test_cases: DataFrame with test cases
            batch_size: Batch size to use
            reset_frequency: Reset frequency (None = no reset)
            
        Returns:
            dict: Test results
        """
        # Initialize LLM predictor
        self.logger.info(f"Running batch test: batch_size={batch_size}, reset_frequency={reset_frequency or 'None'}")
        
        llm_predictor = LLMPredictor(
            api_key=self.api_key,
            model="qwen-2.5-32b"
        )
        llm_predictor.prompt = definitive_prompt()
        
        # Track time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Test stats
        predictions = []
        correct_count = 0
        n_samples = len(test_cases)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Conversation context
        context = []
        
        # Timing and memory stats
        processing_times = []
        memory_samples = []
        
        try:
            for batch_idx in range(n_batches):
                # Get current batch
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, n_samples)
                batch = test_cases.iloc[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_idx+1}/{n_batches} ({len(batch)} samples)")
                
                # Process each sample in the batch
                for _, sample in batch.iterrows():
                    # Get sample data
                    event_time = sample['DateTime']
                    actual_class = sample['actual_classification']
                    
                    # Log progress (without using too many lines)
                    self.logger.info(f"Processing sample at {event_time} (actual: {actual_class})")
                    
                    # Memory before prediction
                    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
                    
                    # Get data window
                    data_window = self.data_manager.get_data_window(event_time, window_hours=12)
                    
                    # Time the prediction
                    pred_start = time.time()
                    try:
                        prediction = llm_predictor.predict(
                            event_time,
                            data_window['compressor_str'],
                            data_window['weather_str']
                        )
                    except Exception as e:
                        self.logger.error(f"Error in prediction: {str(e)}")
                        prediction = {
                            'classification': 'NORMAL VALUE',
                            'confidence': 'low',
                            'key_indicators': 'Error in prediction',
                            'recommendation': 'Check error logs',
                            'type': ''
                        }
                    
                    # Record time
                    pred_time = time.time() - pred_start
                    processing_times.append(pred_time)
                    
                    # Check correctness
                    is_correct = prediction['classification'] == actual_class
                    if is_correct:
                        correct_count += 1
                    
                    # Memory after prediction
                    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
                    memory_used = mem_after - mem_before
                    memory_samples.append(memory_used)
                    
                    # Add to context
                    context.append({
                        'time': event_time,
                        'input': data_window['compressor_str'],
                        'prediction': prediction['classification']
                    })
                    
                    # Save result
                    predictions.append({
                        'datetime': str(event_time),
                        'actual_classification': actual_class,
                        'predicted_classification': prediction['classification'],
                        'is_correct': is_correct,
                        'processing_time': pred_time,
                        'memory_used': memory_used
                    })
                    
                    # Simple progress summary
                    self.logger.info(f"Result: {prediction['classification']} ({'✓' if is_correct else '✗'}) - {pred_time:.2f}s")
                    
                    # Backup periodically
                    if len(predictions) % self.backup_frequency == 0:
                        self._save_backup(predictions, batch_size, reset_frequency)
                
                # Apply reset if configured
                if reset_frequency is not None and (batch_idx + 1) % reset_frequency == 0:
                    self.logger.info(f"Resetting context after batch {batch_idx+1}")
                    context = []
                    gc.collect()
        
        except Exception as e:
            self.logger.error(f"Error during test: {str(e)}")
            traceback.print_exc()
            # Save what we have so far
            if predictions:
                self._save_backup(predictions, batch_size, reset_frequency, is_emergency=True)
        
        # Calculate final metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        total_time = end_time - start_time
        memory_change = end_memory - start_memory
        accuracy = correct_count / n_samples if n_samples > 0 else 0
        
        # Calculate averages
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        
        # Compile results
        result = {
            'batch_size': batch_size,
            'reset_frequency': reset_frequency if reset_frequency is not None else "None",
            'accuracy': accuracy,
            'processing_time_per_sample': avg_time,
            'samples_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'avg_memory_usage': avg_memory,
            'max_memory_usage': max(memory_samples) if memory_samples else 0,
            'n_samples': n_samples,
            'total_runtime': total_time,
            'predictions': predictions
        }
        
        # Log summary
        self.logger.info(f"Test completed: accuracy={accuracy:.4f}, " 
                         f"time_per_sample={avg_time:.2f}s, "
                         f"samples/sec={1.0/avg_time:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"{self.output_dir}/qwen_quick_b{batch_size}_r{reset_frequency or 'None'}_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            # Save a version without the full predictions
            save_result = result.copy()
            save_result.pop('predictions')
            json.dump(save_result, f, indent=2)
        
        return result
    
    def _save_backup(self, predictions, batch_size, reset_frequency, is_emergency=False):
        """Save a backup of predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "emergency_" if is_emergency else ""
        backup_path = f"{self.output_dir}/backups/{prefix}quick_b{batch_size}_r{reset_frequency or 'None'}_{timestamp}_{len(predictions)}cases.json"
        
        with open(backup_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        self.logger.info(f"Saved backup to {backup_path}")
    
    def run_quicktest(self, n_samples=25, single_config=True, custom_config=None):
        """
        Run a quick scalability test
        
        Args:
            n_samples: Approximate number of samples to test
            single_config: If True, only test one configuration
            custom_config: Custom configuration dict with batch_size and reset_frequency
            
        Returns:
            DataFrame with test results
        """
        self.logger.info("Starting quick Qwen scalability test")
        
        # Select test cases
        test_cases = self.select_minimal_test_cases(n_samples)
        
        # Store test cases for reference
        test_cases.to_csv(f"{self.output_dir}/quick_test_cases.csv", index=False)
        
        all_results = []
        
        # Use custom config if provided
        if custom_config:
            self.logger.info(f"Using custom config: {custom_config}")
            configs_to_test = [custom_config]
        # Test single config if requested
        elif single_config:
            self.logger.info("Testing only one configuration (batch_size=5, reset=None)")
            configs_to_test = [{'batch_size': 5, 'reset_frequency': None}]
        # Otherwise test all combinations (still a reduced set)
        else:
            configs_to_test = []
            for bs in self.batch_sizes:
                for rf in self.reset_frequencies:
                    configs_to_test.append({'batch_size': bs, 'reset_frequency': rf})
        
        # Run tests for each configuration
        for config in configs_to_test:
            bs = config['batch_size']
            rf = config['reset_frequency']
            
            self.logger.info(f"\n===== Testing batch_size={bs}, reset_frequency={rf or 'None'} =====")
            
            try:
                # Run the test
                result = self.run_batch_test(test_cases, bs, rf)
                all_results.append(result)
                
                # Save current results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"{self.output_dir}/quick_all_results_{timestamp}.json", 'w') as f:
                    # Save without predictions for readability
                    save_results = [{k: v for k, v in r.items() if k != 'predictions'} for r in all_results]
                    json.dump(save_results, f, indent=2)
                
                # Force garbage collection
                gc.collect()
                
                # Pause between tests
                if len(configs_to_test) > 1:
                    self.logger.info("Pausing 5 seconds before next test")
                    time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error running test with config {config}: {str(e)}")
                traceback.print_exc()
        
        # Convert to DataFrame
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'predictions'} for r in all_results])
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.output_dir}/qwen_quicktest_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Generate simple report
        self._generate_simple_report(results_df)
        
        self.logger.info(f"Quick test completed. Results saved to {results_file}")
        return results_df
    
    def _generate_simple_report(self, results_df):
        """Generate a simple single-page report"""
        if len(results_df) == 0:
            self.logger.warning("No results to generate report from")
            return
        
        # Simple table
        table_rows = ""
        for _, row in results_df.iterrows():
            table_rows += f"""
            <tr>
                <td>{row['batch_size']}</td>
                <td>{row['reset_frequency']}</td>
                <td>{row['accuracy']:.4f}</td>
                <td>{row['samples_per_second']:.4f}</td>
                <td>{row['processing_time_per_sample']:.2f}</td>
                <td>{row['avg_memory_usage']:.2f}</td>
            </tr>
            """
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Qwen Quick Scalability Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Qwen 2.5 Quick Scalability Test Results</h1>
    <p>Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Results Summary</h2>
    <table>
        <tr>
            <th>Batch Size</th>
            <th>Reset Frequency</th>
            <th>Accuracy</th>
            <th>Samples/Sec</th>
            <th>Time/Sample (s)</th>
            <th>Avg Memory (MB)</th>
        </tr>
        {table_rows}
    </table>
    
    <h3>Key Observations</h3>
    <ul>
        <li>Average processing time: {results_df['processing_time_per_sample'].mean():.2f} seconds per sample</li>
        <li>Average accuracy: {results_df['accuracy'].mean():.4f}</li>
        <li>Average memory usage: {results_df['avg_memory_usage'].mean():.2f} MB</li>
    </ul>
    
    <p><em>Note: This is a quick test with minimal samples. For more comprehensive analysis, run the full scalability test.</em></p>
</body>
</html>
"""
        
        # Save report
        report_path = f"{self.output_dir}/qwen_quicktest_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"Simple report generated at {report_path}")


def main():
    """Main function to run quick Qwen scalability test"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("GROQ_API_KEY not found in environment variables.")
            api_key = input("Please enter your Groq API key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                return
        
        # Create output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "qwen_quicktest_scalability")
        
        # Options for quick testing
        print("\n=== Quick Qwen Scalability Test ===")
        print("This script will quickly test Qwen's performance with minimal configurations.")
        
        # Get sample size
        try:
            sample_input = input("Enter sample size (default: 25, min: 11 for all anomalies): ").strip()
            sample_size = int(sample_input) if sample_input else 25
            
            if sample_size < 11:
                print("Warning: Sample size must be at least 11 to include all labeled anomalies.")
                print("Using minimum size (11).")
                sample_size = 11
                
        except ValueError:
            print("Invalid input. Using default sample size of 25.")
            sample_size = 25
        
        # Test mode
        try:
            mode_input = input("Test mode - 1: Single config (fastest), 2: Multiple configs: ").strip()
            single_config = mode_input != "2"  # Default to single config
        except ValueError:
            print("Invalid input. Using single configuration mode.")
            single_config = True
        
        # Custom config
        custom_config = None
        if single_config:
            try:
                use_custom = input("Use custom configuration? (y/n, default: n): ").strip().lower()
                if use_custom == 'y':
                    bs_input = input("Enter batch size (default: 5): ").strip()
                    rf_input = input("Enter reset frequency (number or 'none', default: none): ").strip().lower()
                    
                    bs = int(bs_input) if bs_input else 5
                    rf = None if not rf_input or rf_input == 'none' else int(rf_input)
                    
                    custom_config = {'batch_size': bs, 'reset_frequency': rf}
            except ValueError:
                print("Invalid input. Using default configuration.")
                custom_config = None
        
        # Initialize analyzer
        analyzer = QuickQwenScalabilityAnalyzer(output_dir, api_key)
        
        # Run quick test
        print(f"\nRunning quick test with {sample_size} samples...")
        print("This will take some time. Progress will be logged.")
        
        try:
            # Time the analysis
            start_time = time.time()
            
            # Run the quick test
            results_df = analyzer.run_quicktest(
                n_samples=sample_size,
                single_config=single_config,
                custom_config=custom_config
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            minutes, seconds = divmod(total_time, 60)
            
            print(f"\nQuick test completed in {int(minutes)}m {int(seconds)}s!")
            print(f"Report available at: {os.path.join(output_dir, 'qwen_quicktest_report.html')}")
            
        except Exception as e:
            print(f"\nError during quick test: {str(e)}")
            traceback.print_exc()
            return
            
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
