import os
import pandas as pd
import numpy as np
import json
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import time
import traceback
import base64
import jinja2
from io import BytesIO
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

# Import our modular components
from llm.core.data_manager import CompressorDataManager
from llm.core.llm_predictor import LLMPredictor
from llm.prompts.expert_prompts import definitive_prompt

# Set up colors for better visualization
colors = {
    'true_positive': 'green',
    'false_positive': 'orange',
    'false_negative': 'red',
    'normal': 'blue'
}

def plot_anomaly_data(df, anomalies, false_positives, cols=None, window=24):
    """
    Plot the data with anomaly markers
    """
    if cols is None:
        cols = ['Temperature', 'Vibration', 'Current', 'Pressure']
    
    # Set up the plot
    fig, axes = plt.subplots(len(cols), 1, figsize=(15, 4 * len(cols)), sharex=True)
    
    for i, col in enumerate(cols):
        # Plot the raw data
        axes[i].plot(df['DateTime'], df[col], color='gray', alpha=0.7, label=col)
        
        # Highlight true anomalies
        for _, anomaly in anomalies.iterrows():
            anomaly_date = anomaly['DateTime']
            start_idx = max(0, df[df['DateTime'] == anomaly_date].index[0] - window//2)
            end_idx = min(len(df), df[df['DateTime'] == anomaly_date].index[0] + window//2)
            
            # Extract the corresponding data range
            subset = df.iloc[start_idx:end_idx]
            axes[i].plot(subset['DateTime'], subset[col], color=colors['true_positive'], 
                        linewidth=2, label=f"Anomaly: {anomaly['Notes']}")
        
        # Highlight false positives
        for _, fp in false_positives.iterrows():
            fp_date = fp['DateTime']
            start_idx = max(0, df[df['DateTime'] == fp_date].index[0] - window//2)
            end_idx = min(len(df), df[df['DateTime'] == fp_date].index[0] + window//2)
            
            # Extract the corresponding data range
            subset = df.iloc[start_idx:end_idx]
            axes[i].plot(subset['DateTime'], subset[col], color=colors['false_positive'], 
                        linewidth=2, label=f"False Positive: {fp['Notes']}")
        
        axes[i].set_title(f"{col} Over Time")
        axes[i].set_ylabel(col)
        axes[i].grid(True)
    
    # Set the x-axis label on the bottom subplot
    axes[-1].set_xlabel('Date and Time')
    
    # Adjust layout
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    return fig, axes

def format_elapsed_time(seconds):
    """Format elapsed time in a human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} minutes {seconds:.1f} seconds"

def analyze_anomalies(llm_predictor, data_manager, time_window=24, sample_size=5):
    """
    Use Groq LLM to analyze anomalies in the compressor data
    
    Args:
        llm_predictor: The LLM predictor instance
        data_manager: The data manager instance
        time_window: Time window size in hours
        sample_size: Number of cases to analyze (default: 5)
    """
    # Track start time
    start_time = time.time()
    print(f"Starting anomaly analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up results directory
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get labeled events for analysis
    anomalies_df, false_positives_df = data_manager.get_labeled_events()
    
    # Combine all events we want to analyze
    all_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly', 'event_type']],
        false_positives_df[['DateTime', 'Notes', 'Anomaly', 'event_type']]
    ])
    
    # Sort by datetime
    all_events = all_events.sort_values('DateTime')
    
    # Use the specified sample size
    analysis_events = all_events.iloc[:sample_size]
    print(f"Analyzing {len(analysis_events)} events from the dataset")
    
    # Prepare to store results and backup info
    results = []
    results_file = os.path.join(results_dir, 'llm_analysis_results.json')
    results_csv = os.path.join(results_dir, 'llm_analysis_results.csv')
    backup_dir = os.path.join(results_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # For each event, analyze the surrounding data
    for idx, event in analysis_events.iterrows():
        event_time = event['DateTime']
        event_type = event['event_type']
        actual_note = event['Notes']
        
        # Print timing information
        elapsed = time.time() - start_time
        print(f"\nAnalyzing event {idx+1}/{len(analysis_events)} at {event_time} - Elapsed: {format_elapsed_time(elapsed)}")
        print(f"Event type: {event_type}: {actual_note}")
        
        query_start = time.time()
        print(f"Sending query to LLM at {datetime.now().strftime('%H:%M:%S')}...")
        
        # Get data for time window around the event
        data_window = data_manager.get_data_window(event_time, time_window)
        
        # Get prediction from LLM
        parsed_response = llm_predictor.predict(
            event_time, 
            data_window['compressor_str'], 
            data_window['weather_str']
        )
        
        # Add timing for LLM response
        query_elapsed = time.time() - query_start
        print(f"LLM response received in {format_elapsed_time(query_elapsed)}")
        
        # Print the response immediately for the user
        print(f"\nLLM RESPONSE:")
        print(f"Classification: {parsed_response['classification']}")
        if parsed_response['type']:
            print(f"Type: {parsed_response['type']}")
        print(f"Confidence: {parsed_response['confidence']}")
        print(f"Key Indicators: {parsed_response['key_indicators']}")
        print(f"Recommendation: {parsed_response['recommendation']}")
        
        # Save result
        results.append({
            'datetime': event_time,
            'actual_type': event_type,
            'actual_note': actual_note,
            'llm_response': json.dumps(parsed_response),  # Store serialized version
            'parsed_response': parsed_response,
            'data_window': {
                'compressor': data_window['compressor_data'].to_dict('records'),
                'weather': data_window['weather_data'].to_dict('records')
            }
        })
        
        # Create a backup every 5 cases
        if (idx + 1) % 5 == 0 or idx == len(analysis_events) - 1:
            print(f"Creating backup after {idx+1} cases...")
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f'analysis_results_backup_{timestamp}_{idx+1}cases.json')
            
            # Save backup JSON
            with open(backup_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save main results file
            with open(results_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV version
            save_results_to_csv(results, results_csv)
            
            print(f"Backup created: {backup_file}")
    
    # Save results to file in the results directory
    results_file = os.path.join(results_dir, 'llm_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, default=str, indent=2)
    
    # Calculate and print total time
    total_time = time.time() - start_time
    print(f"\nCompleted analysis of {len(results)} events in {format_elapsed_time(total_time)}")
    print(f"Results saved to '{results_file}'")
    
    return results

def run_validation(llm_predictor, data_manager, results_dir, sample_size=100, continue_from_backup=False, preloaded_results=None, start_idx=0):
    """
    Run validation on anomalies with unified approach - optimized for speed
    
    Args:
        llm_predictor: LLM predictor instance
        data_manager: Data manager instance
        results_dir: Directory for saving results
        sample_size: Number of cases to analyze
        continue_from_backup: Whether to load from a backup file
        preloaded_results: Optional pre-loaded results (if provided, continue_from_backup is ignored)
        start_idx: Starting index for preloaded results
    """
    # Track start time
    start_time = time.time()
    print(f"\n========== RUNNING VALIDATION WITH DEFINITIVE PROMPT ==========")
    print(f"Starting validation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Target sample size: {sample_size} total cases")
    
    # 1. Get all labeled events
    anomalies_df, false_positives_df = data_manager.get_labeled_events()
    
    # 2. Combine labeled events
    all_labeled_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'event_type', 'actual_classification']],
        false_positives_df[['DateTime', 'Notes', 'event_type', 'actual_classification']]
    ])
    
    print(f"Found {len(all_labeled_events)} labeled events ({sum(all_labeled_events['event_type'] == 'anomaly')} anomalies, "
          f"{sum(all_labeled_events['event_type'] == 'false_positive')} false positives)")
    
    # 3. Determine how many normal cases to add - FIXED to respect target sample size
    labeled_count = len(all_labeled_events)
    normal_cases_count = sample_size - labeled_count  # Changed to use full sample_size
    normal_cases_count = max(normal_cases_count, 0)  # Ensure non-negative
    
    print(f"Will select {normal_cases_count} normal cases to reach target of {sample_size} total cases")
    
    # 4. Select normal cases if needed
    if normal_cases_count > 0:
        print(f"Selecting {normal_cases_count} normal cases...")
        normal_cases = data_manager.select_normal_cases(
            all_labeled_events['DateTime'].tolist(),
            n_samples=normal_cases_count
        )
        
        # Combine with labeled events
        test_cases = pd.concat([
            all_labeled_events,
            normal_cases[['DateTime', 'Notes', 'event_type', 'actual_classification']]
        ])
    else:
        test_cases = all_labeled_events
    
    # Important: Sort by DateTime to maintain chronological order
    test_cases = test_cases.sort_values('DateTime')
    
    # Verify we have the correct number of cases
    print(f"\nPrepared validation set with {len(test_cases)} cases:")
    print(f"- {sum(test_cases['event_type'] == 'anomaly')} anomalies")
    print(f"- {sum(test_cases['event_type'] == 'false_positive')} false positives")
    print(f"- {sum(test_cases['event_type'] == 'normal')} normal cases")
    
    # REMOVED: The code that was limiting test_cases to 128 cases
    
    # Add verification of target vs actual cases
    if len(test_cases) < sample_size:
        print(f"\n⚠️ Warning: Could only prepare {len(test_cases)} cases out of requested {sample_size}")
        print("This is the maximum number of cases available with current data")
        proceed = input("Would you like to proceed with available cases? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborting validation.")
            return [], {}
    
    # 6. Setup results tracking and checkpoint files
    checkpoint_file = os.path.join(results_dir, 'validation_checkpoint.json')
    predictions_csv = os.path.join(results_dir, 'prediction_results.csv')
    backup_dir = os.path.join(results_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # If preloaded results are provided, use them directly
    results = []
    start_idx = 0
    
    if preloaded_results is not None:
        results = preloaded_results
        start_idx = len(results)
        print(f"Using preloaded results with {start_idx} cases already processed")
    elif continue_from_backup:
        results, backup_start_idx = load_analysis_from_backup(backup_dir)
        if results is not None:
            start_idx = backup_start_idx
        else:
            print("No backup loaded. Starting from beginning.")
            
    # If not continuing from backup or no backup loaded, check for checkpoint
    elif os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} cases already processed")
        except:
            print("Checkpoint corrupted, starting from beginning")
    
    # Add timing information to the setup
    setup_time = time.time() - start_time
    print(f"Setup completed in {format_elapsed_time(setup_time)}")
    
    # Add diagnostic logging for checkpoint resumption
    if start_idx > 0:
        print(f"\n=== RESUMING FROM CASE #{start_idx+1} ===")
        print(f"Last processed case was at: {results[-1]['datetime']}")
        
        # FIXED: Changed the check to use sample_size instead of len(test_cases)
        if start_idx >= sample_size:
            print(f"\n✅ ALL CASES ALREADY PROCESSED!")
            print(f"The backup contains {start_idx} cases, which is equal to or more than the requested {sample_size} cases.")
            print(f"No more cases to process. Calculating final metrics...")
            
            metrics = calculate_metrics(results)
            with open(os.path.join(results_dir, 'prediction_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\n========== VALIDATION RESULTS ==========")
            print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"Anomaly precision: {metrics['anomaly_precision']:.2%}")
            print(f"Anomaly recall: {metrics['anomaly_recall']:.2%}")
            print(f"Anomaly F1 score: {metrics['anomaly_f1']:.2%}")
            print(f"Normal value accuracy: {metrics['specificity']:.2%}")
            
            return results, metrics
        else:
            remaining = sample_size - start_idx
            print(f"Will process {remaining} more cases to reach target of {sample_size}")
            print(f"Next case to process will be: {test_cases.iloc[start_idx]['DateTime']}")
    
    # Add memory management - clear caches periodically to avoid memory issues
    memory_safe_interval = 25  # Clear caches every 25 cases

    # Token limit detection - track request failures
    consecutive_token_failures = 0
    max_token_failures = 3  # After this many failures, assume token limit reached

    # 8. Process cases in batches for better efficiency
    try:
        # Ottimizzazione: aumentare dimensione batch per ridurre overhead
        batch_size = 8  # Aumentato da 5 a 8 per efficienza
        remaining_cases = len(test_cases) - start_idx
        validation_start = time.time()
        
        # Calculate estimated time based on progress
        def estimate_remaining_time(processed, total, elapsed):
            if processed == 0:
                return "Unknown"
            avg_time_per_case = elapsed / processed
            remaining = avg_time_per_case * (total - processed)
            return format_elapsed_time(remaining)
        
        # Process in batches
        for batch_start in range(start_idx, len(test_cases), batch_size):
            batch_start_time = time.time()
            batch_end = min(batch_start + batch_size, len(test_cases))
            
            # Print progress with timing information
            elapsed = time.time() - validation_start
            current_batch_num = batch_start//batch_size + 1
            total_batches = (remaining_cases + batch_size - 1)//batch_size
            print(f"\n=== PROCESSING BATCH {current_batch_num}/{total_batches} (CASES {batch_start+1}-{batch_end}) ===")
            print(f"Elapsed time: {format_elapsed_time(elapsed)}")
            
            if batch_start > start_idx:
                est_remaining = estimate_remaining_time(batch_start - start_idx, remaining_cases, elapsed)
                print(f"Estimated remaining time: {est_remaining}")
            
            # Ottimizzazione: caricare i dati in parallelo
            print(f"Pre-loading data for all cases in batch {current_batch_num} in parallel...")
            
            # Usa un thread pool per caricare i dati in parallelo
            with ThreadPoolExecutor(max_workers=min(8, batch_end - batch_start)) as executor:
                # Submit all data loading tasks
                futures = {
                    executor.submit(
                        data_manager.get_data_window, 
                        pd.Timestamp(test_cases.iloc[idx]['DateTime']), 
                        window_hours=6  # Ridotta ulteriormente la finestra temporale (da 12 a 6)
                    ): idx 
                    for idx in range(batch_start, batch_end)
                }
                
                # Wait for all to complete
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # Questo garantisce che l'eccezione venga sollevata se presente
                        print(f"  ✓ Pre-loaded data for case {idx+1}")
                    except Exception as e:
                        print(f"  ✗ Error pre-loading data for case {idx+1}: {e}")
            
            # Ora processa ogni caso nel batch con timeout più brevi
            batch_results = []
            
            # Ottimizzazione: Clear cache ogni 50 casi invece di 25 per ridurre overhead
            cases_processed = len(results)
            if cases_processed > 0 and cases_processed % 50 == 0:
                print(f"Clearing prediction cache after {cases_processed} cases to free memory...")
                if hasattr(llm_predictor, 'prediction_cache'):
                    cache_size = len(llm_predictor.prediction_cache)
                    # Salva la cache prima di cancellarla
                    if hasattr(llm_predictor, '_save_cache'):
                        llm_predictor._save_cache()
                    
                    # Mantieni solo le ultime 50 predizioni (le più recenti)
                    recent_keys = list(llm_predictor.prediction_cache.keys())[-50:]
                    recent_cache = {k: llm_predictor.prediction_cache[k] for k in recent_keys}
                    llm_predictor.prediction_cache = recent_cache
                    print(f"Optimized cache: kept {len(recent_cache)} recent entries, cleared {cache_size - len(recent_cache)} older entries")
                
                # Clear data manager cache if it exists
                if hasattr(data_manager, '_data_window_cache'):
                    cache_size = len(data_manager._data_window_cache)
                    # Mantieni solo le ultime 20 finestre di dati
                    recent_keys = list(data_manager._data_window_cache.keys())[-20:]
                    recent_cache = {k: data_manager._data_window_cache[k] for k in recent_keys}
                    data_manager._data_window_cache = recent_cache
                    print(f"Optimized data cache: kept {len(recent_cache)} recent entries, cleared {cache_size - len(recent_cache)} older entries")
                
                # Forza garbage collection
                import gc
                gc.collect()
                print("Forced garbage collection to free memory")
            
            # Process each case in the batch
            special_watch_cases = [127, 128, 129] # Cases 128, 129, 130
            for idx in range(batch_start, batch_end):
                case = test_cases.iloc[idx]
                event_time = pd.Timestamp(case['DateTime'])
                event_type = case['event_type']
                actual_class = case['actual_classification']
                
                # Extra logging for these special cases
                special_case = idx in special_watch_cases
                if special_case:
                    print(f"\n!!! SPECIAL ATTENTION CASE #{idx+1} !!!")
                    print(f"Processing case #{idx+1} with timestamp {event_time}")
                    print(f"Event type: {event_type}, Classification: {actual_class}")
                else:
                    print(f"\nProcessing case {idx+1}/{len(test_cases)}: {event_time} ({event_type})")
                
                try:
                    # Get data around the event (will use cache)
                    data_window = data_manager.get_data_window(event_time, window_hours=12)
                    
                    # Get prediction from LLM
                    prediction_start_time = time.time()
                    
                    # Make prediction with explicit error handling
                    try:
                        prediction = llm_predictor.predict(
                            event_time,
                            data_window['compressor_str'],
                            data_window['weather_str']
                        )
                        # If successful, reset token failure counter
                        consecutive_token_failures = 0
                        
                    except Exception as prediction_error:
                        error_str = str(prediction_error).lower()
                        print(f"  LLM prediction failed: {error_str}")
                        
                        # Check for token limit errors
                        if any(term in error_str for term in ["token", "limit", "quota", "capacity", "exceeded"]):
                            consecutive_token_failures += 1
                            print(f"⚠️ Possible token limit error detected ({consecutive_token_failures}/{max_token_failures})")
                            
                            # If we've hit the threshold, create emergency backup and stop
                            if consecutive_token_failures >= max_token_failures:
                                print("\n🛑 TOKEN LIMIT REACHED! Creating emergency backup and stopping analysis.")
                                emergency_file = os.path.join(backup_dir, f'token_limit_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{len(results)}cases.json')
                                
                                with open(emergency_file, 'w') as f:
                                    json.dump(results, f, default=str, indent=2)
                                print(f"Emergency backup saved: {emergency_file}")
                                print("Please resume your analysis later using the 'Resume from backup' option.")
                                return results, calculate_metrics(results)
                        
                        # Use fallback prediction
                        prediction = {
                            'classification': 'NORMAL VALUE',  # Default to normal as safer option
                            'type': '',
                            'confidence': 'low',
                            'key_indicators': 'Prediction failed',
                            'recommendation': 'Check system manually due to prediction failure'
                        }
                    
                    prediction_end_time = time.time()
                    print(f"  Prediction completed in {prediction_end_time - prediction_start_time:.2f} seconds")
                    
                    # Determine if prediction is correct
                    is_correct = prediction['classification'] == actual_class
                    
                    # Create result
                    result = {
                        'datetime': event_time.isoformat(),
                        'actual_type': event_type,
                        'actual_classification': actual_class,
                        'predicted_classification': prediction['classification'],
                        'predicted_type': prediction.get('type', 'N/A'),
                        'confidence': prediction.get('confidence', 'high'),
                        'key_indicators': prediction.get('key_indicators', ''),
                        'recommendation': prediction.get('recommendation', ''),
                        'is_correct': is_correct,
                        'temperature': data_window['exact_readings'].get('temperature'),
                        'vibration': data_window['exact_readings'].get('vibration'),
                        'pressure': data_window['exact_readings'].get('pressure'),
                        'current': data_window['exact_readings'].get('current')
                    }
                    
                    batch_results.append(result)
                    
                    # Show results
                    print(f"  Prediction: {prediction['classification']} (Actual: {actual_class})")
                    print(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}")
                    
                    # Add extra pause if this is a special attention case
                    if special_case:
                        print(f"Completed special attention case #{idx+1} successfully.")
                        # Add a small pause to ensure API cooldown
                        time.sleep(3)
                        
                except Exception as e:
                    print(f"  Error processing case {idx+1}: {str(e)}")
                    traceback.print_exc()
                    
                    # Create an emergency backup of results so far if error happens
                    emergency_file = os.path.join(backup_dir, f'emergency_backup_before_case_{idx+1}.json')
                    with open(emergency_file, 'w') as f:
                        json.dump(results, f, default=str, indent=2)
                    print(f"Created emergency backup at case #{idx+1}: {emergency_file}")
                    
                    # For special cases, retry with extreme caution
                    if special_case and "429" not in str(e):
                        print(f"Attempting careful retry of special case #{idx+1}...")
                        time.sleep(10)  # Longer pause before retry
                        try:
                            # Simplified retry with basic prediction
                            simplified_prediction = {
                                'classification': 'NORMAL VALUE',  # Default to normal as safer option
                                'type': '',
                                'confidence': 'low',
                                'key_indicators': 'Special case retry',
                                'recommendation': 'Check system manually due to retry failure'
                            }
                            
                            # Create a simplified result
                            result = {
                                'datetime': event_time.isoformat(),
                                'actual_type': event_type,
                                'actual_classification': actual_class,
                                'predicted_classification': simplified_prediction['classification'],
                                'predicted_type': simplified_prediction.get('type', 'N/A'),
                                'confidence': simplified_prediction['confidence'],
                                'key_indicators': simplified_prediction['key_indicators'],
                                'recommendation': simplified_prediction['recommendation'],
                                'is_correct': simplified_prediction['classification'] == actual_class,
                                'temperature': data_window['exact_readings'].get('temperature'),
                                'vibration': data_window['exact_readings'].get('vibration'),
                                'pressure': data_window['exact_readings'].get('pressure'),
                                'current': data_window['exact_readings'].get('current')
                            }
                            
                            batch_results.append(result)
                            print(f"  Simplified retry prediction completed")
                            
                        except:
                            print(f"Retry failed for case #{idx+1}. Continuing with next case.")
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Show current metrics every batch
            if len(results) >= 10:  # Only show metrics when we have enough data
                interim_metrics = print_current_metrics(results, "INTERIM METRICS")
            
            # Ottimizzazione: salvataggio incrementale più efficiente
            cases_processed = len(results)
            if cases_processed % 10 == 0 or idx == len(test_cases) - 1:
                # Usa un thread separato per il salvataggio del checkpoint per non bloccare l'esecuzione
                def save_checkpoint():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = os.path.join(backup_dir, f'validation_backup_{timestamp}_{cases_processed}cases.json')
                    
                    # Save timestamped backup
                    with open(backup_file, 'w') as f:
                        json.dump(results, f, default=str, indent=2)
                        
                    # Also save regular checkpoint file
                    with open(checkpoint_file, 'w') as f:
                        json.dump(results, f, default=str, indent=2)
                    
                    # Save CSV version
                    save_results_to_csv(results, predictions_csv)
                    
                    print(f"Checkpoint and backup created successfully")
                
                # Esegui il salvataggio in un thread separato
                threading.Thread(target=save_checkpoint).start()
            
            # Ottimizzazione: riduzione delle pause tra batch quando non necessarie
            if batch_end < len(test_cases):
                if any(special_watch_cases) in range(batch_start - batch_size, batch_end + batch_size):
                    print("Special cases approaching or processed - taking extra pause between batches...")
                    time.sleep(5)  # Ridotto da 10 a 5 secondi
                else:
                    # Calcola il tempo atteso in base al rate limit corrente
                    if hasattr(llm_predictor, 'current_interval'):
                        expected_wait = max(0.5, llm_predictor.current_interval * 0.5)  # Metà del current_interval
                        print(f"Short optimized pause between batches ({expected_wait:.1f}s)...")
                        time.sleep(expected_wait)
                    else:
                        print("Short pause between batches...")
                        time.sleep(1.0)  # Ridotto da 2.0 a 1.0 secondi
        
        # Calculate and print metrics before finally block
        metrics = calculate_metrics(results)
            
    except Exception as e:
        # Add error logging
        print(f"\n❌ ERROR during validation: {str(e)}")
        traceback.print_exc()
        # Ensure metrics are calculated even if there's an exception
        metrics = calculate_metrics(results) if results else {}
        
    finally:
        # Calculate and print total time
        total_time = time.time() - start_time
        print(f"\n========== VALIDATION COMPLETED ==========")
        print(f"Total validation time: {format_elapsed_time(total_time)}")
        print(f"Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
        print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
        
        # Save final results
        if results:
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            save_results_to_csv(results, predictions_csv)
            
            # Ensure metrics are calculated if not already done
            if 'metrics' not in locals() or not metrics:
                metrics = calculate_metrics(results)
                
            # Save metrics
            with open(os.path.join(results_dir, 'prediction_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\n========== VALIDATION RESULTS ==========")
            print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"Anomaly precision: {metrics['anomaly_precision']:.2%}")
            print(f"Anomaly recall: {metrics['anomaly_recall']:.2%}")
            print(f"Anomaly F1 score: {metrics['anomaly_f1']:.2%}")
            print(f"Normal value accuracy: {metrics['specificity']:.2%}")
    
    # Return statement AFTER the try/finally block
    # Ensure metrics exist even if we exit early
    if 'metrics' not in locals() or not metrics:
        metrics = calculate_metrics(results) if results else {}
    return results, metrics

def save_results_to_csv(results, csv_file):
    """
    Save results to a CSV file for easy import into visualization tools
    """
    # Create a DataFrame with essential columns for analysis
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    return df

def calculate_metrics(results):
    """
    Calculate comprehensive metrics from results
    """
    if not results:
        return {}
    
    # Extract classifications
    y_true = [r['actual_classification'] for r in results]
    y_pred = [r['predicted_classification'] for r in results]
    
    # Calculate base metrics
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 'ANOMALY' and y_pred[i] == 'ANOMALY')
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 'NORMAL VALUE' and y_pred[i] == 'ANOMALY')
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 'NORMAL VALUE' and y_pred[i] == 'NORMAL VALUE')
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 'ANOMALY' and y_pred[i] == 'NORMAL VALUE')
    
    # Calculate derived metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Create confusion matrix
    cm = [[tn, fp], [fn, tp]]  # [[TN, FP], [FN, TP]]
    
    return {
        'overall_accuracy': accuracy,
        'anomaly_precision': precision,
        'anomaly_recall': recall,
        'anomaly_f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'total_cases': len(results),
        'execution_time': datetime.now().isoformat()
    }

def print_current_metrics(results, title="CURRENT METRICS"):
    """
    Print current metrics in a standardized format during execution
    """
    if not results:
        print("\nNo results yet to calculate metrics.")
        return
    
    metrics = calculate_metrics(results)
    
    print(f"\n========== {title} ==========")
    print(f"Cases processed: {metrics['total_cases']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Anomaly precision: {metrics['anomaly_precision']:.2%}")
    print(f"Anomaly recall: {metrics['anomaly_recall']:.2%}")
    print(f"Anomaly F1 score: {metrics['anomaly_f1']:.2%}")
    print(f"Normal cases accuracy: {metrics['specificity']:.2%}")
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"            | Pred: NORMAL | Pred: ANOMALY |")
    print(f"True NORMAL | {cm[0][0]:12d} | {cm[0][1]:13d} |")
    print(f"True ANOMALY| {cm[1][0]:12d} | {cm[1][1]:13d} |")
    
    return metrics

def analyze_false_positives(llm_predictor, data_manager, results_dir):
    """
    Run targeted analysis on known false positive cases - optimized for speed
    """
    # Track start time
    start_time = time.time()
    print(f"\n========== RUNNING FALSE POSITIVE ANALYSIS ==========")
    print(f"Starting false positive analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    # Get all labeled false positives
    _, false_positives = data_manager.get_labeled_events()
    
    print(f"Found {len(false_positives)} labeled false positives")
    results = []
    
    # Setup backup directory
    backup_dir = os.path.join(results_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'false_positive_analysis_results.json')
    csv_file = os.path.join(results_dir, 'false_positive_analysis_results.csv')
    
    # Process in batches for better efficiency
    batch_size = 3
    total_processed = 0
    
    for batch_start in range(0, len(false_positives), batch_size):
        batch_end = min(batch_start + batch_size, len(false_positives))
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(false_positives) + batch_size - 1)//batch_size}")
        
        # Process each case in batch
        batch_results = []
        for idx in range(batch_start, batch_end):
            case = false_positives.iloc[idx]
            event_time = pd.Timestamp(case['DateTime'])
            actual_note = case['Notes']
            
            print(f"\nProcessing false positive case {idx+1}/{len(false_positives)}: {event_time}")
            
            # Get data window
            data_window = data_manager.get_data_window(event_time, window_hours=12)
            
            # Get prediction
            prediction = llm_predictor.predict(
                event_time,
                data_window['compressor_str'],
                data_window['weather_str']
            )
            
            # Add to results
            is_correct = prediction['classification'] == 'NORMAL VALUE'
            results.append({
                'datetime': event_time.isoformat(),
                'actual_type': 'false_positive',
                'actual_classification': 'NORMAL VALUE',
                'actual_note': actual_note,
                'predicted_classification': prediction['classification'],
                'predicted_type': prediction.get('type', ''),
                'confidence': prediction.get('confidence', ''),
                'key_indicators': prediction.get('key_indicators', ''),
                'recommendation': prediction.get('recommendation', ''),
                'is_correct': is_correct,
                'temperature': data_window['exact_readings'].get('temperature'),
                'vibration': data_window['exact_readings'].get('vibration'),
                'pressure': data_window['exact_readings'].get('pressure'),
                'current': data_window['exact_readings'].get('current')
            })
            
            # Print result
            print(f"  LLM Classification: {prediction['classification']} (Should be: NORMAL VALUE)")
            if is_correct:
                print(f"✓ Correct classification")
            else:
                print(f"✗ Incorrect classification")
        
        # Add batch results to overall results
        results.extend(batch_results)
        total_processed += len(batch_results)
        
        # Show metrics after each batch if we have enough data
        if len(results) >= 5:
            print_current_metrics(results, "FALSE POSITIVE ANALYSIS METRICS")
        
        # Create backup every 5 cases
        if total_processed % 5 == 0 or idx == len(false_positives) - 1:
            print(f"Creating backup after {total_processed} false positive cases...")
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f'fp_analysis_backup_{timestamp}_{total_processed}cases.json')
            
            # Save timestamped backup
            with open(backup_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Also save main results file
            with open(results_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV version
            save_results_to_csv(results, csv_file)
            
            print(f"Backup created: {backup_file}")
        
        # Save intermediate results after each batch
        results_file = os.path.join(results_dir, 'false_positive_analysis_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        # Also save as CSV
        csv_file = os.path.join(results_dir, 'false_positive_analysis_results.csv')
        save_results_to_csv(results, csv_file)
        
        # Brief pause between batches
        if batch_end < len(false_positives):
            time.sleep(2)
    
    # Print summary with improved metrics
    metrics = calculate_metrics(results)
    print("\n========== FALSE POSITIVE ANALYSIS RESULTS ==========")
    print(f"Cases analyzed: {metrics['total_cases']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"False positive detection accuracy: {metrics['specificity']:.2%}")
    
    # Calculate and print total time
    total_time = time.time() - start_time
    print(f"\nCompleted analysis of {len(results)} false positive cases in {format_elapsed_time(total_time)}")
    print(f"Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
    print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results saved to '{results_file}'")
    
    return results

def load_analysis_from_backup(backup_dir):
    """
    Load analysis results from a backup JSON file
    
    Returns:
        Tuple containing (results list, start_idx for continuing analysis)
    """
    # List all json backups
    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
    
    if not backup_files:
        print("No backup files found.")
        return None, 0
    
    # Sort backups by timestamp (newest first)
    backup_files.sort(reverse=True)
    
    # Display available backups
    print("\nAvailable backup files:")
    for i, file in enumerate(backup_files):
        # Extract number of cases from filename if available
        cases_count = "Unknown"
        if "_cases" in file:
            try:
                cases_count = file.split("_")[-1].replace("cases.json", "")
            except:
                pass
        
        # Determine if this is a full analysis backup
        analysis_type = "Full Database" if "full_analysis" in file else "Standard"
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(backup_dir, file)))
        print(f"{i+1}. {file} - {cases_count} cases - {analysis_type} - {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Let user select backup file
    choice = input("\nEnter number of backup to load (or 0 to cancel): ")
    try:
        choice = int(choice)
        if choice == 0:
            return None, 0
        
        selected_file = os.path.join(backup_dir, backup_files[choice-1])
        print(f"Loading backup from: {selected_file}")
        
        # Load the backup
        with open(selected_file, 'r') as f:
            results = json.load(f)
            
        start_idx = len(results)
        print(f"Loaded {start_idx} results from backup")
        return results, start_idx
            
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {str(e)}")
        return None, 0

def run_full_database_analysis(llm_predictor, data_manager, results_dir, continue_from_backup=False, preloaded_results=None, start_idx=0):
    """
    Run analysis on EVERY record in the database - all 2208 cases
    
    Args:
        llm_predictor: LLM predictor instance
        data_manager: Data manager instance
        results_dir: Directory for saving results
        continue_from_backup: Whether to load from a backup file
        preloaded_results: Optional pre-loaded results (if provided, continue_from_backup is ignored)
        start_idx: Starting index for preloaded results
    """
    # Track start time
    start_time = time.time()
    print(f"\n========== RUNNING FULL DATABASE ANALYSIS ==========")
    print(f"Starting full database analysis at {datetime.now().strftime('%H:%M:%S')}")
    
    # Connect to database to get ALL records
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    
    # Setup checkpoint files and backup directory
    checkpoint_file = os.path.join(results_dir, 'full_analysis_checkpoint.json')
    predictions_csv = os.path.join(results_dir, 'full_analysis_results.csv')
    backup_dir = os.path.join(results_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Load data and get all timestamps from the database
    conn = sqlite3.connect(db_path)
    print("Loading all timestamps from database...")
    all_timestamps = pd.read_sql("SELECT DISTINCT DateTime FROM compressor_data_with_weather ORDER BY DateTime", conn)
    all_timestamps['DateTime'] = pd.to_datetime(all_timestamps['DateTime'])
    
    # Load labeled events to add correct classification information
    anomalies_df, false_positives_df = data_manager.get_labeled_events()
    
    # Convert to sets for faster lookup
    anomaly_dates = set(pd.to_datetime(anomalies_df['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S'))
    false_positive_dates = set(pd.to_datetime(false_positives_df['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"Preparing analysis for {len(all_timestamps)} total records")
    
    # Add classification column to all timestamps
    all_timestamps['actual_type'] = 'normal'
    all_timestamps['actual_classification'] = 'NORMAL VALUE'
    all_timestamps['Notes'] = 'Unlabeled datapoint'
    
    # Add correct labels for known anomalies and false positives
    for idx, row in all_timestamps.iterrows():
        date_str = row['DateTime'].strftime('%Y-%m-%d %H:%M:%S')
        if date_str in anomaly_dates:
            all_timestamps.loc[idx, 'actual_type'] = 'anomaly'
            all_timestamps.loc[idx, 'actual_classification'] = 'ANOMALY'
            # Find the specific note for this anomaly
            matching_row = anomalies_df[anomalies_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S') == date_str]
            if not matching_row.empty:
                all_timestamps.loc[idx, 'Notes'] = matching_row.iloc[0]['Notes']
        elif date_str in false_positive_dates:
            all_timestamps.loc[idx, 'actual_type'] = 'false_positive'
            all_timestamps.loc[idx, 'actual_classification'] = 'NORMAL VALUE'
            # Find the specific note for this false positive
            matching_row = false_positives_df[false_positives_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S') == date_str]
            if not matching_row.empty:
                all_timestamps.loc[idx, 'Notes'] = matching_row.iloc[0]['Notes']
    
    # Close connection after data is loaded
    conn.close()
    
    # Count statistics
    anomaly_count = sum(all_timestamps['actual_type'] == 'anomaly')
    fp_count = sum(all_timestamps['actual_type'] == 'false_positive')
    normal_count = sum(all_timestamps['actual_type'] == 'normal')
    
    print(f"Dataset prepared with {len(all_timestamps)} total records:")
    print(f"- {anomaly_count} labeled anomalies")
    print(f"- {fp_count} labeled false positives")
    print(f"- {normal_count} normal/unlabeled points")
    
    # If preloaded results are provided, use them directly
    results = []
    start_idx = 0
    
    if preloaded_results is not None:
        results = preloaded_results
        start_idx = len(results)
        print(f"Using preloaded results with {start_idx} cases already processed")
    elif continue_from_backup:
        results, backup_start_idx = load_analysis_from_backup(backup_dir)
        if results is not None:
            start_idx = backup_start_idx
        else:
            print("No backup loaded. Starting from beginning.")
            
    # If not continuing from backup or no backup loaded, check for checkpoint
    elif os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} cases already processed")
        except:
            print("Checkpoint corrupted, starting from beginning")
    
    # Setup timing and progress tracking
    setup_time = time.time() - start_time
    print(f"Setup completed in {format_elapsed_time(setup_time)}")
    
    # Add diagnostic logging for checkpoint resumption
    if start_idx > 0:
        print(f"\n=== RESUMING FROM CASE #{start_idx+1} ===")
        print(f"Last processed case was at: {results[-1]['datetime']}")
        
        # Check if all cases have already been processed
        if start_idx >= len(all_timestamps):
            print(f"\n✅ ALL CASES ALREADY PROCESSED!")
            print(f"The backup contains {start_idx} cases, which is equal to or more than the {len(all_timestamps)} cases in the analysis set.")
            print(f"No more cases to process. Calculating final metrics...")
            
            metrics = calculate_metrics(results)
            with open(os.path.join(results_dir, 'full_analysis_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\n========== FULL DATABASE ANALYSIS RESULTS ==========")
            print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"Anomaly precision: {metrics['anomaly_precision']:.2%}")
            print(f"Anomaly recall: {metrics['anomaly_recall']:.2%}")
            print(f"Anomaly F1 score: {metrics['anomaly_f1']:.2%}")
            print(f"Normal value accuracy: {metrics['specificity']:.2%}")
            
            return results, metrics
        else:
            print(f"Next case to process will be: {all_timestamps.iloc[start_idx]['DateTime']}")
    
    # Add memory management - clear caches periodically to avoid memory issues
    memory_safe_interval = 25  # Clear caches every 25 cases

    # Token limit detection - track request failures
    consecutive_token_failures = 0
    max_token_failures = 3  # After this many failures, assume token limit reached

    # Process in batches for efficiency
    try:
        # Define smaller batch size for the large dataset
        batch_size = 5  # Process 5 cases at a time
        remaining_cases = len(all_timestamps) - start_idx
        analysis_start = time.time()
        
        # Calculate estimated time based on progress
        def estimate_remaining_time(processed, total, elapsed):
            if processed == 0:
                return "Unknown"
            avg_time_per_case = elapsed / processed
            remaining = avg_time_per_case * (total - processed)
            return format_elapsed_time(remaining)
        
        # Process in batches
        for batch_start in range(start_idx, len(all_timestamps), batch_size):
            batch_start_time = time.time()
            batch_end = min(batch_start + batch_size, len(all_timestamps))
            
            # Print progress with timing information
            elapsed = time.time() - analysis_start
            current_batch_num = batch_start//batch_size + 1
            total_batches = (remaining_cases + batch_size - 1)//batch_size
            print(f"\n=== PROCESSING BATCH {current_batch_num}/{total_batches} (CASES {batch_start+1}-{batch_end}) ===")
            print(f"Elapsed time: {format_elapsed_time(elapsed)}")
            
            if batch_start > start_idx:
                est_remaining = estimate_remaining_time(batch_start - start_idx, remaining_cases, elapsed)
                print(f"Estimated remaining time: {est_remaining}")
                print(f"Estimated completion: {datetime.now() + timedelta(seconds=elapsed/max(1, batch_start - start_idx) * remaining_cases)}")
            
            # Pre-extract data windows for all cases in the batch to benefit from caching
            for idx in range(batch_start, batch_end):
                event_time = all_timestamps.iloc[idx]['DateTime']
                print(f"  Pre-loading data for case {idx+1}/{len(all_timestamps)}: {event_time}")
                data_manager.get_data_window(event_time, window_hours=12)  # Pre-load into cache with smaller window
            
            # Now process each case in the batch
            batch_results = []
            
            # Clear prediction cache periodically
            cases_processed = len(results)
            if cases_processed % memory_safe_interval == 0:
                print(f"Clearing prediction cache after {cases_processed} cases to free memory...")
                if hasattr(llm_predictor, 'prediction_cache'):
                    cache_size = len(llm_predictor.prediction_cache)
                    llm_predictor.prediction_cache.clear()
                    print(f"Cleared {cache_size} entries from prediction cache")
                
                # Also clear data manager cache
                if hasattr(data_manager, '_data_window_cache'):
                    cache_size = len(data_manager._data_window_cache)
                    data_manager._data_window_cache.clear()
                    print(f"Cleared {cache_size} entries from data window cache")
            
            for idx in range(batch_start, batch_end):
                case = all_timestamps.iloc[idx]
                event_time = case['DateTime']
                event_type = case['actual_type']
                actual_class = case['actual_classification']
                
                print(f"\nProcessing case {idx+1}/{len(all_timestamps)}: {event_time} ({event_type})")
                
                try:
                    # Get data around the event (will use cache)
                    data_window = data_manager.get_data_window(event_time, window_hours=12)
                    
                    # Get prediction from LLM
                    prediction_start_time = time.time()
                    
                    # Make prediction with explicit error handling
                    try:
                        prediction = llm_predictor.predict(
                            event_time,
                            data_window['compressor_str'],
                            data_window['weather_str']
                        )
                        # Reset token failure counter on success
                        consecutive_token_failures = 0
                        
                    except Exception as prediction_error:
                        error_str = str(prediction_error).lower()
                        print(f"  LLM prediction failed: {error_str}")
                        
                        # Check for token limit errors
                        if any(term in error_str for term in ["token", "limit", "quota", "capacity", "exceeded"]):
                            consecutive_token_failures += 1
                            print(f"⚠️ Possible token limit error detected ({consecutive_token_failures}/{max_token_failures})")
                            
                            # If we've hit the threshold, create emergency backup and stop
                            if consecutive_token_failures >= max_token_failures:
                                print("\n🛑 TOKEN LIMIT REACHED! Creating emergency backup and stopping analysis.")
                                emergency_file = os.path.join(backup_dir, f'token_limit_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{len(results)}cases.json')
                                
                                with open(emergency_file, 'w') as f:
                                    json.dump(results, f, default=str, indent=2)
                                print(f"Emergency backup saved: {emergency_file}")
                                print("Please resume your analysis later using the 'Resume from backup' option.")
                                return results, calculate_metrics(results)
                        
                        # Use fallback prediction
                        prediction = {
                            'classification': 'NORMAL VALUE',  # Default to normal as safer option
                            'type': '',
                            'confidence': 'low',
                            'key_indicators': 'Prediction failed',
                            'recommendation': 'Check system manually due to prediction failure'
                        }
                    
                    prediction_end_time = time.time()
                    print(f"  Prediction completed in {prediction_end_time - prediction_start_time:.2f} seconds")
                    
                    # Determine if prediction is correct
                    is_correct = prediction['classification'] == actual_class
                    
                    # Create result
                    result = {
                        'datetime': event_time.isoformat(),
                        'actual_type': event_type,
                        'actual_classification': actual_class,
                        'predicted_classification': prediction['classification'],
                        'predicted_type': prediction.get('type', 'N/A'),
                        'confidence': prediction.get('confidence', 'high'),
                        'key_indicators': prediction.get('key_indicators', ''),
                        'recommendation': prediction.get('recommendation', ''),
                        'is_correct': is_correct,
                        'temperature': data_window['exact_readings'].get('temperature'),
                        'vibration': data_window['exact_readings'].get('vibration'),
                        'pressure': data_window['exact_readings'].get('pressure'),
                        'current': data_window['exact_readings'].get('current')
                    }
                    
                    batch_results.append(result)
                    
                    # Show results
                    print(f"  Prediction: {prediction['classification']} (Actual: {actual_class})")
                    print(f"  {'✓ Correct' if is_correct else '✗ Incorrect'}")
                    
                except Exception as e:
                    print(f"  Error processing case {idx+1}: {str(e)}")
                    traceback.print_exc()
                    
                    # Create emergency backup if error happens
                    emergency_file = os.path.join(backup_dir, f'emergency_backup_before_case_{idx+1}.json')
                    with open(emergency_file, 'w') as f:
                        json.dump(results, f, default=str, indent=2)
                    print(f"Created emergency backup at case #{idx+1}: {emergency_file}")
            
            # Add batch results to overall results
            results.extend(batch_results)
            
            # Create reliable backups after every 5 cases
            cases_processed = len(results)
            if cases_processed % 5 == 0 or idx == len(all_timestamps) - 1:
                print(f"Creating backup after {cases_processed} total cases processed...")
                
                # Create timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(backup_dir, f'full_analysis_backup_{timestamp}_{cases_processed}cases.json')
                
                # Save timestamped backup
                with open(backup_file, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                
                # Also save regular checkpoint file
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                
                # Save CSV version
                save_results_to_csv(results, predictions_csv)
                
                print(f"Checkpoint and backup created successfully")
            
            # Also save checkpoint after each batch
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV version
            save_results_to_csv(results, predictions_csv)
            
            # Add timing for batch completion
            batch_time = time.time() - batch_start_time
            print(f"Batch completed in {format_elapsed_time(batch_time)}")
            
            # Show interim metrics
            if results:
                correct = sum(1 for r in results if r['is_correct'])
                accuracy = correct / len(results)
                print(f"\nInterim accuracy: {accuracy:.2%} ({correct}/{len(results)})")
                
            # Brief pause between batches to avoid rate limits
            if batch_end < len(all_timestamps):
                print("Short pause between batches...")
                time.sleep(2)
        
        # Calculate metrics at the end of successful processing
        metrics = calculate_metrics(results)
    
    except Exception as e:
        # Add error logging
        print(f"\n❌ ERROR during full database analysis: {str(e)}")
        traceback.print_exc()
        # Ensure metrics are calculated even if there's an exception
        metrics = calculate_metrics(results) if results else {}
    
    finally:
        # Calculate and print total time
        total_time = time.time() - start_time
        print(f"\n========== FULL DATABASE ANALYSIS COMPLETED ==========")
        print(f"Total analysis time: {format_elapsed_time(total_time)}")
        print(f"Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")
        print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
        
        save_results_to_csv(results, predictions_csv)
        
        # Save final results
        if results:
            with open(checkpoint_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            save_results_to_csv(results, predictions_csv)
            
            # Ensure metrics are calculated if not already done
            if 'metrics' not in locals() or not metrics:
                metrics = calculate_metrics(results)
            
            # Calculate and save metrics
            with open(os.path.join(results_dir, 'full_analysis_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print("\n========== FULL DATABASE ANALYSIS RESULTS ==========")
            print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
            print(f"Anomaly precision: {metrics['anomaly_precision']:.2%}")
            print(f"Anomaly recall: {metrics['anomaly_recall']:.2%}")
            print(f"Anomaly F1 score: {metrics['anomaly_f1']:.2%}")
            print(f"Normal value accuracy: {metrics['specificity']:.2%}")
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"            | Pred: NORMAL | Pred: ANOMALY |")
            print(f"True NORMAL | {cm[0][0]:12d} | {cm[0][1]:13d} |")
            print(f"True ANOMALY| {cm[1][0]:12d} | {cm[1][1]:13d} |")
    
    # Return statement AFTER the try/finally block
    # Ensure metrics exist even if we exit early
    if 'metrics' not in locals() or not metrics:
        metrics = calculate_metrics(results) if results else {}
    return results, metrics

def main():
    """
    Main function with simplified structure
    """
    main_start_time = time.time()
    print(f"Started at {datetime.now().strftime('%H:%M:%S')}")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        return
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    labeled_db_path = os.path.join(current_dir, "datasets", "compressor_data_2024_etichettato.db")
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize components
    data_manager = CompressorDataManager(db_path, labeled_db_path)
    
    # Initialize LLM predictor with llama model instead of qwen
    print("Initializing LLM predictor with llama-3.2-11b-vision-preview model...")
    llm_predictor = LLMPredictor(api_key, model="llama-3.2-11b-vision-preview")
    # Replace the prompt with our definitive_prompt and add format instructions
    
    def_prompt = definitive_prompt()
    
    # Add explicit format instructions to ensure proper output from Qwen model
    format_reminder = """
=== ISTRUZIONI DI FORMATO CRITICHE ===
DEVI RISPONDERE ESATTAMENTE IN QUESTO FORMATO:

CLASSIFICAZIONE: [ANOMALIA o VALORE NORMALE]
TIPO: [SOLO se ANOMALIA: guasto cuscinetti/surriscaldamento/calo di pressione/squilibrio motore/fluttuazione tensione]
CONFIDENZA: [alta/media/bassa]
INDICATORI_CHIAVE: [Elenca 2-3 letture specifiche con valori esatti]
RACCOMANDAZIONE: [1 frase concisa]

NON AGGIUNGERE SPIEGAZIONI O ALTRI TESTI. SOLO IL FORMATO STRUTTURATO MOSTRATO SOPRA.
"""
    
    if "FORMATO" not in def_prompt:
        def_prompt += format_reminder
    
    print(f"Using definitive prompt ({len(def_prompt)} chars)")
    llm_predictor.prompt = def_prompt
    
    # Load labeled events
    anomalies, false_positives = data_manager.get_labeled_events()
    
    # Load data directly (needed for plotting)
    conn = sqlite3.connect(db_path)
    print("Loading compressor data...")
    df_clean = pd.read_sql("SELECT * FROM compressor_data", conn)
    df_clean['DateTime'] = pd.to_datetime(df_clean['DateTime'])
    conn.close()
    print(f"Loaded data with {len(anomalies)} anomalies and {len(false_positives)} false positives")
    
    # Plot some of the anomalies with surrounding data
    print("Generating plots for anomalies...")
    anomalies_fig, axes = plot_anomaly_data(df_clean, anomalies, false_positives)
    anomalies_plot_path = os.path.join(results_dir, 'anomalies_plot.png')
    anomalies_fig.savefig(anomalies_plot_path)
    plt.close(anomalies_fig)
    print(f"Plot saved as '{anomalies_plot_path}'")
    
    # Use smallest effective window size and batch processing
    def analyze_anomalies_fast(llm_predictor, data_manager):
        """Optimized quick analysis"""
        return analyze_anomalies(llm_predictor, data_manager, time_window=12)
    
    # Get database statistics for complete analysis
    conn = sqlite3.connect(db_path)
    total_records = pd.read_sql("SELECT COUNT(*) as count FROM compressor_data_with_weather", conn).iloc[0]['count']
    conn.close()
    
    # Interactive menu
    try:
        while True:
            # Show elapsed time in menu
            elapsed = time.time() - main_start_time
            print(f"\nElapsed session time: {format_elapsed_time(elapsed)}")
            print("\nChoose an analysis option:")
            print("1. Quick analysis with 20 cases")
            print("2. Custom number of cases")
            print(f"3. Full database analysis (all {total_records} cases)")
            print("4. Resume from backup file")
            print("5. Exit")
            choice = input("Enter your choice (1-5): ")
            option_start_time = time.time()
            
            if choice == '1':
                # Quick analysis with 20 cases
                print("\nRunning quick analysis with 20 cases...")
                analysis_results = analyze_anomalies(llm_predictor, data_manager, time_window=12, sample_size=20)
                
                # Print metrics at the end
                if analysis_results:
                    metrics = print_current_metrics(analysis_results, "QUICK ANALYSIS RESULTS")
                
                # Save results
                prediction_results_file = os.path.join(results_dir, 'quick_analysis_results.csv')
                save_results_to_csv(analysis_results, prediction_results_file)
                print(f"Results saved as CSV for reuse: {prediction_results_file}")
                
                # Print total option time
                option_elapsed = time.time() - option_start_time
                print(f"\nQuick analysis completed in {format_elapsed_time(option_elapsed)}")
                
            elif choice == '2':
                # Custom number of cases
                print("\nRunning analysis with custom number of cases...")
                try:
                    sample_size = int(input("Enter number of cases to analyze: "))
                    if sample_size <= 0:
                        print("Please enter a positive number.")
                        continue
                    print(f"Running validation with {sample_size} cases...")
                    
                    # Add try-except around validation function call
                    try:
                        validation_results, validation_metrics = run_validation(
                            llm_predictor,
                            data_manager,
                            results_dir,
                            sample_size
                        )
                        
                        # Print full metrics at the end
                        if validation_results:
                            print_current_metrics(validation_results, "FINAL VALIDATION METRICS")
                    except Exception as validation_error:
                        print(f"\n❌ Error during validation: {validation_error}")
                        traceback.print_exc()
                    
                    # Print total option time
                    option_elapsed = time.time() - option_start_time
                    print(f"\nCustom analysis completed in {format_elapsed_time(option_elapsed)}")
                except ValueError:
                    print("Invalid number. Please enter a valid integer.")
                    
            elif choice == '3':
                # Full database analysis - CHANGED to use run_full_database_analysis
                print(f"\nRunning full database analysis on ALL {total_records} cases...")
                print("Warning: This will analyze EVERY record in the database and will take a very long time.")
                print(f"Estimated time: {total_records//30:.1f} hours (approx 30 cases per hour)")
                confirmation = input("Are you sure you want to proceed? (y/n): ")
                if confirmation.lower() == 'y':
                    # Use run_full_database_analysis instead of run_validation
                    try:
                        validation_results, validation_metrics = run_full_database_analysis(
                            llm_predictor,
                            data_manager,
                            results_dir
                        )
                    except Exception as analysis_error:
                        print(f"\n❌ Error during full database analysis: {analysis_error}")
                        traceback.print_exc()
                        
                    # Print total option time
                    option_elapsed = time.time() - option_start_time
                    print(f"\nFull database analysis completed in {format_elapsed_time(option_elapsed)}")
                    
            elif choice == '4':
                # Resume from backup - UPDATED to provide clear options
                print("\nResuming analysis from backup file...")
                
                # Define backup directory
                backup_dir = os.path.join(results_dir, 'backups')
                
                # Ask which type of analysis to resume
                print("\nWhich type of analysis do you want to resume?")
                print("1. Standard validation (labeled events + sample)")
                print("2. Full database analysis (all 2208 cases)")
                resume_choice = input("Enter your choice (1-2): ")
                
                try:
                    if resume_choice == "1":
                        # Resume standard validation
                        print("\nResuming standard validation...")
                        results, start_idx = load_analysis_from_backup(backup_dir)
                        if results is None:
                            print("No valid backup found. Starting from beginning.")
                            continue
                            
                        # Ask for total number of cases to analyze
                        resume_sample_size_input = input("Enter TOTAL number of cases to analyze (default: 128): ")
                        if resume_sample_size_input.strip():
                            resume_sample_size = int(resume_sample_size_input)
                        else:
                            resume_sample_size = 128
                            
                        # Check if we've already processed more cases than requested
                        if start_idx >= resume_sample_size:
                            print(f"\n⚠️ Warning: Already processed {start_idx} cases, which is more than the {resume_sample_size} requested.")
                            print("Would you like to:")
                            print("1. Process additional cases (extend beyond current count)")
                            print("2. Return to main menu")
                            extend_choice = input("Enter your choice (1-2): ")
                            
                            if extend_choice == "1":
                                new_size = int(input(f"Enter new total number of cases (must be > {start_idx}): "))
                                if new_size <= start_idx:
                                    print("Total cases must exceed already processed cases. Returning to main menu.")
                                    continue
                                resume_sample_size = new_size
                            else:
                                continue
                        
                        # Show analysis plan
                        remaining = resume_sample_size - start_idx
                        print(f"\nResuming analysis:")
                        print(f"- {start_idx} cases already processed")
                        print(f"- Will process {remaining} more cases")
                        print(f"- Target total: {resume_sample_size} cases")
                        
                        try:
                            validation_results, validation_metrics = run_validation(
                                llm_predictor,
                                data_manager,
                                results_dir,
                                sample_size=resume_sample_size,
                                continue_from_backup=False,
                                preloaded_results=results,
                                start_idx=start_idx
                            )
                        except Exception as resume_error:
                            print(f"\n❌ Error during resumed validation: {resume_error}")
                            traceback.print_exc()
                            
                    elif resume_choice == "2":
                        # Resume full database analysis
                        print("\nResuming full database analysis (all 2208 cases)...")
                        results, start_idx = load_analysis_from_backup(backup_dir)
                        if results is None:
                            print("No valid backup found. Starting from beginning.")
                            continue
                            
                        try:
                            validation_results, validation_metrics = run_full_database_analysis(
                                llm_predictor,
                                data_manager,
                                results_dir,
                                continue_from_backup=False,
                                preloaded_results=results,
                                start_idx=start_idx
                            )
                        except Exception as resume_error:
                            print(f"\n❌ Error during resumed database analysis: {resume_error}")
                            traceback.print_exc()
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                        continue
                        
                    option_elapsed = time.time() - option_start_time
                    print(f"\nResumed analysis completed in {format_elapsed_time(option_elapsed)}")
                    
                except ValueError as e:
                    print(f"Invalid number: {e}")

            elif choice == '5':
                # Track total session time
                total_session_time = time.time() - main_start_time
                print(f"Exiting the program. Total session time: {format_elapsed_time(total_session_time)}")
                print(f"Started: {datetime.fromtimestamp(main_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                
    except Exception as e:
        # Print elapsed time even on error
        error_time = time.time() - main_start_time
        print(f"Error after {format_elapsed_time(error_time)}: {e}")
        traceback.print_exc()

# Ensure main function is called when the script is executed directly
if __name__ == "__main__":
    main()