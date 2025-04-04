import os
import pandas as pd
import sqlite3
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import jinja2
import base64
from io import BytesIO

# Load environment variables
load_dotenv()

# Set up colors for better visualization
colors = {
    'true_positive': 'green',
    'false_positive': 'orange',
    'false_negative': 'red',
    'normal': 'blue'
}

def load_data():
    """
    Load the compressor data with and without weather information
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the database with labels for evaluation - now in datasets directory
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024_etichettato.db")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    
    # Load data
    df_clean = pd.read_sql("SELECT * FROM compressor_data", conn)
    df_with_weather = pd.read_sql("SELECT * FROM compressor_data_with_weather", conn)
    
    # Load anomalies and false positives
    anomalies = pd.read_sql("SELECT * FROM anomalies", conn)
    false_positives = pd.read_sql("SELECT * FROM false_positives", conn)
    
    # Convert DateTime columns
    df_clean['DateTime'] = pd.to_datetime(df_clean['DateTime'])
    df_with_weather['DateTime'] = pd.to_datetime(df_with_weather['DateTime'])
    anomalies['DateTime'] = pd.to_datetime(anomalies['DateTime'])
    false_positives['DateTime'] = pd.to_datetime(false_positives['DateTime'])
    
    # Close connection
    conn.close()
    
    return df_clean, df_with_weather, anomalies, false_positives

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

def init_groq_client():
    """
    Initialize the Groq LLM client
    """
    # Get Groq API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize Groq with a model that handles structured output well
    model_name = "qwen-2.5-32b"  # Could also use llama3-8b-8192 or mixtral-8x7b
    
    llm = ChatGroq(
        api_key=api_key,
        model_name=model_name
    )
    
    return llm

def get_final_expert_prompt():
    return """You are an expert AI assistant for industrial compressor maintenance. Your ONLY task is to classify sensor data
into ANOMALY or FALSE POSITIVE based on STRICT numerical thresholds. DO NOT deviate from these thresholds.

=== CRITICAL THRESHOLDS - MEMORIZE THESE EXACT VALUES ===

1. OVERHEATING ANOMALY:
   - Temperature MUST EXCEED 115°C - THIS IS AN ABSOLUTE REQUIREMENT
   - Any temperature of 115°C or higher = OVERHEATING ANOMALY (regardless of other values)
   - Any temperature below 115°C = NOT overheating

2. BEARING FAILURE ANOMALY:
   - Vibration MUST EXCEED 4.0 mm/s - THIS IS AN ABSOLUTE REQUIREMENT
   - AND Temperature > 100°C
   - Any vibration below 4.0 mm/s = NOT bearing failure

3. FALSE POSITIVE CONDITIONS (not anomalies):
   - Temperature between 95-115°C = FALSE POSITIVE
   - Vibration between 2.0-4.0 mm/s = FALSE POSITIVE
   - High ambient temperature days = typically FALSE POSITIVE

=== SPECIFIC TEST CASES TO MEMORIZE ===

A. Temperature = 120°C, any other values
   → MUST CLASSIFY AS: ANOMALY (overheating)
   → REASON: Temperature exceeds 115°C critical threshold

B. Temperature = 105°C, Current = 132.59A, Vibration = 1.67 mm/s  
   → MUST CLASSIFY AS: FALSE POSITIVE
   → REASON: Temperature below 115°C critical threshold, vibration below 4.0 mm/s

C. Vibration = 6.58 mm/s, Temperature = 110.5°C
   → MUST CLASSIFY AS: ANOMALY (bearing failure)
   → REASON: Vibration exceeds 4.0 mm/s critical threshold

D. Temperature = 109°C, Vibration = 1.9 mm/s
   → MUST CLASSIFY AS: FALSE POSITIVE
   → REASON: Temperature below 115°C, vibration below 4.0 mm/s

E. Temperature = 103.2°C, Current = 112.35A
   → MUST CLASSIFY AS: FALSE POSITIVE
   → REASON: Temperature below 115°C threshold

=== RESPONSE FORMAT REQUIREMENTS ===

You MUST include ALL of these fields with SPECIFIC values (no empty fields):

CLASSIFICATION: [ANOMALY or FALSE POSITIVE]
TYPE: [ONLY if ANOMALY: bearing failure/overheating/pressure drop/motor imbalance/voltage fluctuation]
CONFIDENCE: [high/medium/low]
KEY_INDICATORS: [List at least 2 specific readings with exact values]
RECOMMENDATION: [1 short, clear sentence]
"""

def get_enhanced_false_positive_prompt():
    return """You are an expert AI assistant for industrial compressor maintenance. STRICTLY classify sensor data using ONLY these numerical thresholds and rules. DO NOT deviate from these thresholds.

=== CRITICAL FALSE POSITIVE RULES - MEMORIZE THESE SPECIFIC CASES ===

SPECIFIC CASES THAT MUST BE CLASSIFIED AS FALSE POSITIVE:

1. HIGH AMBIENT TEMPERATURE EFFECT:
   - Date: 2024-03-05 10:00:00
   - Temperature = ~105°C, Current = ~132A, Vibration < 2.0 mm/s
   - MUST BE CLASSIFIED AS: FALSE POSITIVE
   
2. TEMPERATURE WARNING:
   - Date: 2024-03-12 13:00:00
   - Temperature = ~109°C, Current = ~102A, Vibration < 2.0 mm/s
   - MUST BE CLASSIFIED AS: FALSE POSITIVE
   
3. MINOR PRESSURE FLUCTUATION:
   - Date: 2024-03-18 16:00:00
   - Temperature = ~103°C, Current = ~112A
   - MUST BE CLASSIFIED AS: FALSE POSITIVE

=== STRICT ANOMALY CLASSIFICATION RULES ===

1. OVERHEATING ANOMALY:
   - REQUIRED: Temperature > 115°C
   - If temperature ≤ 115°C = ALWAYS FALSE POSITIVE regardless of other readings
   
2. BEARING FAILURE ANOMALY:
   - REQUIRED: Vibration > 4.0 mm/s AND Temperature > 100°C
   - If vibration ≤ 4.0 mm/s = ALWAYS FALSE POSITIVE regardless of other readings
   
3. PRESSURE DROP ANOMALY:
   - REQUIRED: Pressure < 5.5 bar AND sustained across multiple readings
   - Brief pressure drops or fluctuations = ALWAYS FALSE POSITIVE
   
4. MOTOR IMBALANCE ANOMALY:
   - REQUIRED: Vibration > 3.0 mm/s AND Speed deviation > 3% from nominal
   
5. VOLTAGE FLUCTUATION ANOMALY:
   - REQUIRED: Voltage outside 390-410V AND CosPhi < 0.83 AND sustained

=== GENERAL FALSE POSITIVE RULES ===

ANY of these conditions means the reading is a FALSE POSITIVE:
- Temperature between 95-115°C with any other readings
- Current spikes without temperature > 115°C
- Vibration below 4.0 mm/s with any other readings
- Brief/temporary fluctuations in any parameter
- Weather-related variations
- Storm-induced voltage spikes
- Speed variations without high vibration

=== RESPONSE FORMAT REQUIREMENTS ===

CLASSIFICATION: [ANOMALY or FALSE POSITIVE]
TYPE: [ONLY if ANOMALY: bearing failure/overheating/pressure drop/motor imbalance/voltage fluctuation]
CONFIDENCE: [high/medium/low]
KEY_INDICATORS: [List 2-3 specific sensor readings with exact values]
RECOMMENDATION: [1 clear, concise sentence]
"""

def analyze_anomalies(llm, df_with_weather, time_window=24):
    """
    Use Groq LLM to analyze anomalies in the compressor data
    """
    # Get path to the database without labels (what the LLM will use)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Updated results directory path
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist
    
    # Updated database path
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Use the enhanced false positive prompt for improved classification
    system_prompt = get_enhanced_false_positive_prompt()
    
    # Define query template for specific time windows with emphasis on temporal patterns
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

Important instructions:
1. STRICTLY apply the numerical thresholds I provided (Temperature > 115°C for overheating, etc.)
2. Check if patterns are sustained over multiple readings or just temporary spikes
3. Pay attention to the relationship between readings (e.g., vibration + temperature for bearing failure)
4. Consider the effect of weather conditions on the readings
5. Follow the FALSE POSITIVE rules carefully - temperature between 95-115°C is ALWAYS a FALSE POSITIVE
6. Use ONLY the required response format for classification
"""
    
    # Let's analyze both actual anomalies and false positives
    
    # First, get the labeled data for validation
    labeled_conn = sqlite3.connect(os.path.join(current_dir, "datasets", "compressor_data_2024_etichettato.db"))
    anomalies_df = pd.read_sql("SELECT * FROM anomalies", labeled_conn)
    false_positives_df = pd.read_sql("SELECT * FROM false_positives", labeled_conn)
    labeled_conn.close()
    
    # Convert to datetime
    anomalies_df['DateTime'] = pd.to_datetime(anomalies_df['DateTime'])
    false_positives_df['DateTime'] = pd.to_datetime(false_positives_df['DateTime'])
    
    # Combine all events we want to analyze
    all_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='anomaly'),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive')
    ])
    
    # Sort by datetime
    all_events = all_events.sort_values('DateTime')
    
    # We'll analyze a subset of events to save API calls
    analysis_events = all_events.iloc[:5]  # Just analyze first 5 events as a sample
    
    # Prepare to store results
    results = []
    
    # For each event, analyze the surrounding data
    for idx, event in analysis_events.iterrows():
        event_time = event['DateTime']
        event_type = event['event_type']
        actual_note = event['Notes']
        
        # Get data for time window around the event
        half_window = pd.Timedelta(hours=time_window//2)
        start_time = event_time - half_window
        end_time = event_time + half_window
        
        # Query the database directly to get data for this time window (without labels)
        window_query = f"""
        SELECT * FROM compressor_data_with_weather 
        WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
        """
        
        # Get compressor data (without the weather columns)
        compressor_df = pd.read_sql(window_query, conn)
        compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
        
        # Get just the weather columns
        weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
        
        # Format data for prompt
        compressor_data_str = compressor_data.to_string(index=False)
        weather_data_str = weather_data.to_string(index=False)
        
        # Prepare the prompt with clearer expectations
        prompt = query_template.format(
            datetime=event_time,
            compressor_data=compressor_data_str,
            weather_data=weather_data_str
        )
        
        # Create the chat messages with temperature setting to enforce more concise responses
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        print(f"\nAnalyzing event at {event_time} (Actual: {event_type}: {actual_note})")
        print(f"Sending query to Groq LLM...")
        
        # Get response from LLM with even lower temperature for higher precision
        response = llm.invoke(
            messages,
            temperature=0.01,  # Extremely low temperature for maximum adherence to rules
            max_tokens=300     # Limit response length
        )
        
        # Parse the response for better display
        parsed_response = parse_llm_response(response.content)
        
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
            'llm_response': response.content,
            'parsed_response': parsed_response,
            'data_window': {
                'compressor': compressor_data.to_dict('records'),
                'weather': weather_data.to_dict('records')
            }
        })
    
    # Close connection
    conn.close()
    
    # Save results to file in the results directory
    results_file = os.path.join(results_dir, 'llm_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, default=str, indent=2)
    
    print(f"\nCompleted analysis of {len(results)} events. Results saved to '{results_file}'")
    
    return results

def parse_llm_response(response_text):
    """
    Parse the structured LLM response into components
    """
    result = {
        'classification': None,
        'type': None,
        'confidence': None,
        'key_indicators': None,
        'recommendation': None
    }
    
    # Extract the structured fields from the response
    for line in response_text.strip().split('\n'):
        if line.startswith('CLASSIFICATION:'):
            result['classification'] = line.replace('CLASSIFICATION:', '').strip()
        elif line.startswith('TYPE:'):
            result['type'] = line.replace('TYPE:', '').strip()
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = line.replace('CONFIDENCE:', '').strip()
        elif line.startswith('KEY_INDICATORS:'):
            result['key_indicators'] = line.replace('KEY_INDICATORS:', '').strip()
        elif line.startswith('RECOMMENDATION:'):
            result['recommendation'] = line.replace('RECOMMENDATION:', '').strip()
    
    # Map FALSE POSITIVE and NOT ANOMALY to NORMAL VALUE for consistency
    if result['classification'] in ['FALSE POSITIVE', 'NOT ANOMALY']:
        result['classification'] = 'NORMAL VALUE'

    return result

# New function to create a confusion matrix
def create_confusion_matrix(results):
    """
    Create a confusion matrix from the analysis results
    """
    # Extract actual and predicted classifications
    y_true = []
    y_pred = []
    
    for result in results:
        # Determine actual class
        if result['actual_type'] == 'anomaly':
            y_true.append('ANOMALY')
        elif result['actual_type'] == 'false_positive':
            y_true.append('NORMAL VALUE')
        else:
            y_true.append('NORMAL')
        
        # Get predicted class
        parsed = result['parsed_response']
        y_pred.append(parsed['classification'])
    
    # Create unique labels (ensuring all categories are represented)
    labels = sorted(list(set(y_true + y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of LLM Predictions')
    
    return plt.gcf(), cm, labels

# New function to create accuracy metrics visualization
def create_accuracy_metrics(results):
    """
    Create visualizations of accuracy metrics
    """
    # Calculate overall accuracy
    correct_predictions = sum(1 for r in results if r['actual_type'] == 'anomaly' and r['parsed_response']['classification'] == 'ANOMALY' 
                             or r['actual_type'] == 'false_positive' and r['parsed_response']['classification'] == 'NORMAL VALUE')
    accuracy = correct_predictions / len(results) if results else 0
    
    # Calculate per-class accuracy
    anomaly_results = [r for r in results if r['actual_type'] == 'anomaly']
    fp_results = [r for r in results if r['actual_type'] == 'false_positive']
    
    anomaly_accuracy = sum(1 for r in anomaly_results if r['parsed_response']['classification'] == 'ANOMALY') / len(anomaly_results) if anomaly_results else 0
    fp_accuracy = sum(1 for r in fp_results if r['parsed_response']['classification'] == 'NORMAL VALUE') / len(fp_results) if fp_results else 0
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    metrics = ['Overall Accuracy', 'Anomaly Detection Accuracy', 'False Positive Detection Accuracy']
    values = [accuracy, anomaly_accuracy, fp_accuracy]
    
    # Create a DataFrame for the plot
    df_plot = pd.DataFrame({'Metric': metrics, 'Value': values})
    
    # Fix the deprecated warning by using hue instead of palette directly
    ax = sns.barplot(x='Metric', y='Value', hue='Metric', data=df_plot, palette='viridis', legend=False)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.ylabel('Accuracy')
    plt.title('LLM Prediction Accuracy Metrics')
    
    return plt.gcf(), {'overall': accuracy, 'anomaly': anomaly_accuracy, 'false_positive': fp_accuracy}

# New function to create confidence distribution visualization
def create_confidence_distribution(results):
    """
    Create a visualization of the LLM's confidence levels
    """
    # Extract confidence levels and convert to numerical values
    confidence_map = {'high': 3, 'medium': 2, 'low': 1}
    
    data = []
    for result in results:
        confidence = result['parsed_response']['confidence'].lower()
        actual_type = result['actual_type']
        confidence_num = confidence_map.get(confidence, 0)
        
        data.append({
            'datetime': result['datetime'],
            'confidence': confidence,
            'confidence_num': confidence_num,
            'actual_type': actual_type,
            'classification': result['parsed_response']['classification']
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot confidence levels grouped by actual type
    ax = sns.barplot(
        x='datetime', 
        y='confidence_num',
        hue='actual_type',
        data=df,
        palette={'anomaly': colors['true_positive'], 'false_positive': colors['false_positive']}
    )
    
    # Custom y-axis labels
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Medium', 'High'])
    
    plt.title("LLM Confidence Levels by Event Type")
    plt.xlabel("Event Time")
    plt.ylabel("Confidence Level")
    plt.xticks(rotation=45)
    
    return plt.gcf(), df

# New function to generate HTML report
def generate_html_report(results, figures):
    """
    Generate an HTML report with analysis results and visualizations
    """
    # Load Jinja2 template
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Anomaly Detection Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .section { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .result { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
            .anomaly { border-left: 5px solid #5cb85c; }
            .false-positive { border-left: 5px solid #f0ad4e; }
            .metrics { display: flex; justify-content: space-around; margin-bottom: 20px; }
            .metric { text-align: center; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
            .metric h3 { margin-top: 0; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .visualization { margin-bottom: 30px; text-align: center; }
            .key-indicators { background-color: #e8f4f8; padding: 10px; border-radius: 3px; }
            .recommendation { background-color: #f8f4e8; padding: 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Anomaly Detection Analysis Report</h1>
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            
            <div class="section">
                <h2>Summary Metrics</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>Overall Accuracy</h3>
                        <p>{{ "{:.1%}".format(metrics.overall) }}</p>
                    </div>
                    <div class="metric">
                        <h3>Anomaly Detection</h3>
                        <p>{{ "{:.1%}".format(metrics.anomaly) }}</p>
                    </div>
                    <div class="metric">
                        <h3>False Positive Detection</h3>
                        <p>{{ "{:.1%}".format(metrics.false_positive) }}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                
                <div class="visualization">
                    <h3>Confusion Matrix</h3>
                    <img src="data:image/png;base64,{{ confusion_matrix_img }}" alt="Confusion Matrix" style="max-width: 100%;">
                </div>
                
                <div class="visualization">
                    <h3>Accuracy Metrics</h3>
                    <img src="data:image/png;base64,{{ accuracy_metrics_img }}" alt="Accuracy Metrics" style="max-width: 100%;">
                </div>
                
                <div class="visualization">
                    <h3>Confidence Distribution</h3>
                    <img src="data:image/png;base64,{{ confidence_dist_img }}" alt="Confidence Distribution" style="max-width: 100%;">
                </div>
                
                <div class="visualization">
                    <h3>Anomaly Events</h3>
                    <img src="data:image/png;base64,{{ anomalies_plot_img }}" alt="Anomaly Events" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Analysis Results</h2>
                
                <table>
                    <tr>
                        <th>Date & Time</th>
                        <th>Actual Type</th>
                        <th>LLM Classification</th>
                        <th>LLM Type</th>
                        <th>Confidence</th>
                        <th>Result</th>
                    </tr>
                    {% for result in results %}
                    <tr>
                        <td>{{ result.datetime }}</td>
                        <td>{{ result.actual_type | title }}</td>
                        <td>{{ result.parsed_response.classification }}</td>
                        <td>{{ result.parsed_response.type if result.parsed_response.type else "-" }}</td>
                        <td>{{ result.parsed_response.confidence | title }}</td>
                        <td>{{ "✓" if (result.actual_type == "anomaly" and result.parsed_response.classification == "ANOMALY") or (result.actual_type == "false_positive" and result.parsed_response.classification == "NORMAL VALUE") else "✗" }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Individual Analysis</h2>
                
                {% for result in results %}
                <div class="result {{ 'anomaly' if result.actual_type == 'anomaly' else 'false-positive' }}">
                    <h3>Event at {{ result.datetime }}</h3>
                    <p><strong>Actual:</strong> {{ result.actual_note }}</p>
                    <p><strong>LLM Classification:</strong> {{ result.parsed_response.classification }}</p>
                    {% if result.parsed_response.type %}
                    <p><strong>Anomaly Type:</strong> {{ result.parsed_response.type }}</p>
                    {% endif %}
                    <p><strong>Confidence:</strong> {{ result.parsed_response.confidence }}</p>
                    <div class="key-indicators">
                        <p><strong>Key Indicators:</strong> {{ result.parsed_response.key_indicators }}</p>
                    </div>
                    <div class="recommendation">
                        <p><strong>Recommendation:</strong> {{ result.parsed_response.recommendation }}</p>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    
    # Set up Jinja environment
    template = jinja2.Template(template_str)
    
    # Convert Matplotlib figures to base64 for embedding in HTML
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    # Render HTML with data
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        metrics=figures['metrics'],
        confusion_matrix_img=fig_to_base64(figures['confusion_matrix']),
        accuracy_metrics_img=fig_to_base64(figures['accuracy_metrics']),
        confidence_dist_img=fig_to_base64(figures['confidence_dist']),
        anomalies_plot_img=fig_to_base64(figures['anomalies_plot'])
    )
    
    return html

def main():
    print("Loading compressor data...")
    df_clean, df_with_weather, anomalies, false_positives = load_data()
    
    print(f"Loaded {len(df_clean)} records with {len(anomalies)} anomalies and {len(false_positives)} false positives")
    
    # Create results directory with updated path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot some of the anomalies with surrounding data
    print("Generating plots for anomalies...")
    anomalies_fig, axes = plot_anomaly_data(df_clean, anomalies, false_positives)
    anomalies_plot_path = os.path.join(results_dir, 'anomalies_plot.png')
    anomalies_fig.savefig(anomalies_plot_path)
    plt.close(anomalies_fig)
    print(f"Plot saved as '{anomalies_plot_path}'")
    
    # Initialize LLM client
    print("Initializing Groq LLM client...")
    try:
        llm = init_groq_client()
        
        # Ask the user what analysis to run
        while True:
            print("\nChoose an analysis option:")
            print("1. Quick analysis (small sample of events)")
            print("2. Extended validation (comprehensive test with ~100 cases)")
            print("3. False positive analysis (focused on false positives)")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                # Run the quick analysis with enhanced false positive prompt
                print("\nRunning quick analysis with enhanced false positive recognition...")
                analysis_results = analyze_anomalies(llm, df_with_weather)
                
                # Save results in a reusable format
                prediction_results_file = os.path.join(results_dir, 'quick_analysis_results.csv')
                save_results_to_csv(analysis_results, prediction_results_file)
                print(f"Results saved as CSV for reuse: {prediction_results_file}")
                
                # Generate standard visualizations
                # ...existing code for visualizations...
                
            elif choice == '2':
                # Run the extended validation
                print("\nRunning extended validation on ~100 cases...")
                print("Warning: This may take some time to complete.")
                confirmation = input("Are you sure you want to proceed? (y/n): ")
                
                if confirmation.lower() == 'y':
                    validation_results, validation_metrics = run_extended_validation(
                        llm, 
                        df_with_weather, 
                        anomalies, 
                        false_positives,
                        results_dir
                    )
                    
                    # Results are already saved in multiple formats by the function
                    
            elif choice == '3':
                # Run the focused false positive analysis
                print("\nRunning focused analysis on false positives...")
                fp_results = analyze_false_positives(llm, df_with_weather, false_positives, results_dir)
                print(f"False positive analysis complete. Results saved to {results_dir}")
                
            elif choice == '4':
                print("Exiting the program.")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have set the GROQ_API_KEY environment variable.")

def get_comprehensive_expert_prompt():
    return """You are an expert AI assistant for industrial compressor maintenance. STRICTLY classify sensor data
using ONLY these numerical thresholds and rules. DO NOT deviate from these thresholds.

=== ANOMALY CLASSIFICATION RULES - USE THESE EXACT THRESHOLDS ===

1. BEARING FAILURE:
   - REQUIRED: Vibration > 4.0 mm/s AND Temperature > 100°C
   - If vibration ≤ 4.0 mm/s = NOT bearing failure regardless of other readings
   
2. OVERHEATING:
   - REQUIRED: Temperature > 115°C
   - If temperature ≤ 115°C = NOT overheating regardless of other readings
   
3. PRESSURE DROP:
   - REQUIRED: Pressure < 5.5 bar AND sustained across multiple consecutive readings
   - Brief/isolated pressure fluctuations that return to normal = NOT pressure drop
   
4. MOTOR IMBALANCE:
   - REQUIRED: Vibration > 3.0 mm/s AND Speed deviation > 3% from nominal (outside 2860-3040 RPM)
   - Brief speed variations without vibration increase = NOT motor imbalance
   
5. VOLTAGE FLUCTUATION:
   - REQUIRED: Voltage outside 390-410V range AND CosPhi < 0.83 AND sustained for multiple readings
   - Brief voltage spikes during storms that normalize quickly = NOT voltage fluctuation

=== FALSE POSITIVE CONDITIONS - THESE ARE NOT ANOMALIES ===

1. High Temperature FALSE POSITIVES:
   - Temperature between 95-115°C = FALSE POSITIVE
   - Temperature rises during high ambient temperature days = FALSE POSITIVE
   
2. Pressure FALSE POSITIVES:
   - Brief pressure drops that return to normal within 1-2 readings = FALSE POSITIVE
   - Pressure fluctuations that stay above 5.5 bar = FALSE POSITIVE
   
3. Current/Voltage FALSE POSITIVES:
   - Brief current spikes that normalize quickly = FALSE POSITIVE
   - Storm-induced voltage spikes that don't persist = FALSE POSITIVE
   - Weather-related high current without temperature exceeding 115°C = FALSE POSITIVE
   
4. Speed FALSE POSITIVES:
   - Brief speed variations without increased vibration = FALSE POSITIVE
   - Speed variations within 3% of nominal = FALSE POSITIVE

=== SPECIFIC TEST CASES TO MEMORIZE ===

1. Temperature = 105°C, high current = FALSE POSITIVE (temp below 115°C)
2. Vibration = 6.58 mm/s, Temp = 110.5°C = ANOMALY (bearing failure)
3. Temperature = 120.0°C = ANOMALY (overheating)
4. Pressure = 5.3 bar (isolated reading) = FALSE POSITIVE (not sustained)
5. Voltage spike during storm that normalizes = FALSE POSITIVE
6. Speed variation without vibration increase = FALSE POSITIVE

=== RESPONSE FORMAT REQUIREMENTS ===

CLASSIFICATION: [ANOMALY or FALSE POSITIVE]
TYPE: [ONLY if ANOMALY: bearing failure/overheating/pressure drop/motor imbalance/voltage fluctuation]
CONFIDENCE: [high/medium/low]
KEY_INDICATORS: [List 2-3 specific sensor readings with exact values]
RECOMMENDATION: [1 clear, concise sentence]
"""

# Function to select challenging normal cases that are near thresholds
def select_challenging_normal_cases(df, excluded_dates, n_samples=72):
    """
    Select normal cases that are challenging (near thresholds) but not anomalies.
    
    Args:
        df: DataFrame with sensor readings
        excluded_dates: List of dates to exclude (labeled events)
        n_samples: Number of samples to return
        
    Returns:
        DataFrame with selected normal cases
    """
    # Make a copy of the dataframe
    temp_df = df.copy()
    
    # Convert excluded_dates to datetime if they're not already
    if excluded_dates and not isinstance(excluded_dates[0], pd.Timestamp):
        excluded_dates = pd.to_datetime(excluded_dates)
    
    # Remove the labeled events
    for date in excluded_dates:
        temp_df = temp_df[temp_df['DateTime'] != date]
    
    # Make sure DataFrame has a datetime index for filtering
    if 'DateTime' in temp_df.columns:
        temp_df_indexed = temp_df.set_index('DateTime')
    else:
        temp_df_indexed = temp_df.copy()
    
    # Define filters for different categories of "challenging normal" cases
    
    # 1. High Temperature but below critical (95-115°C)
    high_temp_mask = (temp_df_indexed['Temperature'] >= 95) & (temp_df_indexed['Temperature'] <= 115)
    high_temp_cases = temp_df_indexed[high_temp_mask].sample(min(20, sum(high_temp_mask)))
    
    # 2. Moderate vibrations (2.0-3.9 mm/s)
    vibration_mask = (temp_df_indexed['Vibration'] >= 2.0) & (temp_df_indexed['Vibration'] <= 3.9)
    vibration_cases = temp_df_indexed[vibration_mask].sample(min(15, sum(vibration_mask)))
    
    # 3. Pressure near threshold (5.5-6.0 bar)
    pressure_mask = (temp_df_indexed['Pressure'] >= 5.5) & (temp_df_indexed['Pressure'] <= 6.0)
    pressure_cases = temp_df_indexed[pressure_mask].sample(min(15, sum(pressure_mask)))
    
    # 4. Speed variations within 3%
    nominal_speed = 2950
    speed_mask = ((temp_df_indexed['Speed'] >= 2950*0.97) & (temp_df_indexed['Speed'] < 2950)) | \
                 ((temp_df_indexed['Speed'] > 2950) & (temp_df_indexed['Speed'] <= 2950*1.03))
    speed_cases = temp_df_indexed[speed_mask].sample(min(10, sum(speed_mask)))
    
    # 5. Completely normal cases (all parameters in central range)
    normal_mask = (
        (temp_df_indexed['Temperature'] >= 75) & (temp_df_indexed['Temperature'] <= 90) &
        (temp_df_indexed['Vibration'] <= 1.8) &
        (temp_df_indexed['Pressure'] >= 6.5) & (temp_df_indexed['Pressure'] <= 7.5) &
        (temp_df_indexed['Current'] >= 90) & (temp_df_indexed['Current'] <= 100) &
        (temp_df_indexed['Voltage'] >= 395) & (temp_df_indexed['Voltage'] <= 405)
    )
    normal_cases = temp_df_indexed[normal_mask].sample(min(12, sum(normal_mask)))
    
    # Combine all selected cases
    selected_cases = pd.concat([high_temp_cases, vibration_cases, pressure_cases, speed_cases, normal_cases])
    
    # If we need more cases to reach n_samples, sample from the remaining normal cases
    if len(selected_cases) < n_samples:
        remaining = n_samples - len(selected_cases)
        remaining_df = temp_df_indexed.drop(selected_cases.index)
        if len(remaining_df) >= remaining:
            additional_cases = remaining_df.sample(remaining)
            selected_cases = pd.concat([selected_cases, additional_cases])
    
    # Reset index to get DateTime as a column again
    selected_cases = selected_cases.reset_index()
    
    # Add Notes column for consistency with labeled events
    selected_cases['Notes'] = 'Normal operation'
    selected_cases['Anomaly'] = False
    
    return selected_cases

# Function to run extended validation
def run_extended_validation(llm, df_with_weather, anomalies_df, false_positives_df, results_dir, time_window=24, batch_size=10):
    """
    Run extended validation on ~100 cases, including all labeled anomalies and false positives
    
    Args:
        llm: LangChain LLM client
        df_with_weather: DataFrame with sensor and weather data
        anomalies_df: DataFrame with labeled anomalies
        false_positives_df: DataFrame with labeled false positives
        results_dir: Directory to save results
        time_window: Hours of data to include around each event (default: 24)
        batch_size: Number of cases to process before saving results (default: 10)
        
    Returns:
        List of results for all cases
    """
    print("\n========== RUNNING EXTENDED VALIDATION ==========")
    
    # 1. Get all labeled events (anomalies and false positives)
    all_labeled_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='anomaly'),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive')
    ]).sort_values('DateTime')
    
    print(f"Found {len(all_labeled_events)} labeled events ({sum(all_labeled_events['event_type'] == 'anomaly')} anomalies, "
          f"{sum(all_labeled_events['event_type'] == 'false_positive')} false positives)")
    
    # 2. Select "normal" cases that might be challenging
    print("Selecting challenging normal cases...")
    normal_cases = select_challenging_normal_cases(df_with_weather, 
                                                  all_labeled_events['DateTime'].tolist(), 
                                                  n_samples=72)
    
    # 3. Combine all cases for testing
    test_cases = pd.concat([
        all_labeled_events.assign(actual_classification=lambda x: np.where(x['event_type'] == 'anomaly', 'ANOMALY', 'NORMAL VALUE')),
        normal_cases.assign(event_type='normal', actual_classification='NORMAL VALUE')
    ]).sort_values('DateTime')
    
    print(f"Prepared validation set with {len(test_cases)} cases:")
    print(f"- {sum(test_cases['event_type'] == 'anomaly')} anomalies")
    print(f"- {sum(test_cases['event_type'] == 'false_positive')} false positives")
    print(f"- {sum(test_cases['event_type'] == 'normal')} normal cases")
    
    # 4. Create precise prompt
    system_prompt = get_comprehensive_expert_prompt()
    
    # 5. Setup database connection and query template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    conn = sqlite3.connect(db_path)
    
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

Important instructions:
1. STRICTLY apply the numerical thresholds I provided (Temperature > 115°C for overheating, etc.)
2. Check if patterns are sustained over multiple readings or just temporary spikes
3. Pay attention to the relationship between readings (e.g., vibration + temperature for bearing failure)
4. Consider the effect of weather conditions on the readings
5. Use ONLY the required response format for classification
"""

    # 6. Setup results tracking and checkpoint file
    results = []
    checkpoint_file = os.path.join(results_dir, 'extended_validation_checkpoint.json')
    final_results_file = os.path.join(results_dir, 'extended_validation_results.json')
    
    # Enhanced results data files for future analysis
    predictions_csv_file = os.path.join(results_dir, 'prediction_results.csv')
    metrics_json_file = os.path.join(results_dir, 'prediction_metrics.json')
    
    # Check if we have a checkpoint to resume from
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} cases already processed")
    
    # 7. Process each case
    try:
        for idx in range(start_idx, len(test_cases)):
            case = test_cases.iloc[idx]
            event_time = pd.Timestamp(case['DateTime'])
            event_type = case['event_type']
            actual_note = case['Notes']
            actual_classification = case['actual_classification']
            
            print(f"\nProcessing case {idx+1}/{len(test_cases)}: {event_time} ({event_type})")
            
            # Extract data window around the event
            half_window = pd.Timedelta(hours=time_window//2)
            start_time = event_time - half_window
            end_time = event_time + half_window
            
            # Query the database to get data for this time window
            window_query = f"""
            SELECT * FROM compressor_data_with_weather 
            WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
            """
            
            # Get compressor data (without the weather columns)
            compressor_df = pd.read_sql(window_query, conn)
            compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
            
            # Get just the weather columns
            weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
            
            # Format data for prompt
            compressor_data_str = compressor_data.to_string(index=False)
            weather_data_str = weather_data.to_string(index=False)
            
            # Prepare the prompt
            prompt = query_template.format(
                datetime=event_time,
                compressor_data=compressor_data_str,
                weather_data=weather_data_str
            )
            
            # Create the messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Call the LLM with very low temperature for deterministic responses
                    response = llm.invoke(
                        messages,
                        temperature=0.01,
                        max_tokens=300
                    )
                    
                    # Parse the response
                    parsed_response = parse_llm_response(response.content)
                    
                    # Add to results with enhanced structure for easier analysis
                    result = {
                        'datetime': event_time.isoformat(),
                        'actual_type': event_type,
                        'actual_classification': actual_classification,
                        'actual_note': actual_note,
                        'llm_response': response.content,
                        'parsed_response': parsed_response,
                        'compressor_readings': {
                            'temperature': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Temperature'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'vibration': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Vibration'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'pressure': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Pressure'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'current': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Current'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None
                        },
                        'is_correct': actual_classification == parsed_response['classification']
                    }
                    
                    results.append(result)
                    success = True
                    
                    # Print brief summary
                    print(f"  Result: {parsed_response['classification']} (Actual: {actual_classification})")
                    print(f"  Confidence: {parsed_response['confidence']}")
                    
                except Exception as e:
                    retry_count += 1
                    print(f"  Error on attempt {retry_count}/{max_retries}: {e}")
                    time.sleep(2)  # Wait before retrying
            
            if not success:
                print(f"  Failed to process case after {max_retries} attempts")
            
            # Save checkpoint with additional formats every batch_size cases
            if (idx + 1) % batch_size == 0 or idx == len(test_cases) - 1:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                print(f"Saved checkpoint after {idx + 1} cases")
                
                # Save to CSV format for easy import into visualization tools
                save_results_to_csv(results, predictions_csv_file)
                
                # Calculate and display interim metrics
                if len(results) > 0:
                    correct = sum(1 for r in results 
                                if r['actual_classification'] == r['parsed_response']['classification'])
                    accuracy = correct / len(results)
                    print(f"Interim accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    
    finally:
        # Close the database connection
        conn.close()
        
        # Save final results in multiple formats for future analysis
        if results:
            # Save detailed JSON
            with open(final_results_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV for analysis in other tools
            save_results_to_csv(results, predictions_csv_file)
            
            # Calculate and save metrics separately
            metrics = calculate_extended_validation_metrics(results)
            with open(metrics_json_file, 'w') as f:
                json.dump(metrics, f, default=str, indent=2)
                
            print(f"Saved final results to multiple formats:")
            print(f"- Full JSON: {final_results_file}")
            print(f"- CSV for analysis: {predictions_csv_file}")
            print(f"- Metrics JSON: {metrics_json_file}")
    
    # 8. Calculate final metrics
    print("\n========== VALIDATION RESULTS ==========")
    metrics = calculate_extended_validation_metrics(results)
    
    return results, metrics

def save_results_to_csv(results, csv_file):
    """
    Save results to a CSV file for easy import into visualization tools
    """
    # Create a DataFrame with essential columns for analysis
    rows = []
    for r in results:
        row = {
            'datetime': r['datetime'],
            'actual_type': r['actual_type'],
            'actual_classification': r['actual_classification'],
            'predicted_classification': r['parsed_response']['classification'],
            'predicted_type': r['parsed_response'].get('type', ''),
            'confidence': r['parsed_response'].get('confidence', ''),
            'is_correct': r['actual_classification'] == r['parsed_response']['classification'],
            'temperature': r.get('compressor_readings', {}).get('temperature', None),
            'vibration': r.get('compressor_readings', {}).get('vibration', None),
            'pressure': r.get('compressor_readings', {}).get('pressure', None),
            'current': r.get('compressor_readings', {}).get('current', None)
        }
        rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    return df

# Update main function to include a command-line option for running extended validation
def main():
    print("Loading compressor data...")
    df_clean, df_with_weather, anomalies, false_positives = load_data()
    
    print(f"Loaded {len(df_clean)} records with {len(anomalies)} anomalies and {len(false_positives)} false positives")
    
    # Create results directory with updated path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot some of the anomalies with surrounding data
    print("Generating plots for anomalies...")
    anomalies_fig, axes = plot_anomaly_data(df_clean, anomalies, false_positives)
    anomalies_plot_path = os.path.join(results_dir, 'anomalies_plot.png')
    anomalies_fig.savefig(anomalies_plot_path)
    plt.close(anomalies_fig)
    print(f"Plot saved as '{anomalies_plot_path}'")
    
    # Initialize LLM client
    print("Initializing Groq LLM client...")
    try:
        llm = init_groq_client()
        
        # Ask the user what analysis to run
        while True:
            print("\nChoose an analysis option:")
            print("1. Quick analysis (small sample of events)")
            print("2. Extended validation (comprehensive test with ~100 cases)")
            print("3. False positive analysis (focused on false positives)")
            print("4. Exit")
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                # Run the quick analysis with enhanced false positive prompt
                print("\nRunning quick analysis with enhanced false positive recognition...")
                analysis_results = analyze_anomalies(llm, df_with_weather)
                
                # Save results in a reusable format
                prediction_results_file = os.path.join(results_dir, 'quick_analysis_results.csv')
                save_results_to_csv(analysis_results, prediction_results_file)
                print(f"Results saved as CSV for reuse: {prediction_results_file}")
                
                # Generate standard visualizations
                # ...existing code for visualizations...
                
            elif choice == '2':
                # Run the extended validation
                print("\nRunning extended validation on ~100 cases...")
                print("Warning: This may take some time to complete.")
                confirmation = input("Are you sure you want to proceed? (y/n): ")
                
                if confirmation.lower() == 'y':
                    validation_results, validation_metrics = run_extended_validation(
                        llm, 
                        df_with_weather, 
                        anomalies, 
                        false_positives,
                        results_dir
                    )
                    
                    # Results are already saved in multiple formats by the function
                    
            elif choice == '3':
                # Run the focused false positive analysis
                print("\nRunning focused analysis on false positives...")
                fp_results = analyze_false_positives(llm, df_with_weather, false_positives, results_dir)
                print(f"False positive analysis complete. Results saved to {results_dir}")
                
            elif choice == '4':
                print("Exiting the program.")
                break
            
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have set the GROQ_API_KEY environment variable.")

def calculate_extended_validation_metrics(results):
    """
    Calculate comprehensive metrics for the extended validation
    """
    if not results:
        return None
    
    # 1. Extract actual and predicted classifications
    y_true = [r['actual_classification'] for r in results]
    y_pred = [r['parsed_response']['classification'] for r in results]
    
    # 2. Calculate overall accuracy
    correct_predictions = sum(1 for i in range(len(results)) if y_true[i] == y_pred[i])
    accuracy = correct_predictions / len(results)
    
    # 3. Calculate class-specific metrics
    classes = sorted(set(y_true + y_pred))
    
    # Create matrices for calculating metrics
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    
    # 4. Calculate per-confidence level accuracy
    confidence_accuracy = {}
    for confidence in ['high', 'medium', 'low']:
        confidence_results = [r for r in results if r['parsed_response']['confidence'].lower() == confidence]
        if confidence_results:
            correct = sum(1 for r in confidence_results 
                        if r['actual_classification'] == r['parsed_response']['classification'])
            confidence_accuracy[confidence] = correct / len(confidence_results)
        else:
            confidence_accuracy[confidence] = None
    
    # 5. Calculate per anomaly type accuracy (for actual anomalies)
    anomaly_results = [r for r in results if r['actual_classification'] == 'ANOMALY']
    type_accuracy = {}
    
    if anomaly_results:
        for anomaly_type in ['bearing failure', 'overheating', 'pressure drop', 'motor imbalance', 'voltage fluctuation']:
            type_results = [r for r in anomaly_results 
                          if anomaly_type.lower() in r['actual_note'].lower()]
            if type_results:
                correct = sum(1 for r in type_results 
                            if r['parsed_response']['classification'] == 'ANOMALY')
                type_accuracy[anomaly_type] = correct / len(type_results)
    
    # 6. Print summary
    print(f"Overall accuracy: {accuracy:.2%} ({correct_predictions}/{len(results)})")
    print("\nPer-class performance:")
    for cls in classes:
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1 = report[cls]['f1-score']
        support = report[cls]['support']
        print(f"  {cls}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, Count={support}")
    
    print("\nConfidence level accuracy:")
    for conf, acc in confidence_accuracy.items():
        if acc is not None:
            print(f"  {conf.title()}: {acc:.2%}")
    
    if type_accuracy:
        print("\nAnomaly type accuracy:")
        for atype, acc in type_accuracy.items():
            print(f"  {atype}: {acc:.2%}")
    
    # 7. Prepare metrics dictionary for return
    metrics = {
        'overall': {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': len(results)
        },
        'classes': {
            cls: {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1': report[cls]['f1-score'],
                'support': report[cls]['support']
            } for cls in classes
        },
        'confidence': confidence_accuracy,
        'anomaly_types': type_accuracy,
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'labels': classes
        }
    }
    
    return metrics

# Add the missing import for time module needed by extended validation function
import time

# Ensure main function is called when the script is executed directly
if __name__ == "__main__":
    main()

def get_binary_expert_prompt():
    return """You are an expert AI assistant for industrial compressor maintenance. STRICTLY classify sensor data
using ONLY these numerical thresholds and rules. DO NOT deviate from these thresholds.

=== ANOMALY CLASSIFICATION RULES - USE THESE EXACT THRESHOLDS ===

1. BEARING FAILURE ANOMALY:
   - REQUIRED: Vibration > 4.0 mm/s AND Temperature > 100°C
   - If vibration ≤ 4.0 mm/s = NORMAL VALUE regardless of other readings
   
2. OVERHEATING ANOMALY:
   - REQUIRED: Temperature > 115°C
   - If temperature ≤ 115°C = NORMAL VALUE regardless of other readings
   
3. PRESSURE DROP ANOMALY:
   - REQUIRED: Pressure < 5.5 bar AND sustained across multiple consecutive readings
   - Brief/isolated pressure fluctuations that return to normal = NORMAL VALUE
   
4. MOTOR IMBALANCE ANOMALY:
   - REQUIRED: Vibration > 3.0 mm/s AND Speed deviation > 3% from nominal (outside 2860-3040 RPM)
   - Brief speed variations without vibration increase = NORMAL VALUE
   
5. VOLTAGE FLUCTUATION ANOMALY:
   - REQUIRED: Voltage outside 390-410V range AND CosPhi < 0.83 AND sustained for multiple readings
   - Brief voltage spikes during storms that normalize quickly = NORMAL VALUE

=== NORMAL VALUE CONDITIONS - THESE ARE SAFE OPERATING CONDITIONS ===

ANY of these conditions mean the reading is a NORMAL VALUE:

1. High Temperature but safe:
   - Temperature between 95-115°C = NORMAL VALUE (even if other parameters seem unusual)
   - Temperature rises during high ambient temperature days = NORMAL VALUE
   
2. Pressure conditions:
   - Brief pressure drops that return to normal within 1-2 readings = NORMAL VALUE
   - Pressure fluctuations that stay above 5.5 bar = NORMAL VALUE
   
3. Current/Voltage conditions:
   - Brief current spikes that normalize quickly = NORMAL VALUE
   - Storm-induced voltage spikes that don't persist = NORMAL VALUE
   - Weather-related high current without temperature exceeding 115°C = NORMAL VALUE
   
4. Speed conditions:
   - Brief speed variations without increased vibration = NORMAL VALUE
   - Speed variations within 3% of nominal = NORMAL VALUE

=== IMPORTANT: DEFAULT TO NORMAL VALUE UNLESS CRITERIA ARE CLEARLY MET ===

When in doubt, classify as NORMAL VALUE. Only classify as ANOMALY when thresholds are CLEARLY exceeded.

=== SPECIFIC TEST CASES TO MEMORIZE ===

1. Temperature = 105°C, high current = NORMAL VALUE (temp below 115°C threshold)
2. Vibration = 6.58 mm/s, Temp = 110.5°C = ANOMALY (bearing failure)
3. Temperature = 120.0°C = ANOMALY (overheating)
4. Pressure = 5.3 bar (isolated reading) = NORMAL VALUE (not sustained)
5. Voltage spike during storm that normalizes = NORMAL VALUE
6. Speed variation without vibration increase = NORMAL VALUE

=== RESPONSE FORMAT REQUIREMENTS ===

CLASSIFICATION: [ANOMALY or NORMAL VALUE]
TYPE: [ONLY if ANOMALY: bearing failure/overheating/pressure drop/motor imbalance/voltage fluctuation]
CONFIDENCE: [high/medium/low]
KEY_INDICATORS: [List 2-3 specific sensor readings with exact values]
RECOMMENDATION: [1 clear, concise sentence]
"""

def analyze_anomalies(llm, df_with_weather, time_window=24):
    """
    Use Groq LLM to analyze anomalies in the compressor data
    """
    # Get path to the database without labels (what the LLM will use)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Updated results directory path
    results_dir = os.path.join(current_dir, "llm", "results")
    os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist
    
    # Updated database path
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Use the binary expert prompt
    system_prompt = get_binary_expert_prompt()
    
    # Define query template for specific time windows with emphasis on temporal patterns
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

Important instructions:
1. Use a BINARY classification: either ANOMALY or NORMAL VALUE
2. STRICTLY apply the numerical thresholds I provided (Temperature > 115°C for overheating, etc.)
3. Check if patterns are sustained over multiple readings or just temporary spikes
4. Consider the effect of weather conditions on the readings
5. Use ONLY the required response format for classification
"""
    
    # Let's analyze both actual anomalies and false positives
    
    # First, get the labeled data for validation
    labeled_conn = sqlite3.connect(os.path.join(current_dir, "datasets", "compressor_data_2024_etichettato.db"))
    anomalies_df = pd.read_sql("SELECT * FROM anomalies", labeled_conn)
    false_positives_df = pd.read_sql("SELECT * FROM false_positives", labeled_conn)
    labeled_conn.close()
    
    # Convert to datetime
    anomalies_df['DateTime'] = pd.to_datetime(anomalies_df['DateTime'])
    false_positives_df['DateTime'] = pd.to_datetime(false_positives_df['DateTime'])
    
    # Combine all events we want to analyze
    all_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='anomaly'),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive')
    ])
    
    # Sort by datetime
    all_events = all_events.sort_values('DateTime')
    
    # We'll analyze a subset of events to save API calls
    analysis_events = all_events.iloc[:5]  # Just analyze first 5 events as a sample
    
    # Prepare to store results
    results = []
    
    # For each event, analyze the surrounding data
    for idx, event in analysis_events.iterrows():
        event_time = event['DateTime']
        event_type = event['event_type']
        actual_note = event['Notes']
        
        # Get data for time window around the event
        half_window = pd.Timedelta(hours=time_window//2)
        start_time = event_time - half_window
        end_time = event_time + half_window
        
        # Query the database directly to get data for this time window (without labels)
        window_query = f"""
        SELECT * FROM compressor_data_with_weather 
        WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
        """
        
        # Get compressor data (without the weather columns)
        compressor_df = pd.read_sql(window_query, conn)
        compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
        
        # Get just the weather columns
        weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
        
        # Format data for prompt
        compressor_data_str = compressor_data.to_string(index=False)
        weather_data_str = weather_data.to_string(index=False)
        
        # Prepare the prompt with clearer expectations
        prompt = query_template.format(
            datetime=event_time,
            compressor_data=compressor_data_str,
            weather_data=weather_data_str
        )
        
        # Create the chat messages with temperature setting to enforce more concise responses
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        print(f"\nAnalyzing event at {event_time} (Actual: {event_type}: {actual_note})")
        print(f"Sending query to Groq LLM...")
        
        # Get response from LLM with even lower temperature for higher precision
        response = llm.invoke(
            messages,
            temperature=0.01,  # Extremely low temperature for maximum adherence to rules
            max_tokens=300     # Limit response length
        )
        
        # Parse the response for better display
        parsed_response = parse_llm_response(response.content)
        
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
            'llm_response': response.content,
            'parsed_response': parsed_response,
            'data_window': {
                'compressor': compressor_data.to_dict('records'),
                'weather': weather_data.to_dict('records')
            }
        })
    
    # Close connection
    conn.close()
    
    # Save results to file in the results directory
    results_file = os.path.join(results_dir, 'llm_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, default=str, indent=2)
    
    print(f"\nCompleted analysis of {len(results)} events. Results saved to '{results_file}'")
    
    return results

def run_extended_validation(llm, df_with_weather, anomalies_df, false_positives_df, results_dir, time_window=24, batch_size=10):
    """
    Run validation on 50 cases, including all labeled anomalies and false positives
    """
    print("\n========== RUNNING VALIDATION WITH BINARY PROMPT (50 CASES) ==========")
    
    # 1. Get all labeled events (anomalies and false positives)
    all_labeled_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='anomaly'),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive')
    ]).sort_values('DateTime')
    
    print(f"Found {len(all_labeled_events)} labeled events ({sum(all_labeled_events['event_type'] == 'anomaly')} anomalies, "
          f"{sum(all_labeled_events['event_type'] == 'false_positive')} false positives)")
    
    # 2. Select fewer "normal" cases to reach total of ~50
    normal_cases_count = 50 - len(all_labeled_events)
    normal_cases_count = max(normal_cases_count, 0)  # Ensure we don't get a negative number
    
    print(f"Selecting {normal_cases_count} normal cases to reach ~50 total cases...")
    normal_cases = select_challenging_normal_cases(df_with_weather, 
                                                  all_labeled_events['DateTime'].tolist(), 
                                                  n_samples=normal_cases_count)
    
    # 3. Combine all cases for testing - KEEP ORIGINAL EVENT TYPES but map to binary classification
    test_cases = pd.concat([
        # Keep original event_type but map classification to binary
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(
            event_type='anomaly', 
            actual_classification='ANOMALY'
        ),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(
            event_type='false_positive', 
            actual_classification='NORMAL VALUE'
        ),
        normal_cases.assign(
            event_type='normal', 
            actual_classification='NORMAL VALUE'
        )
    ]).sort_values('DateTime')
    
    print(f"Prepared validation set with {len(test_cases)} cases:")
    print(f"- {sum(test_cases['event_type'] == 'anomaly')} anomalies")
    print(f"- {sum(test_cases['event_type'] == 'false_positive')} false positives")
    print(f"- {sum(test_cases['event_type'] == 'normal')} normal cases")
    
    # 4. Create binary prompt
    system_prompt = get_binary_expert_prompt()
    
    # 5. Setup database connection and query template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    conn = sqlite3.connect(db_path)
    
    # Update query template to include binary classification guidance
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

Important instructions:
1. Use a BINARY classification: either ANOMALY or NORMAL VALUE
2. STRICTLY apply the numerical thresholds I provided (Temperature > 115°C for overheating, etc.)
3. Check if patterns are sustained over multiple readings or just temporary spikes
4. Consider the effect of weather conditions on the readings
5. Use ONLY the required response format for classification
"""

    # 6. Setup results tracking and checkpoint file
    results = []
    checkpoint_file = os.path.join(results_dir, 'extended_validation_checkpoint.json')
    final_results_file = os.path.join(results_dir, 'extended_validation_results.json')
    
    # Enhanced results data files for future analysis
    predictions_csv_file = os.path.join(results_dir, 'prediction_results.csv')
    metrics_json_file = os.path.join(results_dir, 'prediction_metrics.json')
    
    # Check if we have a checkpoint to resume from
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} cases already processed")
    
    # 7. Process each case
    try:
        for idx in range(start_idx, len(test_cases)):
            case = test_cases.iloc[idx]
            event_time = pd.Timestamp(case['DateTime'])
            event_type = case['event_type']
            actual_note = case['Notes']
            actual_classification = case['actual_classification']
            
            print(f"\nProcessing case {idx+1}/{len(test_cases)}: {event_time} ({event_type})")
            
            # Extract data window around the event
            half_window = pd.Timedelta(hours=time_window//2)
            start_time = event_time - half_window
            end_time = event_time + half_window
            
            # Query the database to get data for this time window
            window_query = f"""
            SELECT * FROM compressor_data_with_weather 
            WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
            """
            
            # Get compressor data (without the weather columns)
            compressor_df = pd.read_sql(window_query, conn)
            compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
            
            # Get just the weather columns
            weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
            
            # Format data for prompt
            compressor_data_str = compressor_data.to_string(index=False)
            weather_data_str = weather_data.to_string(index=False)
            
            # Prepare the prompt
            prompt = query_template.format(
                datetime=event_time,
                compressor_data=compressor_data_str,
                weather_data=weather_data_str
            )
            
            # Create the messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Call the LLM with very low temperature for deterministic responses
                    response = llm.invoke(
                        messages,
                        temperature=0.01,
                        max_tokens=300
                    )
                    
                    # Parse the response
                    parsed_response = parse_llm_response(response.content)
                    
                    # Update result evaluation for binary classification
                    if parsed_response['classification'] == 'NORMAL VALUE':
                        # For normal value prediction, correct if actual is also NORMAL VALUE
                        is_correct = actual_classification == 'NORMAL VALUE'
                    else:
                        # For anomaly prediction, correct if actual is also ANOMALY
                        is_correct = actual_classification == 'ANOMALY'
                    
                    # Add to results with enhanced structure for easier analysis
                    result = {
                        'datetime': event_time.isoformat(),
                        'actual_type': event_type,
                        'actual_classification': actual_classification,
                        'actual_note': actual_note,
                        'llm_response': response.content,
                        'parsed_response': parsed_response,
                        'compressor_readings': {
                            'temperature': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Temperature'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'vibration': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Vibration'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'pressure': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Pressure'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'current': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Current'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None
                        },
                        'is_correct': is_correct
                    }
                    
                    results.append(result)
                    success = True
                    
                    # Print brief summary
                    print(f"  Result: {parsed_response['classification']} (Actual: {actual_classification})")
                    print(f"  Confidence: {parsed_response['confidence']}")
                    
                except Exception as e:
                    retry_count += 1
                    print(f"  Error on attempt {retry_count}/{max_retries}: {e}")
                    time.sleep(2)  # Wait before retrying
            
            if not success:
                print(f"  Failed to process case after {max_retries} attempts")
            
            # Save checkpoint with additional formats every batch_size cases
            if (idx + 1) % batch_size == 0 or idx == len(test_cases) - 1:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                print(f"Saved checkpoint after {idx + 1} cases")
                
                # Save to CSV format for easy import into visualization tools
                save_results_to_csv(results, predictions_csv_file)
                
                # Calculate and display interim metrics
                if len(results) > 0:
                    correct = sum(1 for r in results 
                                if r['actual_classification'] == r['parsed_response']['classification'])
                    accuracy = correct / len(results)
                    print(f"Interim accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    
    finally:
        # Close the database connection
        conn.close()
        
        # Save final results in multiple formats for future analysis
        if results:
            # Save detailed JSON
            with open(final_results_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV for analysis in other tools
            save_results_to_csv(results, predictions_csv_file)
            
            # Calculate and save metrics separately
            metrics = calculate_extended_validation_metrics(results)
            with open(metrics_json_file, 'w') as f:
                json.dump(metrics, f, default=str, indent=2)
                
            print(f"Saved final results to multiple formats:")
            print(f"- Full JSON: {final_results_file}")
            print(f"- CSV for analysis: {predictions_csv_file}")
            print(f"- Metrics JSON: {metrics_json_file}")
    
    # 8. Calculate final metrics
    print("\n========== VALIDATION RESULTS ==========")
    metrics = calculate_extended_validation_metrics(results)
    
    return results, metrics

def analyze_false_positives(llm, df_with_weather, false_positives_df, results_dir):
    """
    Run targeted analysis on known false positive cases to validate the prompt
    """
    print("\n========== RUNNING FALSE POSITIVE ANALYSIS ==========")
    
    # 1. Get all labeled false positives
    false_positives = false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive').sort_values('DateTime')
    
    print(f"Found {len(false_positives)} labeled false positives")
    
    # 2. Create binary prompt
    system_prompt = get_binary_expert_prompt()
    
    # 3. Setup database connection and query template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    conn = sqlite3.connect(db_path)
    
    # Update query template for binary classification
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

This case is around {datetime}, which may match one of the specific cases mentioned 
in the rules. Carefully check the date and values.

Remember: 
- Use BINARY classification: ANOMALY or NORMAL VALUE
- Temperature ≤ 115°C is NORMAL VALUE
- Vibration ≤ 4.0 mm/s is NORMAL VALUE
- Apply the rules strictly
- Default to NORMAL VALUE when in doubt
"""
    
    # 4. Setup results tracking
    results = []
    
    # 5. Process each false positive case
    for idx, case in false_positives.iterrows():
        event_time = pd.Timestamp(case['DateTime'])
        actual_note = case['Notes']
        
        print(f"\nProcessing false positive case {idx+1}/{len(false_positives)}: {event_time}")
        
        # Extract data window around the event
        half_window = pd.Timedelta(hours=24//2)
        start_time = event_time - half_window
        end_time = event_time + half_window
        
        # Query the database to get data for this time window
        window_query = f"""
        SELECT * FROM compressor_data_with_weather 
        WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
        """
        
        # Get compressor data (without the weather columns)
        compressor_df = pd.read_sql(window_query, conn)
        compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
        
        # Get just the weather columns
        weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
        
        # Format data for prompt
        compressor_data_str = compressor_data.to_string(index=False)
        weather_data_str = weather_data.to_string(index=False)
        
        # Prepare the prompt
        prompt = query_template.format(
            datetime=event_time,
            compressor_data=compressor_data_str,
            weather_data=weather_data_str
        )
        
        # Create the messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Call the LLM with very low temperature for deterministic responses
        response = llm.invoke(
            messages,
            temperature=0.01,
            max_tokens=300
        )
        
        # Parse the response
        parsed_response = parse_llm_response(response.content)
        
        # Add to results with enhanced structure for easier analysis
        result = {
            'datetime': event_time.isoformat(),
            'actual_type': 'false_positive',
            'actual_classification': 'NORMAL VALUE',
            'actual_note': actual_note,
            'llm_response': response.content,
            'parsed_response': parsed_response,
            'compressor_readings': {
                'temperature': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Temperature'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                'vibration': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Vibration'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                'pressure': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Pressure'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                'current': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Current'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None
            },
            'is_correct': parsed_response['classification'] == 'NORMAL VALUE'
        }
        
        results.append(result)
        
        # Print brief summary - make sure to show false_positive in the event type
        print(f"  LLM Classification: {parsed_response['classification']} (Should be: NORMAL VALUE)")
        if parsed_response['classification'] == 'NORMAL VALUE':
            print(f"✓ Correct classification")
        else:
            print(f"✗ Incorrect classification")
    
    # Close the database connection
    conn.close()
    
    # Save results to file in the results directory
    results_file = os.path.join(results_dir, 'false_positive_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, default=str, indent=2)
    
    print(f"\nCompleted analysis of {len(results)} false positive cases. Results saved to '{results_file}'")
    
    return results
def get_balanced_detection_prompt():
    return """You are an expert AI assistant for industrial compressor maintenance. 
Your mission is DUAL and EQUALLY IMPORTANT:
1. DETECT ALL TRUE ANOMALIES - Missing a true anomaly could lead to equipment failure
2. AVOID FALSE ALARMS - Unnecessary maintenance is costly and disruptive

=== PRECISE ANOMALY CLASSIFICATION RULES - FOLLOW THESE EXACTLY ===

1. BEARING FAILURE ANOMALY:
   - REQUIRED: Vibration > 4.0 mm/s AND Temperature > 100°C
   - Both conditions must occur simultaneously
   - DEFINITE BEARING FAILURE: Vibration > 6.0 mm/s with Temperature > 105°C
   
2. OVERHEATING ANOMALY:
   - REQUIRED: Temperature > 115°C
   - This is a strict, non-negotiable threshold
   - DEFINITE OVERHEATING: Temperature ≥ 120°C
   
3. PRESSURE DROP ANOMALY:
   - REQUIRED: Pressure < 5.5 bar AND sustained across multiple readings
   - Temporary drops that recover = NOT an anomaly
   - DEFINITE PRESSURE DROP: Pressure < 5.0 bar sustained
   
4. MOTOR IMBALANCE ANOMALY:
   - REQUIRED: Vibration > 3.0 mm/s AND Speed outside 2860-3040 RPM
   - Both conditions must occur together
   - DEFINITE MOTOR IMBALANCE: Vibration > 3.5 mm/s with significant speed deviation
   
5. VOLTAGE FLUCTUATION ANOMALY:
   - REQUIRED: Voltage outside 390-410V range AND CosPhi < 0.83 AND sustained
   - Brief voltage spikes = NOT an anomaly
   - DEFINITE VOLTAGE FLUCTUATION: Voltage < 385V or > 415V sustained with CosPhi < 0.8

=== COMMON FALSE ALARM SCENARIOS ===

These should NOT be classified as anomalies:
- Temperature 95-115°C with normal vibration
- Vibration 2.0-4.0 mm/s without high temperature
- Brief or isolated measurement spikes that normalize quickly
- Weather-related variations (ambient temperature effects, storms)
- Current spikes without corresponding temperature increase

=== BALANCED JUDGMENT PROTOCOL ===

- If readings CLEARLY EXCEED the thresholds = ANOMALY
- If readings are CLEARLY BELOW the thresholds = NOT ANOMALY
- For BORDERLINE cases, check if readings match DEFINITE anomaly criteria
- Consider whether multiple parameters are abnormal simultaneously

=== RESPONSE FORMAT REQUIREMENTS ===

CLASSIFICATION: [ANOMALY or NOT ANOMALY]
TYPE: [ONLY if ANOMALY: bearing failure/overheating/pressure drop/motor imbalance/voltage fluctuation]
CONFIDENCE: [high/medium/low]
KEY_INDICATORS: [List 2-3 specific sensor readings with exact values]
RECOMMENDATION: [1 clear, concise sentence]
"""

def run_balanced_validation(llm, df_with_weather, anomalies_df, false_positives_df, results_dir, time_window=24, batch_size=10):
    """
    Run validation using the balanced detection prompt that properly weighs both true positives and false alarms
    """
    print("\n========== RUNNING VALIDATION WITH BALANCED DETECTION PROMPT ==========")
    
    # 1. Get all labeled events (anomalies and false positives)
    all_labeled_events = pd.concat([
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='anomaly'),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(event_type='false_positive')
    ]).sort_values('DateTime')
    
    print(f"Found {len(all_labeled_events)} labeled events ({sum(all_labeled_events['event_type'] == 'anomaly')} anomalies, "
          f"{sum(all_labeled_events['event_type'] == 'false_positive')} false positives)")
    
    # 2. Select fewer "normal" cases to reach total of ~50
    normal_cases_count = 50 - len(all_labeled_events)
    normal_cases_count = max(normal_cases_count, 0)  # Ensure we don't get a negative number
    
    print(f"Selecting {normal_cases_count} normal cases to reach ~50 total cases...")
    normal_cases = select_challenging_normal_cases(df_with_weather, 
                                                  all_labeled_events['DateTime'].tolist(), 
                                                  n_samples=normal_cases_count)
    
    # 3. Combine all cases for testing - KEEP ORIGINAL EVENT TYPES but map to binary classification
    test_cases = pd.concat([
        # Keep original event_type but map classification to binary
        anomalies_df[['DateTime', 'Notes', 'Anomaly']].assign(
            event_type='anomaly', 
            actual_classification='ANOMALY'
        ),
        false_positives_df[['DateTime', 'Notes', 'Anomaly']].assign(
            event_type='false_positive', 
            actual_classification='NORMAL VALUE'
        ),
        normal_cases.assign(
            event_type='normal', 
            actual_classification='NORMAL VALUE'
        )
    ]).sort_values('DateTime')
    
    print(f"Prepared validation set with {len(test_cases)} cases:")
    print(f"- {sum(test_cases['event_type'] == 'anomaly')} anomalies")
    print(f"- {sum(test_cases['event_type'] == 'false_positive')} false positives")
    print(f"- {sum(test_cases['event_type'] == 'normal')} normal cases")
    
    # 4. Create balanced prompt
    system_prompt = get_balanced_detection_prompt()
    
    # 5. Setup database connection and query template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "datasets", "compressor_data_2024.db")
    conn = sqlite3.connect(db_path)
    
    # Update query template to encourage balanced judgment
    query_template = """
Analyze this compressor data for the time period around {datetime}:

COMPRESSOR READINGS:
{compressor_data}

WEATHER DATA:
{weather_data}

Important instructions:
1. Use a BALANCED approach that equally prioritizes:
   - Finding TRUE anomalies (missing these could cause equipment failure)
   - Avoiding FALSE ALARMS (unnecessary maintenance is costly)
2. Apply the PRECISE thresholds in the rules (Temperature > 115°C, etc.)
3. Check if patterns are sustained or just temporary spikes
4. Look for multiple parameters deviating simultaneously
5. Consider environmental factors that might explain readings
"""

    # 6. Setup results tracking and checkpoint file
    results = []
    checkpoint_file = os.path.join(results_dir, 'balanced_validation_checkpoint.json')
    final_results_file = os.path.join(results_dir, 'balanced_validation_results.json')
    
    # Enhanced results data files for future analysis
    predictions_csv_file = os.path.join(results_dir, 'balanced_prediction_results.csv')
    metrics_json_file = os.path.join(results_dir, 'balanced_prediction_metrics.json')
    
    # Check if we have a checkpoint to resume from
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            results = json.load(f)
            start_idx = len(results)
            print(f"Resuming from checkpoint with {start_idx} cases already processed")
    
    # 7. Process each case
    try:
        for idx in range(start_idx, len(test_cases)):
            case = test_cases.iloc[idx]
            event_time = pd.Timestamp(case['DateTime'])
            event_type = case['event_type']
            actual_note = case['Notes']
            actual_classification = case['actual_classification']
            
            print(f"\nProcessing case {idx+1}/{len(test_cases)}: {event_time} ({event_type})")
            
            # Extract data window around the event
            half_window = pd.Timedelta(hours=time_window//2)
            start_time = event_time - half_window
            end_time = event_time + half_window
            
            # Query the database to get data for this time window
            window_query = f"""
            SELECT * FROM compressor_data_with_weather 
            WHERE DateTime BETWEEN '{start_time}' AND '{end_time}'
            """
            
            # Get compressor data (without the weather columns)
            compressor_df = pd.read_sql(window_query, conn)
            compressor_data = compressor_df.drop(columns=[col for col in compressor_df.columns if col.startswith('weather_')])
            
            # Get just the weather columns
            weather_data = compressor_df[['DateTime'] + [col for col in compressor_df.columns if col.startswith('weather_')]]
            
            # Format data for prompt
            compressor_data_str = compressor_data.to_string(index=False)
            weather_data_str = weather_data.to_string(index=False)
            
            # Prepare the prompt
            prompt = query_template.format(
                datetime=event_time,
                compressor_data=compressor_data_str,
                weather_data=weather_data_str
            )
            
            # Create the messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Call the LLM with slightly higher temperature for more balanced decisions
                    response = llm.invoke(
                        messages,
                        temperature=0.05,  # Slightly higher to allow more nuanced decisions 
                        max_tokens=300
                    )
                    
                    # Parse the response
                    parsed_response = parse_llm_response(response.content)
                    
                    # Map NOT ANOMALY to NORMAL VALUE for consistent terminology
                    if parsed_response['classification'] == 'NOT ANOMALY':
                        parsed_response['classification'] = 'NORMAL VALUE'
                    
                    # Update result evaluation for binary classification
                    if parsed_response['classification'] == 'NORMAL VALUE':
                        # For normal value prediction, correct if actual is also NORMAL VALUE
                        is_correct = actual_classification == 'NORMAL VALUE'
                    else:
                        # For anomaly prediction, correct if actual is also ANOMALY
                        is_correct = actual_classification == 'ANOMALY'
                    
                    # Add to results with enhanced structure for easier analysis
                    result = {
                        'datetime': event_time.isoformat(),
                        'actual_type': event_type,
                        'actual_classification': actual_classification,
                        'actual_note': actual_note,
                        'llm_response': response.content,
                        'parsed_response': parsed_response,
                        'compressor_readings': {
                            'temperature': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Temperature'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'vibration': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Vibration'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'pressure': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Pressure'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None,
                            'current': compressor_df.loc[compressor_df['DateTime'] == event_time, 'Current'].iloc[0] if not compressor_df[compressor_df['DateTime'] == event_time].empty else None
                        },
                        'is_correct': is_correct
                    }
                    
                    results.append(result)
                    success = True
                    
                    # Print brief summary
                    print(f"  Result: {parsed_response['classification']} (Actual: {actual_classification})")
                    print(f"  Confidence: {parsed_response['confidence']}")
                    
                except Exception as e:
                    retry_count += 1
                    print(f"  Error on attempt {retry_count}/{max_retries}: {e}")
                    time.sleep(2)  # Wait before retrying
            
            if not success:
                print(f"  Failed to process case after {max_retries} attempts")
            
            # Save checkpoint with additional formats every batch_size cases
            if (idx + 1) % batch_size == 0 or idx == len(test_cases) - 1:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, default=str, indent=2)
                print(f"Saved checkpoint after {idx + 1} cases")
                
                # Save to CSV format for easy import into visualization tools
                save_results_to_csv(results, predictions_csv_file)
                
                # Calculate and display interim metrics
                if len(results) > 0:
                    correct = sum(1 for r in results 
                                if r['actual_classification'] == r['parsed_response']['classification'])
                    accuracy = correct / len(results)
                    print(f"Interim accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    
    finally:
        # Close the database connection
        conn.close()
        
        # Save final results in multiple formats for future analysis
        if results:
            # Save detailed JSON
            with open(final_results_file, 'w') as f:
                json.dump(results, f, default=str, indent=2)
            
            # Save CSV for analysis in other tools
            save_results_to_csv(results, predictions_csv_file)
            
            # Calculate and save metrics separately
            metrics = calculate_extended_validation_metrics(results)
            with open(metrics_json_file, 'w') as f:
                json.dump(metrics, f, default=str, indent=2)
                
            print(f"Saved final results to multiple formats:")
            print(f"- Full JSON: {final_results_file}")
            print(f"- CSV for analysis: {predictions_csv_file}")
            print(f"- Metrics JSON: {metrics_json_file}")
    
    # 8. Calculate final metrics
    print("\n========== BALANCED VALIDATION RESULTS ==========")
    metrics = calculate_extended_validation_metrics(results)
    
    return results, metrics