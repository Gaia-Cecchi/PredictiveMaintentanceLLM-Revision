import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pydot
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Set the Graphviz path for TensorFlow to use
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

def create_lstm_model(input_shape):
    """Creates the LSTM model architecture"""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def visualize_model_architecture():
    """Generate a flowchart of the LSTM architecture"""
    # Create output directory
    output_dir = os.path.join("test_predictions", "lstm", "documentation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model with arbitrary input shape
    # LSTM models typically use (sequence_length, features) shape
    sequence_length = 24  # Example sequence length
    n_features = 12       # Example number of features per timestep
    input_shape = (sequence_length, n_features)
    model = create_lstm_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Plot the model architecture using TensorFlow's built-in utility
    plot_file = os.path.join(output_dir, "lstm_model_architecture.png")
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True, 
               show_layer_activations=True, expand_nested=True, dpi=300)
    
    print(f"Model architecture flowchart saved to {plot_file}")
    
    # Create the technical flowchart
    create_technical_flowchart(output_dir, input_shape)
    
    # Add function to display model weights and biases
    print("Note: To see weights and biases, you need to load a trained model.")
    print("Run 'visualize_model_weights(model_path)' with the path to a trained model.")

def create_technical_flowchart(output_dir, input_shape):
    """Create a technical flowchart with detailed LSTM layer information"""
    fig, ax = plt.figure(figsize=(14, 18)), plt.gca()
    
    # Define architecture specs based on the LSTM model
    # Calculate parameters for LSTM layers:
    # LSTM params = 4 * ((input_dim + units + 1) * units)
    
    sequence_length, n_features = input_shape
    
    # Calculate LSTM layer parameters
    lstm1_params = 4 * ((n_features + 64 + 1) * 64)
    lstm2_params = 4 * ((64 + 32 + 1) * 32)
    
    # Define all layers with their specifications
    layers = [
        {
            "name": "Input Layer", 
            "type": "input",
            "shape": f"batch×{sequence_length}×{n_features}", 
            "color": "#E8D0AA",  # Light brown
            "details": "Time series data"
        },
        # First LSTM Layer with return_sequences=True
        {
            "name": "LSTM Layer 1", 
            "type": "lstm",
            "units": 64, 
            "activation": "ReLU", 
            "return_sequences": True,
            "shape": f"batch×{sequence_length}×64",
            "params": f"{lstm1_params:,}",
            "color": "#85C1E9",  # Blue
            "details": "Processes sequential data, return_sequences=True"
        },
        {
            "name": "Dropout Layer 1", 
            "type": "dropout",
            "rate": "20%", 
            "shape": f"batch×{sequence_length}×64", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        # Second LSTM Layer
        {
            "name": "LSTM Layer 2", 
            "type": "lstm",
            "units": 32, 
            "activation": "ReLU", 
            "return_sequences": False,
            "shape": f"batch×32",
            "params": f"{lstm2_params:,}",
            "color": "#85C1E9",  # Blue
            "details": "Processes sequential data, return_sequences=False"
        },
        {
            "name": "Dropout Layer 2", 
            "type": "dropout",
            "rate": "20%", 
            "shape": "batch×32", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        # Dense Layers
        {
            "name": "Dense Layer 1", 
            "type": "dense",
            "neurons": 16, 
            "activation": "ReLU", 
            "kernel": "32×16", 
            "bias": "16",
            "params": "528",  # 32*16 + 16
            "color": "#A9CCE3",  # Light blue
            "details": "Feature combination"
        },
        {
            "name": "Output Layer", 
            "type": "dense",
            "neurons": 1, 
            "activation": "Sigmoid", 
            "kernel": "16×1", 
            "bias": "1",
            "params": "17",  # 16*1 + 1
            "color": "#A9DFBF",  # Green
            "details": "Binary anomaly classification"
        }
    ]
    
    # Calculate total parameters
    total_params = sum(int(layer["params"].replace(",", "")) if "params" in layer else 0 for layer in layers)
    
    # Layout parameters
    box_width = 6.0   # Wider boxes
    box_height = 2.4  # Taller boxes
    y_spacing = 3.6   # More spacing between layers
    
    # Title
    plt.figtext(0.5, 0.97, "LSTM Model Architecture - Technical Flowchart", 
                ha='center', fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.955, f"Total Parameters: {total_params:,}", 
                ha='center', fontsize=13)
    
    # Draw the layers
    for i, layer in enumerate(layers):
        y_pos = -i * y_spacing
        
        # Main layer box
        rect = plt.Rectangle(
            (1, y_pos - box_height/2), 
            box_width, box_height,
            facecolor=layer["color"], 
            edgecolor='black', 
            alpha=0.9, 
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Layer name
        ax.text(
            1 + box_width/2, 
            y_pos - box_height/2 + box_height * 0.85,
            layer["name"],
            ha='center', 
            va='center', 
            fontweight='bold', 
            fontsize=12
        )
        
        # Add specific details based on layer type
        if layer["type"] == "input":
            # For input layer - improved spacing
            ax.text(
                1 + box_width/2, 
                y_pos - 0.2,  # Adjusted position
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.6,  # Increased vertical gap
                f"{layer['details']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
        elif layer["type"] == "lstm":
            # For LSTM layers - better spacing
            ax.text(
                1 + box_width/4, 
                y_pos - 0.4,  # Adjusted position
                f"Units: {layer['units']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos - 0.4,  # Matched position
                f"Activation: {layer['activation']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
            # LSTM-specific parameters - increased spacing
            return_seq_text = "return_sequences=True" if layer['return_sequences'] else "return_sequences=False"
            ax.text(
                1 + box_width/2, 
                y_pos + 0.0,  # Center position
                return_seq_text,
                ha='center', 
                va='center', 
                fontsize=10
            )
            
            ax.text(
                1 + box_width/2, 
                y_pos + 0.6,  # Increased gap
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
            ax.text(
                1 + box_width/2, 
                y_pos - box_height/2 + box_height * 0.25,
                f"Trainable Parameters: {layer['params']}",
                ha='center', 
                va='center', 
                fontsize=10,
                style='italic'
            )
            
        elif layer["type"] == "dropout":
            # For dropout layers
            ax.text(
                1 + box_width/2, 
                y_pos - 0.2,
                f"Rate: {layer['rate']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.2,
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
        elif layer["type"] == "dense":
            # For dense layers
            ax.text(
                1 + box_width/4, 
                y_pos - 0.2,
                f"Neurons: {layer['neurons']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos - 0.2,
                f"Activation: {layer['activation']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/4, 
                y_pos + 0.2,
                f"Kernel: {layer['kernel']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos + 0.2,
                f"Bias: {layer['bias']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos - box_height/2 + box_height * 0.25,
                f"Trainable Parameters: {layer['params']}",
                ha='center', 
                va='center', 
                fontsize=10,
                style='italic'
            )
        
        # Draw connectors between boxes
        if i > 0:
            ax.arrow(
                1 + box_width/2, y_pos - box_height/2 - 0.1,  # Start point
                0, -y_spacing + box_height + 0.2,  # Direction vector
                head_width=0.15, head_length=0.15,
                fc='black', ec='black', 
                length_includes_head=True
            )
    
    # Legend for color codes
    legend_items = [
        {"label": "Input Layer", "color": "#E8D0AA"},
        {"label": "LSTM Layer", "color": "#85C1E9"},
        {"label": "Dropout Layer", "color": "#F5CBA7"},
        {"label": "Dense Layer", "color": "#A9CCE3"},
        {"label": "Output Layer", "color": "#A9DFBF"}
    ]
    
    legend_x = 0.07
    legend_y = 0.94
    for i, item in enumerate(legend_items):
        legend_box = plt.Rectangle(
            (legend_x, legend_y - i*0.035),  # Increased vertical spacing
            0.025, 0.02,  # Slightly larger boxes
            facecolor=item["color"],
            edgecolor='black',
            linewidth=0.5,
            transform=fig.transFigure,
            figure=fig
        )
        fig.add_artist(legend_box)
        
        plt.figtext(
            legend_x + 0.035,
            legend_y - i*0.035,  # Matched spacing
            item["label"],
            fontsize=10
        )
    
    # Add LSTM cell explanation
    lstm_info_y = 0.85
    plt.figtext(0.07, lstm_info_y, "LSTM Cell Structure:", fontsize=10, fontweight='bold')
    plt.figtext(0.07, lstm_info_y - 0.035, "• Contains forget gate, input gate, cell state, and output gate", fontsize=9)
    plt.figtext(0.07, lstm_info_y - 0.07, "• 4× parameters of comparable Dense layer due to gate mechanisms", fontsize=9)
    plt.figtext(0.07, lstm_info_y - 0.105, "• Allows model to remember long-term dependencies in sequence data", fontsize=9)
    
    # Add compilation info at the bottom
    y_bottom = -(len(layers) * y_spacing + 1)
    compile_box = plt.Rectangle(
        (1, y_bottom),
        box_width, 1.6,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=1.5
    )
    ax.add_patch(compile_box)
    
    # Compilation info text
    ax.text(
        1 + box_width/2, 
        y_bottom + 1.2,
        "Compilation Settings",
        ha='center', 
        va='center', 
        fontweight='bold', 
        fontsize=11
    )
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.8,
        "Optimizer: Adam",
        ha='center', 
        va='center', 
        fontsize=10
    )
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.4,
        "Loss: Binary Cross-Entropy",
        ha='center', 
        va='center', 
        fontsize=10
    )
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.0,
        "Metrics: Accuracy, Precision, Recall",
        ha='center', 
        va='center', 
        fontsize=10
    )
    
    # Set axis limits and turn off axis
    ax.set_xlim(0, box_width + 2)
    ax.set_ylim(y_bottom - 0.5, 1)
    ax.axis('off')
    
    # Save figure
    technical_file = os.path.join(output_dir, "lstm_technical_flowchart.png")
    plt.savefig(technical_file, dpi=300, bbox_inches='tight')
    print(f"Technical flowchart saved to {technical_file}")
    plt.close()

def visualize_model_weights(model_path, output_dir=None):
    """Load a trained model and visualize its weights and biases"""
    if output_dir is None:
        output_dir = os.path.join("test_predictions", "lstm", "documentation")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Get summary of weights and biases
        print("\n=== Model Weights and Biases Summary ===")
        
        # Extract all layers with weights
        lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        
        # Plot weight distributions for LSTM layers
        if lstm_layers:
            plt.figure(figsize=(16, len(lstm_layers) * 6))
            for i, layer in enumerate(lstm_layers):
                weights = layer.get_weights()
                
                # LSTM weights consist of [kernel, recurrent_kernel, bias]
                kernel = weights[0]  # Input weights
                recurrent_kernel = weights[1]  # Recurrent weights
                bias = weights[2]  # Biases
                
                # Plot input weights (kernel) distribution
                plt.subplot(len(lstm_layers), 3, i*3 + 1)
                plt.hist(kernel.flatten(), bins=50, alpha=0.7)
                plt.title(f"LSTM Layer {i+1}: Input Weights")
                plt.xlabel("Weight Value")
                plt.ylabel("Count")
                
                # Plot recurrent weights distribution
                plt.subplot(len(lstm_layers), 3, i*3 + 2)
                plt.hist(recurrent_kernel.flatten(), bins=50, alpha=0.7, color='green')
                plt.title(f"LSTM Layer {i+1}: Recurrent Weights")
                plt.xlabel("Weight Value")
                plt.ylabel("Count")
                
                # Plot bias distribution
                plt.subplot(len(lstm_layers), 3, i*3 + 3)
                plt.hist(bias.flatten(), bins=20, alpha=0.7, color='orange')
                plt.title(f"LSTM Layer {i+1}: Biases")
                plt.xlabel("Bias Value")
                plt.ylabel("Count")
                
                # Print statistics about weights and biases
                # Infer dimensions from weights instead of using input_shape/output_shape
                input_dim = kernel.shape[0]  # Features dimension
                units = recurrent_kernel.shape[0]  # Number of LSTM units
                
                print(f"\nLSTM Layer {i+1}: {layer.name}")
                print(f"  Input features: {input_dim}")
                print(f"  LSTM units: {units}")
                print(f"  Input weights shape: {kernel.shape}")
                print(f"  Recurrent weights shape: {recurrent_kernel.shape}")
                print(f"  Bias shape: {bias.shape}")
                print(f"  Input weights stats - Min: {kernel.min():.4f}, Max: {kernel.max():.4f}, Mean: {kernel.mean():.4f}")
                print(f"  Recurrent weights stats - Min: {recurrent_kernel.min():.4f}, Max: {recurrent_kernel.max():.4f}, Mean: {recurrent_kernel.mean():.4f}")
                print(f"  Bias stats - Min: {bias.min():.4f}, Max: {bias.max():.4f}, Mean: {bias.mean():.4f}")
            
            plt.tight_layout()
            lstm_weights_file = os.path.join(output_dir, "lstm_weights_distribution.png")
            plt.savefig(lstm_weights_file, dpi=300)
            plt.close()
            print(f"\nLSTM weight distributions saved to {lstm_weights_file}")
        
        # Plot weight distributions for Dense layers
        if dense_layers:
            plt.figure(figsize=(15, len(dense_layers) * 4))
            for i, layer in enumerate(dense_layers):
                weights, biases = layer.get_weights()
                
                # Plot weight distribution
                plt.subplot(len(dense_layers), 2, i*2 + 1)
                plt.hist(weights.flatten(), bins=50, alpha=0.7)
                plt.title(f"Dense Layer {i+1}: {layer.name} Weights")
                plt.xlabel("Weight Value")
                plt.ylabel("Count")
                
                # Plot bias distribution
                plt.subplot(len(dense_layers), 2, i*2 + 2)
                plt.hist(biases.flatten(), bins=20, alpha=0.7, color='orange')
                plt.title(f"Dense Layer {i+1}: {layer.name} Biases")
                plt.xlabel("Bias Value")
                plt.ylabel("Count")
                
                # Print statistics about weights and biases
                print(f"\nDense Layer {i+1}: {layer.name}")
                print(f"  Input dimension: {weights.shape[0]}")
                print(f"  Output dimension: {weights.shape[1]}")
                print(f"  Weights shape: {weights.shape}")
                print(f"  Biases shape: {biases.shape}")
                print(f"  Weight stats - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
                print(f"  Bias stats - Min: {biases.min():.4f}, Max: {biases.max():.4f}, Mean: {biases.mean():.4f}")
            
            plt.tight_layout()
            dense_weights_file = os.path.join(output_dir, "lstm_dense_weights_distribution.png")
            plt.savefig(dense_weights_file, dpi=300)
            plt.close()
            print(f"\nDense layer weight distributions saved to {dense_weights_file}")
            
        # Create a visualization of LSTM gate distributions
        if lstm_layers:
            for i, layer in enumerate(lstm_layers):
                weights = layer.get_weights()
                kernel = weights[0]  # Input weights
                recurrent_kernel = weights[1]  # Recurrent weights
                bias = weights[2]  # Biases
                
                # The kernel, recurrent_kernel and bias are split into 4 parts for the LSTM gates
                # [forget gate, input gate, cell state, output gate]
                units = kernel.shape[1] // 4
                
                # Plot the distributions of weights for each gate
                plt.figure(figsize=(16, 13))  # Larger figure
                plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase spacing between subplots
                
                # Input kernel weights for each gate
                plt.subplot(3, 4, 1)
                plt.hist(kernel[:, :units].flatten(), bins=30, alpha=0.7)
                plt.title(f"Forget Gate Input Weights")
                
                plt.subplot(3, 4, 2)
                plt.hist(kernel[:, units:units*2].flatten(), bins=30, alpha=0.7)
                plt.title(f"Input Gate Input Weights")
                
                plt.subplot(3, 4, 3)
                plt.hist(kernel[:, units*2:units*3].flatten(), bins=30, alpha=0.7)
                plt.title(f"Cell State Input Weights")
                
                plt.subplot(3, 4, 4)
                plt.hist(kernel[:, units*3:].flatten(), bins=30, alpha=0.7)
                plt.title(f"Output Gate Input Weights")
                
                # Recurrent kernel weights for each gate
                plt.subplot(3, 4, 5)
                plt.hist(recurrent_kernel[:, :units].flatten(), bins=30, alpha=0.7, color='green')
                plt.title(f"Forget Gate Recurrent Weights")
                
                plt.subplot(3, 4, 6)
                plt.hist(recurrent_kernel[:, units:units*2].flatten(), bins=30, alpha=0.7, color='green')
                plt.title(f"Input Gate Recurrent Weights")
                
                plt.subplot(3, 4, 7)
                plt.hist(recurrent_kernel[:, units*2:units*3].flatten(), bins=30, alpha=0.7, color='green')
                plt.title(f"Cell State Recurrent Weights")
                
                plt.subplot(3, 4, 8)
                plt.hist(recurrent_kernel[:, units*3:].flatten(), bins=30, alpha=0.7, color='green')
                plt.title(f"Output Gate Recurrent Weights")
                
                # Bias values for each gate
                plt.subplot(3, 4, 9)
                plt.bar(range(units), bias[:units], color='orange')
                plt.title(f"Forget Gate Biases")
                
                plt.subplot(3, 4, 10)
                plt.bar(range(units), bias[units:units*2], color='orange')
                plt.title(f"Input Gate Biases")
                
                plt.subplot(3, 4, 11)
                plt.bar(range(units), bias[units*2:units*3], color='orange')
                plt.title(f"Cell State Biases")
                
                plt.subplot(3, 4, 12)
                plt.bar(range(units), bias[units*3:], color='orange')
                plt.title(f"Output Gate Biases")
                
                plt.suptitle(f"LSTM Layer {i+1} Gate Weights and Biases", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                
                gates_file = os.path.join(output_dir, f"lstm_layer{i+1}_gates.png")
                plt.savefig(gates_file, dpi=300)
                plt.close()
                print(f"LSTM gates visualization saved to {gates_file}")
        
        return True
    except Exception as e:
        print(f"Error loading or visualizing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    visualize_model_architecture()
    print("\nFlow chart generation complete!")
    print("The flowchart visualizes the following LSTM architecture:")
    print("1. Input Layer: Time series data with shape (sequence_length, features)")
    print("2. LSTM Layer 1: 64 units with ReLU activation, return_sequences=True")
    print("3. Dropout Layer: 20% dropout rate")
    print("4. LSTM Layer 2: 32 units with ReLU activation, return_sequences=False")
    print("5. Dropout Layer: 20% dropout rate")
    print("6. Dense Layer: 16 neurons with ReLU activation")
    print("7. Output Layer: 1 neuron with Sigmoid activation for binary classification")
    print("\nModel is compiled with:")
    print("- Adam optimizer")
    print("- Binary Cross-Entropy loss function")
    print("- Metrics: Accuracy, Precision, Recall")
    
    # Add optional weight visualization if model path is provided
    import argparse
    parser = argparse.ArgumentParser(description="Visualize LSTM model architecture and optionally weights")
    parser.add_argument("--model_path", help="Path to a trained model file to visualize weights")
    args = parser.parse_args()
    
    # Check if a specific model path was provided
    if args.model_path:
        model_path = args.model_path
    else:
        # If no path specified but we know the model path, use it
        model_path = "test_predictions/lstm/models/lstm_model_final_20250318_121712.h5"
        print(f"Using default model path: {model_path}")
    
    # Visualize the model weights
    if os.path.exists(model_path):
        visualize_model_weights(model_path)
    else:
        print(f"Model file not found at {model_path}")
        print("Please specify a valid model path with --model_path")
