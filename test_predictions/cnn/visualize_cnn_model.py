import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pydot
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Set the Graphviz path for TensorFlow to use
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

def create_cnn_model(input_shape):
    """Crea un modello CNN 1D per serie temporali"""
    model = Sequential([
        # Primo blocco convoluzionale
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Secondo blocco convoluzionale
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Terzo blocco convoluzionale
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Flatten e Dense finale
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def visualize_model_architecture():
    """Generate a flowchart of the CNN architecture"""
    # Create output directory
    output_dir = os.path.join("test_predictions", "cnn", "documentation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model with arbitrary input shape
    # CNN models typically use (sequence_length, features) shape
    sequence_length = 24  # Esempio, la lunghezza della sequenza temporale
    n_features = 12      # Esempio, il numero di feature per ogni timestep
    input_shape = (sequence_length, n_features)
    model = create_cnn_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Plot the model architecture using TensorFlow's built-in utility
    plot_file = os.path.join(output_dir, "cnn_model_architecture.png")
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True, 
               show_layer_activations=True, expand_nested=True, dpi=300)
    
    print(f"Model architecture flowchart saved to {plot_file}")
    
    # Create the technical flowchart
    create_technical_flowchart(output_dir, input_shape)
    
    # Add function to display model weights and biases
    print("Note: To see weights and biases, you need to load a trained model.")
    print("Run 'visualize_model_weights(model_path)' with the path to a trained model.")

def create_technical_flowchart(output_dir, input_shape):
    """Create a technical flowchart with detailed CNN layer information"""
    fig, ax = plt.figure(figsize=(14, 20)), plt.gca()  # Increased height for better spacing
    
    # Define architecture specs based on the CNN model
    # Calculate parameters for CNN layers:
    # Conv1D params = filters * (kernel_size * input_channels + 1)
    # Where +1 is for the bias term
    
    sequence_length, n_features = input_shape
    
    # Define all layers with their specifications
    layers = [
        {
            "name": "Input Layer", 
            "type": "input",
            "shape": f"batch×{sequence_length}×{n_features}", 
            "color": "#E8D0AA",  # Light brown
            "details": "Time series data"
        },
        # First Convolutional Block
        {
            "name": "Conv1D Layer 1", 
            "type": "conv1d",
            "filters": 64, 
            "kernel_size": 3,
            "activation": "ReLU", 
            "padding": "same",
            "shape": f"batch×{sequence_length}×64",
            "params": f"{64 * (3 * n_features + 1):,}",  # filters * (kernel_size * input_channels + 1)
            "color": "#ABD7EB",  # Light blue
            "details": "Feature extraction"
        },
        {
            "name": "Batch Normalization 1", 
            "type": "batch_norm",
            "shape": f"batch×{sequence_length}×64", 
            "params": "256",  # 4 * output_channels
            "color": "#D2B4DE",  # Purple
            "details": "Normalize activations"
        },
        {
            "name": "MaxPooling1D 1", 
            "type": "pooling",
            "pool_size": 2,
            "shape": f"batch×{sequence_length//2}×64", 
            "params": "0",
            "color": "#FADBD8",  # Pink
            "details": "Reduce dimensionality"
        },
        {
            "name": "Dropout Layer 1", 
            "type": "dropout",
            "rate": "20%", 
            "shape": f"batch×{sequence_length//2}×64", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        # Second Convolutional Block
        {
            "name": "Conv1D Layer 2", 
            "type": "conv1d",
            "filters": 128, 
            "kernel_size": 3,
            "activation": "ReLU", 
            "padding": "same",
            "shape": f"batch×{sequence_length//2}×128",
            "params": f"{128 * (3 * 64 + 1):,}",  # filters * (kernel_size * input_channels + 1)
            "color": "#ABD7EB",  # Light blue
            "details": "Feature extraction"
        },
        {
            "name": "Batch Normalization 2", 
            "type": "batch_norm",
            "shape": f"batch×{sequence_length//2}×128", 
            "params": "512",  # 4 * output_channels
            "color": "#D2B4DE",  # Purple
            "details": "Normalize activations"
        },
        {
            "name": "MaxPooling1D 2", 
            "type": "pooling",
            "pool_size": 2,
            "shape": f"batch×{sequence_length//4}×128", 
            "params": "0",
            "color": "#FADBD8",  # Pink
            "details": "Reduce dimensionality"
        },
        {
            "name": "Dropout Layer 2", 
            "type": "dropout",
            "rate": "20%", 
            "shape": f"batch×{sequence_length//4}×128", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        # Third Convolutional Block
        {
            "name": "Conv1D Layer 3", 
            "type": "conv1d",
            "filters": 64, 
            "kernel_size": 3,
            "activation": "ReLU", 
            "padding": "same",
            "shape": f"batch×{sequence_length//4}×64",
            "params": f"{64 * (3 * 128 + 1):,}",  # filters * (kernel_size * input_channels + 1)
            "color": "#ABD7EB",  # Light blue
            "details": "Feature extraction"
        },
        {
            "name": "Batch Normalization 3", 
            "type": "batch_norm",
            "shape": f"batch×{sequence_length//4}×64", 
            "params": "256",  # 4 * output_channels
            "color": "#D2B4DE",  # Purple
            "details": "Normalize activations"
        },
        {
            "name": "MaxPooling1D 3", 
            "type": "pooling",
            "pool_size": 2,
            "shape": f"batch×{sequence_length//8}×64", 
            "params": "0",
            "color": "#FADBD8",  # Pink
            "details": "Reduce dimensionality"
        },
        {
            "name": "Dropout Layer 3", 
            "type": "dropout",
            "rate": "20%", 
            "shape": f"batch×{sequence_length//8}×64", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        # Flatten and Dense Layers
        {
            "name": "Flatten Layer", 
            "type": "flatten",
            "shape": f"batch×{(sequence_length//8) * 64}", 
            "params": "0",
            "color": "#ABEBC6",  # Light green
            "details": "Convert to 1D for dense layers"
        },
        {
            "name": "Dense Layer 1", 
            "type": "dense",
            "neurons": 64, 
            "activation": "ReLU", 
            "kernel": f"{(sequence_length//8) * 64}×64", 
            "bias": "64",
            "params": f"{(sequence_length//8) * 64 * 64 + 64:,}",
            "color": "#A9CCE3",  # Blue
            "details": "Feature combination"
        },
        {
            "name": "Dropout Layer 4", 
            "type": "dropout",
            "rate": "30%", 
            "shape": "batch×64", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        {
            "name": "Dense Layer 2", 
            "type": "dense",
            "neurons": 32, 
            "activation": "ReLU", 
            "kernel": "64×32", 
            "bias": "32",
            "params": "2,080",  # 64*32 + 32
            "color": "#A9CCE3",  # Blue
            "details": "Feature refinement"
        },
        {
            "name": "Output Layer", 
            "type": "dense",
            "neurons": 1, 
            "activation": "Sigmoid", 
            "kernel": "32×1", 
            "bias": "1",
            "params": "33",  # 32*1 + 1
            "color": "#A9DFBF",  # Green
            "details": "Binary anomaly classification"
        }
    ]
    
    # Calculate total parameters
    total_params = sum(int(layer["params"].replace(",", "")) if "params" in layer else 0 for layer in layers)
    
    # Layout parameters
    box_width = 6.0    # Wider boxes
    box_height = 2.0   # Taller boxes
    y_spacing = 2.8    # More vertical spacing between boxes
    text_padding = 0.3 # Increased padding
    
    # Title
    plt.figtext(0.5, 0.97, "CNN Model Architecture - Technical Flowchart", 
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
        
        # Layer name - posizionato più in alto nel box
        ax.text(
            1 + box_width/2, 
            y_pos - box_height/2 + box_height * 0.8,
            layer["name"],
            ha='center', 
            va='center', 
            fontweight='bold', 
            fontsize=12
        )
        
        # Add specific details based on layer type
        if layer["type"] == "input":
            # For input layer
            ax.text(
                1 + box_width/2, 
                y_pos - 0.15,  # Adjusted for better spacing
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.5,  # Increased gap
                f"{layer['details']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
        elif layer["type"] == "conv1d":
            # For Conv1D layers - improved vertical spacing
            ax.text(
                1 + box_width/4, 
                y_pos - 0.35,  # More separation
                f"Filters: {layer['filters']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos - 0.35,
                f"Kernel Size: {layer['kernel_size']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/4, 
                y_pos + 0.3,  # More separation
                f"Activation: {layer['activation']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos + 0.3,
                f"Padding: {layer['padding']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            # Parameters - more space to avoid overlap
            ax.text(
                1 + box_width/2, 
                y_pos - box_height/2 + box_height * 0.22,
                f"Trainable Parameters: {layer['params']}",
                ha='center', 
                va='center', 
                fontsize=10,
                style='italic'
            )
            
        elif layer["type"] == "pooling":
            # For pooling layers
            ax.text(
                1 + box_width/2, 
                y_pos - 0.2,
                f"Pool Size: {layer['pool_size']}",
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
            
        elif layer["type"] == "batch_norm":
            # For batch normalization layers
            ax.text(
                1 + box_width/2, 
                y_pos - 0.2,
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.2,
                f"Parameters: {layer['params']} (γ, β, moving_mean, moving_var)",
                ha='center', 
                va='center', 
                fontsize=10
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
            
        elif layer["type"] == "flatten":
            # For flatten layer
            ax.text(
                1 + box_width/2, 
                y_pos, 
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.4,
                f"{layer['details']}",
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
            # Parameters
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
                head_width=0.15, head_length=0.15,  # Freccia più grande
                fc='black', ec='black', 
                length_includes_head=True
            )
    
    # Legend for color codes
    legend_items = [
        {"label": "Input Layer", "color": "#E8D0AA"},
        {"label": "Conv1D Layer", "color": "#ABD7EB"},
        {"label": "MaxPooling1D", "color": "#FADBD8"},
        {"label": "Batch Normalization", "color": "#D2B4DE"},
        {"label": "Dropout Layer", "color": "#F5CBA7"},
        {"label": "Flatten Layer", "color": "#ABEBC6"},
        {"label": "Dense Layer", "color": "#A9CCE3"},
        {"label": "Output Layer", "color": "#A9DFBF"}
    ]
    
    legend_x = 0.07
    legend_y = 0.94
    for i, item in enumerate(legend_items):
        legend_box = plt.Rectangle(
            (legend_x, legend_y - i*0.03),  # Increased vertical gap
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
            legend_y - i*0.03,  # Matched spacing
            item["label"],
            fontsize=10
        )
    
    # Add compilation info at the bottom
    y_bottom = -(len(layers) * y_spacing + 1)
    compile_box = plt.Rectangle(
        (1, y_bottom),
        box_width, 1.6,  # Box più alto
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
    technical_file = os.path.join(output_dir, "cnn_technical_flowchart.png")
    plt.savefig(technical_file, dpi=300, bbox_inches='tight')
    print(f"Technical flowchart saved to {technical_file}")
    plt.close()

def visualize_model_weights(model_path, output_dir=None):
    """Load a trained model and visualize its weights and biases"""
    if output_dir is None:
        output_dir = os.path.join("test_predictions", "cnn", "documentation")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Get summary of weights and biases
        print("\n=== Model Weights and Biases Summary ===")
        
        # Extract all layers with weights
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        
        # Plot weight distributions for Conv1D layers
        if conv_layers:
            plt.figure(figsize=(15, len(conv_layers) * 4))
            for i, layer in enumerate(conv_layers):
                weights, biases = layer.get_weights()
                
                # Plot kernel weights distribution
                plt.subplot(len(conv_layers), 2, i*2 + 1)
                plt.hist(weights.flatten(), bins=50, alpha=0.7)
                plt.title(f"Conv1D Layer {i+1}: {layer.name} Kernel Weights")
                plt.xlabel("Weight Value")
                plt.ylabel("Count")
                
                # Plot bias distribution
                plt.subplot(len(conv_layers), 2, i*2 + 2)
                plt.hist(biases.flatten(), bins=20, alpha=0.7, color='orange')
                plt.title(f"Conv1D Layer {i+1}: {layer.name} Biases")
                plt.xlabel("Bias Value")
                plt.ylabel("Count")
                
                # Print statistics about weights and biases
                print(f"\nConv1D Layer {i+1}: {layer.name}")
                print(f"  Kernel shape: {weights.shape}")
                print(f"  Biases shape: {biases.shape}")
                print(f"  Weight stats - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
                print(f"  Bias stats - Min: {biases.min():.4f}, Max: {biases.max():.4f}, Mean: {biases.mean():.4f}")
            
            plt.tight_layout()
            conv_weights_file = os.path.join(output_dir, "cnn_conv_weights_distribution.png")
            plt.savefig(conv_weights_file, dpi=300)
            plt.close()
            print(f"\nConv1D weight distributions saved to {conv_weights_file}")
        
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
            dense_weights_file = os.path.join(output_dir, "cnn_dense_weights_distribution.png")
            plt.savefig(dense_weights_file, dpi=300)
            plt.close()
            print(f"\nDense layer weight distributions saved to {dense_weights_file}")
        
        return True
    except Exception as e:
        print(f"Error loading or visualizing model: {e}")
        return False

if __name__ == "__main__":
    visualize_model_architecture()
    print("\nFlow chart generation complete!")
    print("The flowchart visualizes the following CNN architecture:")
    print("1. Input Layer: Time series data")
    print("2. First Convolutional Block:")
    print("   - Conv1D: 64 filters, kernel size 3, ReLU activation")
    print("   - BatchNormalization")
    print("   - MaxPooling1D: pool size 2")
    print("   - Dropout: 20%")
    print("3. Second Convolutional Block:")
    print("   - Conv1D: 128 filters, kernel size 3, ReLU activation")
    print("   - BatchNormalization")
    print("   - MaxPooling1D: pool size 2")
    print("   - Dropout: 20%")
    print("4. Third Convolutional Block:")
    print("   - Conv1D: 64 filters, kernel size 3, ReLU activation")
    print("   - BatchNormalization")
    print("   - MaxPooling1D: pool size 2")
    print("   - Dropout: 20%")
    print("5. Flatten Layer")
    print("6. Dense Layer: 64 neurons, ReLU activation")
    print("7. Dropout: 30%")
    print("8. Dense Layer: 32 neurons, ReLU activation")
    print("9. Output Layer: 1 neuron, Sigmoid activation")
    print("\nModel is compiled with:")
    print("- Adam optimizer")
    print("- Binary Cross-Entropy loss function")
    print("- Metrics: Accuracy, Precision, Recall")
    
    # Add optional weight visualization if model path is provided
    import argparse
    parser = argparse.ArgumentParser(description="Visualize CNN model architecture and optionally weights")
    parser.add_argument("--model_path", help="Path to a trained model file to visualize weights")
    args = parser.parse_args()
    
    if args.model_path:
        visualize_model_weights(args.model_path)
