import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pydot
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Set the Graphviz path for TensorFlow to use
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

def create_ann_model(input_shape):
    """Creates the ANN model architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
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
    """Generate a flowchart of the ANN architecture"""
    # Create output directory
    output_dir = os.path.join("test_predictions", "ann", "documentation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model with arbitrary input shape
    input_shape = 60  # This is just an example, actual shape depends on feature engineering
    model = create_ann_model(input_shape)
    
    # Print model summary
    model.summary()
    
    # Plot the model architecture
    plot_file = os.path.join(output_dir, "ann_model_architecture.png")
    plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True, 
               show_layer_activations=True, expand_nested=True, dpi=300)
    
    print(f"Model architecture flowchart saved to {plot_file}")
    
    # Create a more detailed flowchart with matplotlib
    create_detailed_flowchart(output_dir)
    
    # Create the integrated flowchart
    create_integrated_flowchart(output_dir, input_shape)
    
    # Create the technical flowchart
    create_technical_flowchart(output_dir, input_shape)
    
    # Add function to display model weights and biases
    print("Note: To see weights and biases, you need to load a trained model.")
    print("Run 'visualize_model_weights(model_path)' with the path to a trained model.")

def create_detailed_flowchart(output_dir):
    """Create a more detailed and customized flowchart of the ANN architecture"""
    fig, ax = plt.figure(figsize=(14, 12)), plt.gca()
    
    # Architecture components
    components = [
        {"name": "Input Layer", "neurons": "Input Shape (n)", "details": "Features from window-based statistics"},
        {"name": "Dense Layer", "neurons": "128 neurons", "details": "ReLU activation"},
        {"name": "Batch Normalization", "neurons": "", "details": "Normalizes activations"},
        {"name": "Dropout", "neurons": "30%", "details": "Prevents overfitting"},
        {"name": "Dense Layer", "neurons": "64 neurons", "details": "ReLU activation"},
        {"name": "Batch Normalization", "neurons": "", "details": "Normalizes activations"},
        {"name": "Dropout", "neurons": "30%", "details": "Prevents overfitting"},
        {"name": "Dense Layer", "neurons": "32 neurons", "details": "ReLU activation"},
        {"name": "Dense Layer", "neurons": "16 neurons", "details": "ReLU activation"},
        {"name": "Output Layer", "neurons": "1 neuron", "details": "Sigmoid activation (0-1 probability)"}
    ]
    
    # Improved layout parameters
    box_height = 0.6
    box_width = 4
    y_spacing = 1.5  # Increased spacing between components
    details_x_offset = 0.5  # Space between box and details text
    
    for i, comp in enumerate(components):
        y_pos = -i * y_spacing
        
        # Box color based on layer type
        if "Input" in comp["name"]:
            color = '#AED6F1'  # Light blue
        elif "Dropout" in comp["name"]:
            color = '#F5CBA7'  # Light orange
        elif "Batch" in comp["name"]:
            color = '#D7BDE2'  # Light purple
        elif "Output" in comp["name"]:
            color = '#A9DFBF'  # Light green
        else:
            color = '#D5D8DC'  # Light gray
        
        # Draw box with black border
        rect = plt.Rectangle((0, y_pos - box_height/2), box_width, box_height, 
                         facecolor=color, edgecolor='black', alpha=0.8, linewidth=1)
        ax.add_patch(rect)
        
        # Add layer name (centered)
        ax.text(box_width/2, y_pos, comp["name"], 
                ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Add neuron count (slightly below center)
        if comp["neurons"]:
            ax.text(box_width/2, y_pos - 0.15, comp["neurons"], 
                   ha='center', va='center', style='italic', fontsize=9)
        
        # Add details text outside the box (with background)
        details_x = box_width + details_x_offset
        # Add a light background for details text
        detail_bg = plt.Rectangle((details_x - 0.2, y_pos - 0.2), 5, 0.4, 
                              facecolor='#f8f9fa', alpha=0.7, edgecolor='#e9ecef', linewidth=0.5)
        ax.add_patch(detail_bg)
        
        ax.text(details_x, y_pos, comp["details"], 
               ha='left', va='center', fontsize=10)
        
        # Add connector lines between boxes
        if i > 0:
            ax.plot([box_width/2, box_width/2], 
                   [(-i+1) * -y_spacing + box_height/2, y_pos - box_height/2], 
                   'k-', lw=1.5, alpha=0.6)
    
    # Add training details in a separate box at the bottom
    training_info = [
        "Optimizer: Adam",
        "Loss Function: Binary Cross-Entropy",
        "Metrics: Accuracy, Precision, Recall",
        "Early Stopping: Monitor val_loss",
        "Class Weighting: Applied to handle imbalance"
    ]
    
    # Draw a box for training details
    training_y_start = -(len(components) + 0.5) * y_spacing
    training_height = len(training_info) * 0.5 + 0.5
    training_box = plt.Rectangle((0, training_y_start - training_height), 
                               box_width + 5, training_height, 
                               facecolor='#E8F8F5', edgecolor='#82E0AA', 
                               alpha=0.8, linewidth=1.5)
    ax.add_patch(training_box)
    
    # Add training details title
    ax.text(0.2, training_y_start - 0.5, "Training Configuration:", 
           ha='left', va='center', fontweight='bold', fontsize=11)
    
    # Add training details
    for i, info in enumerate(training_info):
        ax.text(0.4, training_y_start - (i + 1.2) * 0.5, 
               f"• {info}", ha='left', va='center', fontsize=10)
    
    # Set plot limits and remove axes
    ax.set_xlim(-1, box_width + 7)
    ax.set_ylim(training_y_start - training_height - 0.5, 1)
    ax.axis('off')
    
    # Set title
    plt.title("ANN Model Architecture for Predictive Maintenance", 
             fontsize=16, fontweight='bold', pad=20)
    
    # Save figure with tight layout
    detailed_file = os.path.join(output_dir, "ann_detailed_architecture.png")
    plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
    print(f"Detailed architecture flowchart saved to {detailed_file}")
    plt.close()

def create_integrated_flowchart(output_dir, input_shape):
    """Create an integrated flowchart combining technical structure with informative details"""
    fig, ax = plt.figure(figsize=(12, 16)), plt.gca()
    
    # Architecture components with shape information
    components = [
        {"name": "Input Layer", "shape": f"(None, {input_shape})", "details": "Window statistics features", 
         "color": '#AED6F1', "activation": None},
        {"name": "Dense Layer", "shape": "(None, 128)", "details": "Fully connected layer", 
         "color": '#D5D8DC', "activation": "ReLU"},
        {"name": "Batch Normalization", "shape": "(None, 128)", "details": "Normalize & stabilize", 
         "color": '#D7BDE2', "activation": None},
        {"name": "Dropout", "shape": "(None, 128)", "details": "30% dropout rate", 
         "color": '#F5CBA7', "activation": None},
        {"name": "Dense Layer", "shape": "(None, 64)", "details": "Dimensionality reduction", 
         "color": '#D5D8DC', "activation": "ReLU"},
        {"name": "Batch Normalization", "shape": "(None, 64)", "details": "Normalize & stabilize", 
         "color": '#D7BDE2', "activation": None},
        {"name": "Dropout", "shape": "(None, 64)", "details": "30% dropout rate", 
         "color": '#F5CBA7', "activation": None},
        {"name": "Dense Layer", "shape": "(None, 32)", "details": "Feature extraction", 
         "color": '#D5D8DC', "activation": "ReLU"},
        {"name": "Dense Layer", "shape": "(None, 16)", "details": "Further refinement", 
         "color": '#D5D8DC', "activation": "ReLU"},
        {"name": "Output Layer", "shape": "(None, 1)", "details": "Anomaly probability", 
         "color": '#A9DFBF', "activation": "Sigmoid"}
    ]
    
    # Layout parameters
    box_width = 3.2
    box_height = 0.8
    y_spacing = 1.5
    central_column_width = box_width + 3.0
    
    # Draw the components
    for i, comp in enumerate(components):
        y_pos = -i * y_spacing
        
        # Main component box
        main_box = plt.Rectangle(
            (central_column_width/2 - box_width/2, y_pos - box_height/2), 
            box_width, box_height,
            facecolor=comp["color"], 
            edgecolor='black', 
            alpha=0.9, 
            linewidth=1,
            zorder=3
        )
        ax.add_patch(main_box)
        
        # Layer name
        ax.text(
            central_column_width/2, 
            y_pos - 0.1, 
            comp["name"],
            ha='center', 
            va='center', 
            fontweight='bold', 
            fontsize=11,
            zorder=4
        )
        
        # Draw shape info on right side
        if comp["shape"]:
            shape_box = plt.Rectangle(
                (central_column_width/2 + box_width/2 + 0.3, y_pos - 0.3), 
                2.2, 0.6,
                facecolor='#f8f9fa', 
                edgecolor='#e9ecef', 
                alpha=0.9, 
                linewidth=0.8,
                zorder=3
            )
            ax.add_patch(shape_box)
            
            ax.text(
                central_column_width/2 + box_width/2 + 1.4, 
                y_pos, 
                comp["shape"],
                ha='center', 
                va='center', 
                fontsize=9,
                family='monospace',
                zorder=4
            )
        
        # Draw details info on left side
        if comp["details"]:
            details_box = plt.Rectangle(
                (central_column_width/2 - box_width/2 - 2.5, y_pos - 0.3), 
                2.2, 0.6,
                facecolor='#f8f9fa', 
                edgecolor='#e9ecef', 
                alpha=0.9, 
                linewidth=0.8,
                zorder=3
            )
            ax.add_patch(details_box)
            
            ax.text(
                central_column_width/2 - box_width/2 - 1.4, 
                y_pos, 
                comp["details"],
                ha='center', 
                va='center', 
                fontsize=9,
                zorder=4
            )
        
        # Draw activation badge if present
        if comp["activation"]:
            act_box = plt.Rectangle(
                (central_column_width/2 - 0.6, y_pos + box_height/2 - 0.05), 
                1.2, 0.25,
                facecolor='#FCF3CF', 
                edgecolor='#F9E79F', 
                alpha=0.9, 
                linewidth=1,
                zorder=5,
                clip_on=False
            )
            ax.add_patch(act_box)
            
            ax.text(
                central_column_width/2, 
                y_pos + box_height/2 + 0.07, 
                comp["activation"],
                ha='center', 
                va='center', 
                fontsize=8,
                fontweight='bold',
                zorder=6
            )
        
        # Add connector arrows between boxes
        if i > 0:
            arrow = FancyArrowPatch(
                (central_column_width/2, (-i+1) * -y_spacing + box_height/2),
                (central_column_width/2, y_pos - box_height/2),
                connectionstyle="arc3,rad=0", 
                arrowstyle="-|>", 
                mutation_scale=15, 
                linewidth=1.5, 
                color='#5D6D7E',
                zorder=2
            )
            ax.add_patch(arrow)
    
    # Add compilation info box at the bottom
    comp_y_pos = -(len(components) + 1) * y_spacing
    comp_box = plt.Rectangle(
        (central_column_width/2 - box_width - 0.5, comp_y_pos - box_height),
        2*box_width + 1, box_height*1.5,
        facecolor='#EAEDED',
        edgecolor='#5D6D7E',
        alpha=0.9,
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(comp_box)
    
    # Add compilation information
    ax.text(
        central_column_width/2, 
        comp_y_pos - 0.3, 
        "Compilation Settings",
        ha='center', 
        va='center', 
        fontweight='bold', 
        fontsize=11,
        zorder=4
    )
    
    compilation_info = [
        "Optimizer: Adam",
        "Loss: Binary Cross-Entropy",
        "Metrics: Accuracy, Precision, Recall"
    ]
    
    for i, info in enumerate(compilation_info):
        ax.text(
            central_column_width/2, 
            comp_y_pos - 0.3 - (i+1)*0.35, 
            info,
            ha='center', 
            va='center', 
            fontsize=9,
            zorder=4
        )
    
    # Set plot limits and remove axes
    ax.set_xlim(-1, central_column_width + 4)
    ax.set_ylim(comp_y_pos - box_height - 1, 1)
    ax.axis('off')
    
    # Add title and subtitle
    plt.suptitle(
        "ANN Model Architecture for Predictive Maintenance", 
        fontsize=16, 
        fontweight='bold', 
        y=0.98
    )
    plt.figtext(
        0.5, 0.955, 
        "Integrated visualization showing layer structure, shapes, and details", 
        ha='center', 
        fontsize=12, 
        style='italic'
    )
    
    # Add legend
    legend_y = 0.93
    legend_x = 0.05
    legend_items = [
        {"label": "Input Layer", "color": '#AED6F1'},
        {"label": "Dense Layer", "color": '#D5D8DC'},
        {"label": "Batch Normalization", "color": '#D7BDE2'},
        {"label": "Dropout Layer", "color": '#F5CBA7'},
        {"label": "Output Layer", "color": '#A9DFBF'}
    ]
    
    for i, item in enumerate(legend_items):
        legend_box = plt.Rectangle(
            (legend_x, legend_y - i*0.02), 
            0.02, 0.015,
            facecolor=item["color"],
            edgecolor='black',
            linewidth=0.5,
            transform=fig.transFigure,
            figure=fig
        )
        fig.add_artist(legend_box)
        
        plt.figtext(
            legend_x + 0.025, 
            legend_y - i*0.02, 
            item["label"],
            fontsize=9
        )
    
    # Save figure with tight layout
    integrated_file = os.path.join(output_dir, "ann_integrated_architecture.png")
    plt.savefig(integrated_file, dpi=300, bbox_inches='tight')
    print(f"Integrated architecture flowchart saved to {integrated_file}")
    plt.close()

def create_technical_flowchart(output_dir, input_shape):
    """Create a technical flowchart with detailed layer information"""
    fig, ax = plt.figure(figsize=(14, 18)), plt.gca()  # Increased figure height for better spacing
    
    # Define architecture specs - note that this must match the model in create_ann_model
    # We'll calculate the parameter counts
    
    # For a Dense layer, weights = input_dim * output_dim, biases = output_dim
    layers = [
        {
            "name": "Input Layer", 
            "type": "input",
            "shape": f"?×{input_shape}", 
            "color": "#E8D0AA",  # Light brown
            "details": "Window statistics features"
        },
        {
            "name": "Dense Layer 1", 
            "type": "dense",
            "neurons": 128, 
            "activation": "ReLU", 
            "kernel": f"{input_shape}×128", 
            "bias": "128",
            "params": f"{input_shape*128 + 128:,}",
            "color": "#A9CCE3",  # Blue
            "details": "Feature extraction"
        },
        {
            "name": "Batch Normalization 1", 
            "type": "batch_norm",
            "shape": "?×128", 
            "params": "512",  # 4 * units (gamma, beta, moving_mean, moving_variance)
            "color": "#D2B4DE",  # Purple
            "details": "Normalize activations"
        },
        {
            "name": "Dropout Layer 1", 
            "type": "dropout",
            "rate": "30%", 
            "shape": "?×128", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        {
            "name": "Dense Layer 2", 
            "type": "dense",
            "neurons": 64, 
            "activation": "ReLU", 
            "kernel": "128×64", 
            "bias": "64",
            "params": "8,256",  # 128*64 + 64
            "color": "#A9CCE3",  # Blue
            "details": "Dimensionality reduction"
        },
        {
            "name": "Batch Normalization 2", 
            "type": "batch_norm",
            "shape": "?×64", 
            "params": "256",  # 4 * units
            "color": "#D2B4DE",  # Purple
            "details": "Normalize activations"
        },
        {
            "name": "Dropout Layer 2", 
            "type": "dropout",
            "rate": "30%", 
            "shape": "?×64", 
            "params": "0",
            "color": "#F5CBA7",  # Orange
            "details": "Prevents overfitting"
        },
        {
            "name": "Dense Layer 3", 
            "type": "dense",
            "neurons": 32, 
            "activation": "ReLU", 
            "kernel": "64×32", 
            "bias": "32",
            "params": "2,080",  # 64*32 + 32
            "color": "#A9CCE3",  # Blue
            "details": "Feature extraction"
        },
        {
            "name": "Dense Layer 4", 
            "type": "dense",
            "neurons": 16, 
            "activation": "ReLU", 
            "kernel": "32×16", 
            "bias": "16",
            "params": "528",  # 32*16 + 16
            "color": "#A9CCE3",  # Blue
            "details": "Further refinement"
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
    box_width = 5.0   # Increased width
    box_height = 1.6  # Increased height
    y_spacing = 2.4   # Increased spacing between boxes
    
    # Title
    plt.figtext(0.5, 0.97, "ANN Model Architecture - Technical Flowchart", 
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
        
        # Layer name - positioned higher in the box for better spacing
        ax.text(
            1 + box_width/2, 
            y_pos - box_height/2 + box_height * 0.85,  # Positioned higher
            layer["name"],
            ha='center', 
            va='center', 
            fontweight='bold', 
            fontsize=12
        )
        
        # Add specific details based on layer type
        if layer["type"] == "input":
            # For input layer - more vertical spacing
            ax.text(
                1 + box_width/2, 
                y_pos - 0.1,  # Adjusted position
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.4,  # Increased vertical gap
                f"{layer['details']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
        elif layer["type"] == "dense":
            # For dense layers - improved spacing
            ax.text(
                1 + box_width/4, 
                y_pos - 0.25,  # Adjusted position
                f"Neurons: {layer['neurons']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos - 0.25,  # Adjusted position
                f"Activation: {layer['activation']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/4, 
                y_pos + 0.25,  # Adjusted position
                f"Kernel: {layer['kernel']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + 3*box_width/4, 
                y_pos + 0.25,  # Adjusted position
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
            
        elif layer["type"] == "batch_norm":
            # For batch normalization layers - better spacing
            ax.text(
                1 + box_width/2, 
                y_pos - 0.25,  # Adjusted position
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.25,  # Adjusted position
                f"Parameters: {layer['params']} (γ, β, moving_mean, moving_var)",
                ha='center', 
                va='center', 
                fontsize=10
            )
            
        elif layer["type"] == "dropout":
            # For dropout layers - improved spacing
            ax.text(
                1 + box_width/2, 
                y_pos - 0.25,  # Adjusted position
                f"Rate: {layer['rate']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
            ax.text(
                1 + box_width/2, 
                y_pos + 0.25,  # Adjusted position
                f"Shape: {layer['shape']}",
                ha='center', 
                va='center', 
                fontsize=10
            )
        
        # Draw connectors between boxes
        if i > 0:
            ax.arrow(
                1 + box_width/2, y_pos - box_height/2 - 0.1,  # Start point
                0, -y_spacing + box_height + 0.2,  # Direction vector
                head_width=0.1, head_length=0.1, 
                fc='black', ec='black', 
                length_includes_head=True
            )
    
    # Legend for color codes
    legend_items = [
        {"label": "Input Layer", "color": "#E8D0AA"},
        {"label": "Dense Layer", "color": "#A9CCE3"},
        {"label": "Batch Normalization", "color": "#D2B4DE"},
        {"label": "Dropout Layer", "color": "#F5CBA7"},
        {"label": "Output Layer", "color": "#A9DFBF"}
    ]
    
    legend_x = 0.07
    legend_y = 0.94
    for i, item in enumerate(legend_items):
        legend_box = plt.Rectangle(
            (legend_x, legend_y - i*0.03),  # Increased vertical spacing
            0.025, 0.018,
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
        box_width, 1.2,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=1.5
    )
    ax.add_patch(compile_box)
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.9, 
        "Compilation Settings",
        ha='center', 
        va='center', 
        fontweight='bold', 
        fontsize=11
    )
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.6, 
        "Optimizer: Adam",
        ha='center', 
        va='center', 
        fontsize=10
    )
    
    ax.text(
        1 + box_width/2, 
        y_bottom + 0.3, 
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
    technical_file = os.path.join(output_dir, "ann_technical_flowchart.png")
    plt.savefig(technical_file, dpi=300, bbox_inches='tight')
    print(f"Technical flowchart saved to {technical_file}")
    plt.close()

def visualize_model_weights(model_path, output_dir=None):
    """Load a trained model and visualize its weights and biases"""
    if output_dir is None:
        output_dir = os.path.join("test_predictions", "ann", "documentation")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Get summary of weights and biases
        print("\n=== Model Weights and Biases Summary ===")
        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        
        # Create a figure to visualize weights
        plt.figure(figsize=(15, 10))
        
        # Plot weight distributions for each Dense layer
        for i, layer in enumerate(dense_layers):
            weights, biases = layer.get_weights()
            
            # Plot weight distribution
            plt.subplot(len(dense_layers), 2, i*2 + 1)
            plt.hist(weights.flatten(), bins=50, alpha=0.7)
            plt.title(f"Layer {i+1}: {layer.name} Weights")
            plt.xlabel("Weight Value")
            plt.ylabel("Count")
            
            # Plot bias distribution
            plt.subplot(len(dense_layers), 2, i*2 + 2)
            plt.hist(biases.flatten(), bins=20, alpha=0.7, color='orange')
            plt.title(f"Layer {i+1}: {layer.name} Biases")
            plt.xlabel("Bias Value")
            plt.ylabel("Count")
            
            # Print statistics about weights and biases without using input_shape/output_shape directly
            print(f"\nLayer {i+1}: {layer.name}")
            # Use weights shape to infer input and output dimensions
            print(f"  Input dimension: {weights.shape[0]}")
            print(f"  Output dimension: {weights.shape[1]}")
            print(f"  Weights shape: {weights.shape}")
            print(f"  Biases shape: {biases.shape}")
            print(f"  Weight stats - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
            print(f"  Bias stats - Min: {biases.min():.4f}, Max: {biases.max():.4f}, Mean: {biases.mean():.4f}")
        
        plt.tight_layout()
        weights_file = os.path.join(output_dir, "ann_weights_distribution.png")
        plt.savefig(weights_file, dpi=300)
        plt.close()
        print(f"\nWeight distributions saved to {weights_file}")
        
        # Create matrices visualization for the first few layers
        for i, layer in enumerate(dense_layers[:3]):  # Limit to first 3 dense layers to avoid too many plots
            weights, biases = layer.get_weights()
            
            # Only visualize if it's not too large
            if weights.shape[0] <= 128 and weights.shape[1] <= 128:
                plt.figure(figsize=(10, 8))
                plt.imshow(weights, cmap='viridis', aspect='auto')  # Using imshow instead of matshow
                plt.colorbar()
                plt.title(f"Layer {i+1}: {layer.name} Weight Matrix")
                plt.xlabel("Output Neurons")
                plt.ylabel("Input Neurons")
                
                matrix_file = os.path.join(output_dir, f"ann_weight_matrix_layer{i+1}.png")
                plt.savefig(matrix_file, dpi=300)
                plt.close()
                print(f"Weight matrix for layer {i+1} saved to {matrix_file}")
            else:
                print(f"Weight matrix for layer {i+1} is too large to visualize: {weights.shape}")
        
        # Save a single bias visualization
        plt.figure(figsize=(12, 6))
        for i, layer in enumerate(dense_layers):
            weights, biases = layer.get_weights()
            plt.subplot(1, len(dense_layers), i+1)
            plt.bar(range(len(biases)), biases)
            plt.title(f"Layer {i+1} Biases")
            plt.xlabel("Neuron Index")
            plt.ylabel("Bias Value")
        
        plt.tight_layout()
        biases_file = os.path.join(output_dir, "ann_biases.png")
        plt.savefig(biases_file, dpi=300)
        plt.close()
        print(f"Bias values saved to {biases_file}")
        
        return True
    except Exception as e:
        print(f"Error loading or visualizing model: {e}")
        return False

if __name__ == "__main__":
    visualize_model_architecture()
    print("\nFlow chart generation complete!")
    print("The flowchart visualizes the following architecture:")
    print("1. Input Layer: Features from window-based statistics")
    print("2. Dense Layer: 128 neurons with ReLU activation")
    print("3. Batch Normalization Layer")
    print("4. Dropout Layer (30% dropout rate)")
    print("5. Dense Layer: 64 neurons with ReLU activation")
    print("6. Batch Normalization Layer")
    print("7. Dropout Layer (30% dropout rate)")
    print("8. Dense Layer: 32 neurons with ReLU activation")
    print("9. Dense Layer: 16 neurons with ReLU activation")
    print("10. Output Layer: 1 neuron with Sigmoid activation for binary classification")
    print("\nModel is compiled with:")
    print("- Adam optimizer")
    print("- Binary Cross-Entropy loss function")
    print("- Metrics: Accuracy, Precision, Recall")
    
    # Add optional weight visualization if model path is provided
    import argparse
    parser = argparse.ArgumentParser(description="Visualize ANN model architecture and optionally weights")
    parser.add_argument("--model_path", help="Path to a trained model file to visualize weights")
    args = parser.parse_args()
    
    if args.model_path:
        visualize_model_weights(args.model_path)
