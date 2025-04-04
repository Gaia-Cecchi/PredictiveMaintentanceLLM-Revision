import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

def visualize_llm_architecture(detail_level="intermediate"):
    """Generate a flowchart of the LLM (Qwen 2.5) architecture
    
    Args:
        detail_level (str): Level of detail: "simple", "intermediate", or "technical"
    """
    # Create output directory
    output_dir = os.path.join("test_predictions", "llm", "documentation")
    os.makedirs(output_dir, exist_ok=True)
    
    if detail_level == "simple":
        create_simplified_flowchart(output_dir)
    elif detail_level == "intermediate":
        create_intermediate_flowchart(output_dir)
    else:  # "technical" or any other value defaults to technical
        create_technical_flowchart(output_dir)
    
    print(f"LLM architecture flowchart saved to {output_dir}")

def create_simplified_flowchart(output_dir):
    """Create a simplified flowchart for LLM architecture"""
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    
    # Model specs
    model_name = "Qwen 2.5 32B"
    
    # Define simplified components
    layers = [
        {
            "name": "Input Text",
            "details": "Raw text converted to tokens",
            "color": "#FFD580"  # Light orange
        },
        {
            "name": "Embeddings",
            "details": "Words → Vectors + Position Info",
            "color": "#AED6F1"  # Light blue
        },
        {
            "name": "Transformer Blocks (40 layers)",
            "details": "Self-attention + Feed-forward networks",
            "color": "#D5F5E3"  # Light green
        },
        {
            "name": "Output Layer",
            "details": "Generates next token probabilities",
            "color": "#D7BDE2"  # Light purple
        }
    ]
    
    # Layout parameters
    box_width = 6.0
    box_height = 1.5
    y_spacing = 2.2
    
    # Title
    plt.figtext(0.5, 0.95, f"{model_name} - Simplified Architecture",
                ha='center', fontsize=16, fontweight='bold')
    
    # Draw the layers
    for i, layer in enumerate(layers):
        y_pos = 8 - i * y_spacing
        
        # Draw main box
        rect = plt.Rectangle(
            (3, y_pos - box_height/2),
            box_width, box_height,
            facecolor=layer["color"],
            edgecolor='black',
            alpha=0.9,
            linewidth=1.5,
            zorder=1
        )
        ax.add_patch(rect)
        
        # Layer name
        ax.text(
            3 + box_width/2,
            y_pos,
            layer["name"],
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=14
        )
        
        # Layer details
        ax.text(
            3 + box_width/2,
            y_pos + 0.4,
            layer["details"],
            ha='center',
            va='center',
            fontsize=12
        )
        
        # Add special details for transformer block
        if "Transformer Blocks" in layer["name"]:
            # Add a small diagram inside showing attention and FFN
            inner_box_width = box_width * 0.7
            inner_box_height = box_height * 0.4
            
            # Draw inner box representing the structure
            inner_box = plt.Rectangle(
                (3 + (box_width - inner_box_width)/2, y_pos - 0.4),
                inner_box_width, inner_box_height,
                facecolor="#F5F5F5",
                edgecolor='black',
                alpha=0.8,
                linewidth=1,
                zorder=2
            )
            ax.add_patch(inner_box)
            
            # Add simplified diagram of a transformer block
            attention_circle = plt.Circle(
                (3 + box_width/2 - inner_box_width/4, y_pos - 0.2),
                0.15,
                facecolor="#FFD700",  # Gold for attention
                edgecolor='black',
                alpha=0.8,
                linewidth=1,
                zorder=3
            )
            ax.add_patch(attention_circle)
            
            ffn_circle = plt.Circle(
                (3 + box_width/2 + inner_box_width/4, y_pos - 0.2),
                0.15,
                facecolor="#87CEFA",  # Light blue for FFN
                edgecolor='black',
                alpha=0.8,
                linewidth=1,
                zorder=3
            )
            ax.add_patch(ffn_circle)
            
            # Labels for inner components
            ax.text(
                3 + box_width/2 - inner_box_width/4,
                y_pos - 0.2,
                "A",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                zorder=4
            )
            
            ax.text(
                3 + box_width/2 + inner_box_width/4,
                y_pos - 0.2,
                "F",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                zorder=4
            )
            
            # Add legend
            ax.text(
                3 + box_width * 0.05,
                y_pos - box_height/2 + 0.25,
                "A = Attention    F = Feed-Forward",
                ha='left',
                va='center',
                fontsize=9,
                style='italic',
                zorder=4
            )
        
        # Draw connectors between boxes
        if i > 0:
            arrow = FancyArrowPatch(
                (3 + box_width/2, y_pos - box_height/2 - 0.1),
                (3 + box_width/2, (8 - (i-1) * y_spacing) + box_height/2 + 0.1),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2,
                color='#5D6D7E',
                zorder=0
            )
            ax.add_patch(arrow)
    
    # Add key capabilities box
    capabilities_box = plt.Rectangle(
        (10, 5),
        4, 3,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(capabilities_box)
    
    # Capabilities title
    ax.text(
        12,
        7.5,
        "Key Model Capabilities",
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=13
    )
    
    # Simplified capabilities list
    capabilities = [
        "32 billion parameters",
        "128K token context window",
        "Multilingual support",
        "Instruction following",
        "Knowledge cutoff: 2023"
    ]
    
    for i, capability in enumerate(capabilities):
        ax.text(
            12,
            7 - i * 0.4,
            capability,
            ha='center',
            va='center',
            fontsize=11
        )
    
    # Draw connector from transformer block to capabilities
    connector = FancyArrowPatch(
        (3 + box_width, 8 - 2 * y_spacing),
        (10, 6.5),
        connectionstyle="arc3,rad=0.2",
        arrowstyle="-",
        linestyle='--',
        mutation_scale=15,
        linewidth=1,
        color='#5D6D7E',
        zorder=0
    )
    ax.add_patch(connector)
    
    # Set axis limits and turn off axis
    ax.set_xlim(2, 15)
    ax.set_ylim(1, 9)
    ax.axis('off')
    
    # Save figure
    simplified_file = os.path.join(output_dir, "llm_qwen_simplified_flowchart.png")
    plt.savefig(simplified_file, dpi=300, bbox_inches='tight')
    print(f"Simplified LLM flowchart saved to {simplified_file}")
    plt.close()

def create_intermediate_flowchart(output_dir):
    """Create an intermediate flowchart for LLM architecture - balanced level of detail"""
    fig, ax = plt.figure(figsize=(14, 12)), plt.gca()
    
    # Add drop shadow effect for better visual separation
    def add_drop_shadow(ax, x, y, width, height, shadow_offset=0.05, alpha=0.3):
        shadow = plt.Rectangle(
            (x + shadow_offset, y - shadow_offset),
            width, height,
            facecolor='#000000',
            alpha=alpha,
            zorder=0
        )
        ax.add_patch(shadow)
    
    # Model specs (assuming based on general LLM architecture like GPT)
    model_name = "Qwen 2.5 32B"
    num_layers = 40
    num_heads = 40
    hidden_size = 6144
    
    # Define components - more detailed than simplified but less than technical
    layers = [
        {
            "name": "Input",
            "details": "Raw text input",
            "color": "#E8D0AA",  # Light brown
            "extra": None
        },
        {
            "name": "Tokenization",
            "details": "Convert text to token IDs",
            "color": "#D7BDE2",  # Light purple
            "extra": "Vocabulary: 150,000 tokens"
        },
        {
            "name": "Embeddings",
            "details": "Convert tokens to vectors + position info",
            "color": "#85C1E9",  # Blue
            "extra": f"Vector size: {hidden_size}"
        },
        {
            "name": "Transformer Stack",
            "details": f"{num_layers} identical transformer layers",
            "color": "#ABD7EB",  # Light blue
            "extra": "Self-attention + Feed-forward networks"
        },
        {
            "name": "Output Layer",
            "details": "Convert back to token probabilities",
            "color": "#A9DFBF",  # Green
            "extra": "Next token prediction"
        }
    ]
    
    # Layout parameters
    box_width = 8.0
    box_height = 1.7
    y_spacing = 3.0  # Increased spacing for better separation
    x_center = 4
    
    # Set a light gray background
    ax.set_facecolor('#F8F8F8')
    
    # Title
    plt.figtext(0.5, 0.95, f"{model_name} - Architecture Overview",
                ha='center', fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.925, "Intermediate Technical View",
                ha='center', fontsize=14)
    
    # Draw the layers
    for i, layer in enumerate(layers):
        y_pos = 9 - i * y_spacing
        
        # Add drop shadow for main box
        add_drop_shadow(ax, x_center, y_pos - box_height/2, box_width, box_height)
        
        # Draw main box with thicker border
        rect = plt.Rectangle(
            (x_center, y_pos - box_height/2),
            box_width, box_height,
            facecolor=layer["color"],
            edgecolor='black',
            alpha=0.9,
            linewidth=2.0,  # Increased linewidth
            zorder=10  # Higher zorder to ensure visibility
        )
        ax.add_patch(rect)
        
        # Layer name
        ax.text(
            x_center + box_width/2,
            y_pos + 0.3,
            layer["name"],
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=14,
            zorder=11
        )
        
        # Layer details
        ax.text(
            x_center + box_width/2,
            y_pos,
            layer["details"],
            ha='center',
            va='center',
            fontsize=12,
            zorder=11
        )
        
        # Additional details if available
        if layer["extra"]:
            ax.text(
                x_center + box_width/2,
                y_pos - 0.3,
                layer["extra"],
                ha='center',
                va='center',
                fontsize=11,
                style='italic',
                zorder=11
            )
        
        # Special handling for Transformer Stack - add internal details
        if "Transformer Stack" in layer["name"]:
            # Add diagram of a transformer block - INCREASED SIZE
            transformer_detail_width = box_width * 1.8  # Wider
            transformer_detail_height = 3.0  # Taller
            
            # Add drop shadow for detail box
            add_drop_shadow(ax, 
                           x_center + box_width + 0.5, 
                           y_pos - transformer_detail_height/2, 
                           transformer_detail_width, 
                           transformer_detail_height)
            
            # Background for transformer details - with THICKER border and LIGHTER color
            detail_box = plt.Rectangle(
                (x_center + box_width + 0.5, y_pos - transformer_detail_height/2),
                transformer_detail_width, transformer_detail_height,
                facecolor="#FFFFFF",  # Pure white for better contrast
                edgecolor='black',
                alpha=1.0,  # Full opacity
                linewidth=2.5,  # Thicker linewidth
                zorder=10
            )
            ax.add_patch(detail_box)
            
            # Title for detail box - LARGER and with background
            title_text = ax.text(
                x_center + box_width + 0.5 + transformer_detail_width/2,
                y_pos + transformer_detail_height/2 - 0.3,
                "Transformer Block Structure",
                ha='center',
                va='center',
                fontweight='bold',
                fontsize=14,  # Increased font size
                zorder=25,
                bbox=dict(facecolor='#F0F0F0', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.9)
            )
            
            # Draw a simple transformer block diagram - LARGER components
            component_width = transformer_detail_width * 0.5  # Wider components
            component_height = 0.6  # Taller components
            component_spacing = 0.4  # More spacing
            
            # Multi-head attention with drop shadow and thicker border
            add_drop_shadow(ax, 
                           x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2,
                           y_pos + 0.7,  # Adjusted position
                           component_width, 
                           component_height)
            
            mha_box = plt.Rectangle(
                (x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2, 
                 y_pos + 0.7),  # Adjusted position
                component_width, component_height,
                facecolor='#FFD700',  # Gold
                edgecolor='black',
                alpha=0.9,
                linewidth=2.0,  # Thicker border
                zorder=20
            )
            ax.add_patch(mha_box)
            
            # Add + for residual connection (more visible)
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width - 0.7,
                y_pos + 1.0,  # Adjusted position
                "+",
                ha='center',
                va='center',
                fontsize=22,  # Larger + sign
                fontweight='bold',
                color='#B22222',  # Dark red for visibility
                zorder=25
            )
            
            # Layer norm 1 with drop shadow and thicker border
            add_drop_shadow(ax, 
                           x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2,
                           y_pos + 0.7 - component_height - component_spacing,
                           component_width, 
                           component_height)
            
            ln1_box = plt.Rectangle(
                (x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2, 
                 y_pos + 0.7 - component_height - component_spacing),
                component_width, component_height,
                facecolor='#D2B4DE',  # Purple
                edgecolor='black',
                alpha=0.9,
                linewidth=2.0,
                zorder=20
            )
            ax.add_patch(ln1_box)
            
            # Feed-forward network with drop shadow and thicker border
            add_drop_shadow(ax, 
                           x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2,
                           y_pos - 0.7,  # Adjusted position
                           component_width, 
                           component_height)
            
            ffn_box = plt.Rectangle(
                (x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2, 
                 y_pos - 0.7),  # Adjusted position
                component_width, component_height,
                facecolor='#87CEFA',  # Light sky blue
                edgecolor='black',
                alpha=0.9,
                linewidth=2.0,
                zorder=20
            )
            ax.add_patch(ffn_box)
            
            # Add + for second residual connection (more visible)
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width - 0.7,
                y_pos - 0.4,  # Adjusted position
                "+",
                ha='center',
                va='center',
                fontsize=22,
                fontweight='bold',
                color='#B22222',  # Dark red for visibility
                zorder=25
            )
            
            # Layer norm 2 with drop shadow and thicker border
            add_drop_shadow(ax, 
                           x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2,
                           y_pos - 0.7 - component_height - component_spacing,
                           component_width, 
                           component_height)
            
            ln2_box = plt.Rectangle(
                (x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2, 
                 y_pos - 0.7 - component_height - component_spacing),
                component_width, component_height,
                facecolor='#D2B4DE',  # Purple
                edgecolor='black',
                alpha=0.9,
                linewidth=2.0,
                zorder=20
            )
            ax.add_patch(ln2_box)
            
            # Add component labels with better visibility - LARGER TEXT with BORDERS
            ax.text(
                x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2 + component_width/2,
                y_pos + 1.0,  # Adjusted position
                "Multi-Head Attention",
                ha='center',
                va='center',
                fontsize=13,  # Larger font
                fontweight='bold',
                color='black',
                zorder=25,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.3')
            )
            
            ax.text(
                x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2 + component_width/2,
                y_pos - 0.4,  # Adjusted position
                "Feed-Forward Network",
                ha='center',
                va='center',
                fontsize=13,  # Larger font
                fontweight='bold',
                color='black',
                zorder=25,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.3')
            )
            
            ax.text(
                x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2 + component_width/2,
                y_pos + 0.7 - component_height - component_spacing + component_height/2,
                "Layer Norm 1",  # Added numbers for clarity
                ha='center',
                va='center',
                fontsize=13,  # Larger font
                fontweight='bold',
                color='black',
                zorder=25,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.3')
            )
            
            ax.text(
                x_center + box_width + 0.5 + (transformer_detail_width - component_width)/2 + component_width/2,
                y_pos - 0.7 - component_height - component_spacing + component_height/2,
                "Layer Norm 2",  # Added numbers for clarity
                ha='center',
                va='center',
                fontsize=13,  # Larger font
                fontweight='bold',
                color='black',
                zorder=25,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.3')
            )
            
            # Add residual connection labels
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width - 0.7,
                y_pos + 1.3,  # Position above the + sign
                "Residual",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#B22222',
                zorder=25,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=None, pad=1)
            )
            
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width - 0.7,
                y_pos - 0.1,  # Position above the + sign
                "Residual",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                color='#B22222',
                zorder=25,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=None, pad=1)
            )
            
            # Draw connecting arrows - CURVED and THICKER
            arrow1 = FancyArrowPatch(
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos + transformer_detail_height/2 - 0.6),
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos + 0.7 + component_height),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=20,  # Larger arrow
                linewidth=2.5,  # Thicker line
                color='#000080',  # Navy blue for visibility
                zorder=15
            )
            ax.add_patch(arrow1)
            
            # Continue with other arrows with increased linewidth and size
            arrow2 = FancyArrowPatch(
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos + 0.7),
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos + 0.7 - component_height - component_spacing + component_height),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.5,
                color='#000080',
                zorder=15
            )
            ax.add_patch(arrow2)
            
            arrow3 = FancyArrowPatch(
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos + 0.7 - component_height - component_spacing),
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos - 0.7 + component_height),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.5,
                color='#000080',
                zorder=15
            )
            ax.add_patch(arrow3)
            
            arrow4 = FancyArrowPatch(
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos - 0.7),
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos - 0.7 - component_height - component_spacing + component_height),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.5,
                color='#000080',
                zorder=15
            )
            ax.add_patch(arrow4)
            
            arrow5 = FancyArrowPatch(
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos - 0.7 - component_height - component_spacing),
                (x_center + box_width + 0.5 + (transformer_detail_width)/2, y_pos - transformer_detail_height/2 + 0.6),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=2.5,
                color='#000080',
                zorder=15
            )
            ax.add_patch(arrow5)
            
            # Connect main block to detail - THICKER and with TEXT
            connector = FancyArrowPatch(
                (x_center + box_width, y_pos),
                (x_center + box_width + 0.5, y_pos),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",  # Added arrowhead
                linestyle='-',  # Solid line instead of dashed
                mutation_scale=15,
                linewidth=2.5,
                color='#000080',
                zorder=5
            )
            ax.add_patch(connector)
            
            # Add annotation text for the connector
            ax.text(
                x_center + box_width + 0.25,
                y_pos - 0.25,
                "Details",
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold',
                color='#000080',
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            # Additional annotations for attention mechanism - WITH BORDERS
            # Create a box for annotations
            annotation_box = plt.Rectangle(
                (x_center + box_width + 0.5 + transformer_detail_width * 0.65, 
                 y_pos + 0.2),
                transformer_detail_width * 0.32, 1.4,
                facecolor='#F8F8F8',
                edgecolor='black',
                alpha=0.9,
                linewidth=1.0,
                zorder=15
            )
            ax.add_patch(annotation_box)
            
            # Add title for annotations
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width * 0.81,
                y_pos + 1.4,
                "Key Components",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                zorder=20,
                bbox=dict(facecolor='#F0F0F0', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.2')
            )
            
            # Add formatted annotations with bullet points
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width * 0.7,
                y_pos + 1.1,
                f"• {num_heads} attention heads",
                ha='left',
                va='center',
                fontsize=10,
                color='black',
                zorder=20
            )
            
            # ...existing code for remaining annotations...
            
            # Add explanation of data flow
            ax.text(
                x_center + box_width + 0.5 + transformer_detail_width/2,
                y_pos - transformer_detail_height/2 + 0.3,
                "Data flows from top to bottom through the block",
                ha='center',
                va='center',
                fontsize=10,
                fontstyle='italic',
                bbox=dict(facecolor='#FFFFCC', alpha=0.9, edgecolor='gray'),
                zorder=20
            )

        # Draw connectors between main flow boxes (thicker and more visible)
        if i > 0:
            arrow = FancyArrowPatch(
                (x_center + box_width/2, y_pos - box_height/2 - 0.1),
                (x_center + box_width/2, 9 - (i-1) * y_spacing + box_height/2 + 0.1),
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2.5,  # Increased linewidth
                color='#5D6D7E',
                zorder=5
            )
            ax.add_patch(arrow)
    
    # Add key model parameters box with drop shadow
    add_drop_shadow(ax, 0.5, 3, 3.0, 4.0)
    params_box = plt.Rectangle(
        (0.5, 3),
        3.0, 4.0,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=2.0,  # Increased linewidth
        zorder=10
    )
    ax.add_patch(params_box)
    
    # Parameters title
    ax.text(
        2.0,
        6.7,
        "Model Parameters",
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=12
    )
    
    # Parameters list
    parameters = [
        "Parameters: 32 billion",
        "Layers: 40",
        "Hidden size: 6,144",
        "Attention heads: 40",
        "Context length: 128K tokens",
        "Output vocab: 150,000 tokens"
    ]
    
    for i, param in enumerate(parameters):
        ax.text(
            2.0,
            6.3 - i * 0.5,
            param,
            ha='center',
            va='center',
            fontsize=11
        )
    
    # Add key capabilities box with drop shadow
    add_drop_shadow(ax, x_center + box_width + 0.5, 3, 6, 2.5)
    capabilities_box = plt.Rectangle(
        (x_center + box_width + 0.5, 3),
        6, 2.5,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=2.0,  # Increased linewidth
        zorder=10
    )
    ax.add_patch(capabilities_box)
    
    # Capabilities title
    ax.text(
        x_center + box_width + 0.5 + 3,
        4.7,
        "Model Capabilities",
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=12
    )
    
    # Capabilities in two columns
    capabilities = [
        "Text generation",
        "Classification",
        "Summarization",
        "Translation",
        "Question answering",
        "Code completion"
    ]
    
    for i, capability in enumerate(capabilities):
        col = i % 2  # Two columns
        row = i // 2
        
        ax.text(
            x_center + box_width + 0.5 + 1.5 + col * 3,
            4.3 - row * 0.5,
            "• " + capability,
            ha='left',
            va='center',
            fontsize=11
        )
    
    # Add a border around the entire visualization
    border = plt.Rectangle(
        (0, 0.8),
        18.2, 9.4,
        fill=False,
        edgecolor='gray',
        linewidth=1.0,
        zorder=0
    )
    ax.add_patch(border)
    
    # Set axis limits and turn off axis
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.8, 10.2)
    ax.axis('off')
    
    # Save figure with higher DPI for better quality
    intermediate_file = os.path.join(output_dir, "llm_qwen_intermediate_flowchart.png")
    plt.savefig(intermediate_file, dpi=300, bbox_inches='tight')
    print(f"Intermediate LLM flowchart saved to {intermediate_file}")
    plt.close()

def create_technical_flowchart(output_dir):
    """Create a technical flowchart for LLM architecture"""
    fig, ax = plt.figure(figsize=(16, 20)), plt.gca()
    
    # LLM architecture components for Qwen 2.5
    # Note: These are approximations as exact architecture may not be publicly detailed
    
    # Model specs (assuming based on general LLM architecture like GPT)
    model_name = "Qwen 2.5 32B"
    num_layers = 40
    num_heads = 40
    hidden_size = 6144
    embedding_size = 6144
    vocab_size = 150000
    
    # Define components of the model
    layers = [
        {
            "name": "Input Text",
            "type": "input",
            "shape": "batch×sequence_length",
            "color": "#E8D0AA",  # Light brown
            "details": "Raw text input"
        },
        {
            "name": "Tokenization",
            "type": "processor",
            "shape": "batch×sequence_length",
            "color": "#D7BDE2",  # Light purple
            "details": f"Convert text to tokens from vocabulary of {vocab_size:,} tokens",
            "params": "0"
        },
        {
            "name": "Token Embeddings",
            "type": "embedding",
            "shape": f"batch×sequence_length×{embedding_size}",
            "color": "#85C1E9",  # Blue
            "details": "Convert token IDs to dense vectors",
            "params": f"{vocab_size * embedding_size:,}"
        },
        {
            "name": "Positional Embeddings",
            "type": "embedding",
            "shape": f"batch×sequence_length×{embedding_size}",
            "color": "#85C1E9",  # Blue
            "details": "Add position information to tokens",
            "params": f"{4096 * embedding_size:,}"  # Assuming max context length of 4096
        },
        {
            "name": "Embedding Dropout",
            "type": "dropout",
            "rate": "10%",
            "shape": f"batch×sequence_length×{embedding_size}",
            "color": "#F5CBA7",  # Orange
            "details": "Regularization for embeddings",
            "params": "0"
        }
    ]
    
    # Add transformer blocks (simplified representation)
    for i in range(1, 4):  # Just show first 3 blocks for clarity
        block = {
            "name": f"Transformer Block {i}",
            "type": "transformer",
            "color": "#ABD7EB",  # Light blue
            "details": f"Self-attention and feed-forward processing",
            "shape": f"batch×sequence_length×{hidden_size}",
            "params": f"{4 * hidden_size * hidden_size + 8 * num_heads * (hidden_size//num_heads)**2:,}"
        }
        layers.append(block)
    
    # Add ellipsis to indicate more blocks
    layers.append({
        "name": "...",
        "type": "ellipsis",
        "color": "#FFFFFF",  # White
        "details": f"{num_layers-6} more transformer blocks",
        "shape": f"batch×sequence_length×{hidden_size}",
        "params": "..."
    })
    
    # Add last 3 transformer blocks
    for i in range(num_layers-2, num_layers+1):
        block = {
            "name": f"Transformer Block {i}",
            "type": "transformer",
            "color": "#ABD7EB",  # Light blue
            "details": f"Self-attention and feed-forward processing",
            "shape": f"batch×sequence_length×{hidden_size}",
            "params": f"{4 * hidden_size * hidden_size + 8 * num_heads * (hidden_size//num_heads)**2:,}"
        }
        layers.append(block)
    
    # Add final layers
    layers.extend([
        {
            "name": "Layer Normalization",
            "type": "normalization",
            "shape": f"batch×sequence_length×{hidden_size}",
            "color": "#D2B4DE",  # Purple
            "details": "Final normalization",
            "params": f"{2 * hidden_size:,}"
        },
        {
            "name": "Output Projection",
            "type": "dense",
            "shape": f"batch×sequence_length×{vocab_size}",
            "color": "#A9CCE3",  # Light blue
            "details": "Project to vocabulary logits",
            "params": f"{hidden_size * vocab_size + vocab_size:,}"
        },
        {
            "name": "Output Distribution",
            "type": "output",
            "shape": f"batch×sequence_length×{vocab_size}",
            "color": "#A9DFBF",  # Green
            "details": "Probability distribution over vocabulary",
            "params": "0"
        }
    ])
    
    # Calculate total parameters (approximate)
    # For a 32B parameter model like Qwen 2.5 32B
    total_params = 32_000_000_000
    
    # Layout parameters
    box_width = 7.0
    box_height = 1.8
    y_spacing = 2.5
    
    # Title
    plt.figtext(0.5, 0.97, f"{model_name} Architecture - Technical Flowchart",
                ha='center', fontsize=18, fontweight='bold')
    plt.figtext(0.5, 0.955, f"Total Parameters: ~{total_params:,}",
                ha='center', fontsize=14)
    
    # Draw the layers
    for i, layer in enumerate(layers):
        y_pos = -i * y_spacing
        
        # For ellipsis layer, use special formatting
        if layer["type"] == "ellipsis":
            ax.text(
                1 + box_width/2,
                y_pos,
                layer["name"],
                ha='center',
                va='center',
                fontsize=20,
                fontweight='bold'
            )
            ax.text(
                1 + box_width/2,
                y_pos - 0.5,
                layer["details"],
                ha='center',
                va='center',
                fontsize=12
            )
            
            # Draw dotted connectors
            if i > 0:
                arrow = FancyArrowPatch(
                    (1 + box_width/2, (-i+1) * -y_spacing + box_height/2),
                    (1 + box_width/2, y_pos - 0.5),
                    connectionstyle="arc3,rad=0",
                    arrowstyle="-|>",
                    linestyle='dotted',
                    mutation_scale=15,
                    linewidth=2,
                    color='#5D6D7E',
                    zorder=2
                )
                ax.add_patch(arrow)
            
            # Draw connector to next box
            if i < len(layers) - 1:
                arrow = FancyArrowPatch(
                    (1 + box_width/2, y_pos + 0.5),
                    (1 + box_width/2, (-i-1) * -y_spacing - box_height/2),
                    connectionstyle="arc3,rad=0",
                    arrowstyle="-|>",
                    linestyle='dotted',
                    mutation_scale=15,
                    linewidth=2,
                    color='#5D6D7E',
                    zorder=2
                )
                ax.add_patch(arrow)
            
            continue
            
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
        
        # Layer name - positioned higher in box
        ax.text(
            1 + box_width/2,
            y_pos - box_height/2 + box_height * 0.8,
            layer["name"],
            ha='center',
            va='center',
            fontweight='bold',
            fontsize=13
        )
        
        # Add specific details based on layer type
        if layer["type"] in ["input", "processor", "output"]:
            # For simple layers
            ax.text(
                1 + box_width/2,
                y_pos - 0.1,
                f"Shape: {layer['shape']}",
                ha='center',
                va='center',
                fontsize=11
            )
            ax.text(
                1 + box_width/2,
                y_pos + 0.3,
                f"{layer['details']}",
                ha='center',
                va='center',
                fontsize=11
            )
            
        elif layer["type"] in ["embedding", "dense", "normalization"]:
            # For parameter-heavy layers
            ax.text(
                1 + box_width/2,
                y_pos - 0.1,
                f"Shape: {layer['shape']}",
                ha='center',
                va='center',
                fontsize=11
            )
            ax.text(
                1 + box_width/2,
                y_pos + 0.3,
                f"{layer['details']}",
                ha='center',
                va='center',
                fontsize=11
            )
            # Parameters
            ax.text(
                1 + box_width/2,
                y_pos - box_height/2 + box_height * 0.25,
                f"Parameters: {layer['params']}",
                ha='center',
                va='center',
                fontsize=10,
                style='italic'
            )
            
        elif layer["type"] == "dropout":
            # For dropout layers
            ax.text(
                1 + box_width/2,
                y_pos - 0.1,
                f"Rate: {layer['rate']}",
                ha='center',
                va='center',
                fontsize=11
            )
            ax.text(
                1 + box_width/2,
                y_pos + 0.3,
                f"Shape: {layer['shape']}",
                ha='center',
                va='center',
                fontsize=11
            )
            
        elif layer["type"] == "transformer":
            # For transformer blocks - add extra detail
            
            # Draw inner boxes to represent transformer components
            inner_width = box_width * 0.3
            inner_height = box_height * 0.25
            inner_spacing = inner_width * 1.2
            inner_y_offset = -0.05
            
            # Multi-head attention box
            attention_box = Rectangle(
                (1 + box_width/2 - inner_spacing/2 - inner_width, y_pos + inner_y_offset - inner_height/2),
                inner_width, inner_height,
                facecolor='#FFD700',  # Gold
                edgecolor='black',
                alpha=0.7,
                linewidth=0.8,
                zorder=4
            )
            ax.add_patch(attention_box)
            
            # Feed forward box
            ff_box = Rectangle(
                (1 + box_width/2 + inner_spacing/2, y_pos + inner_y_offset - inner_height/2),
                inner_width, inner_height,
                facecolor='#87CEFA',  # Light sky blue
                edgecolor='black',
                alpha=0.7,
                linewidth=0.8,
                zorder=4
            )
            ax.add_patch(ff_box)
            
            # Labels for inner components
            ax.text(
                1 + box_width/2 - inner_spacing/2 - inner_width/2,
                y_pos + inner_y_offset,
                "MHA",
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                zorder=5
            )
            
            ax.text(
                1 + box_width/2 + inner_spacing/2 + inner_width/2,
                y_pos + inner_y_offset,
                "FFN",
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                zorder=5
            )
            
            # Add overall transformer block details
            ax.text(
                1 + box_width/2,
                y_pos - 0.2,
                f"Shape: {layer['shape']}",
                ha='center',
                va='center',
                fontsize=11
            )
            ax.text(
                1 + box_width/2,
                y_pos + 0.4,
                f"{num_heads} attention heads | Hidden dim: {hidden_size}",
                ha='center',
                va='center',
                fontsize=10
            )
            
            # Parameters
            ax.text(
                1 + box_width/2,
                y_pos - box_height/2 + box_height * 0.25,
                f"Parameters: {layer['params']}",
                ha='center',
                va='center',
                fontsize=10,
                style='italic'
            )
        
        # Draw connectors between boxes
        if i > 0 and layer["type"] != "ellipsis" and layers[i-1]["type"] != "ellipsis":
            arrow = FancyArrowPatch(
                (1 + box_width/2, y_pos - box_height/2 - 0.1),  # Start point
                (1 + box_width/2, (-i+1) * -y_spacing + box_height/2),  # End point
                connectionstyle="arc3,rad=0",
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.5,
                color='#5D6D7E',
                zorder=2
            )
            ax.add_patch(arrow)
    
    # Add detailed block explanation
    explanation_x = 0.07
    explanation_y = 0.89
    
    # Title for explanation
    plt.figtext(explanation_x, explanation_y, "Transformer Block Components:", fontsize=11, fontweight='bold')
    
    # Multi-head Attention explanation
    plt.figtext(explanation_x, explanation_y - 0.03, "MHA (Multi-Head Attention):", fontsize=10, fontweight='bold')
    plt.figtext(explanation_x + 0.02, explanation_y - 0.045, "• Projects input into Query, Key, Value matrices", fontsize=9)
    plt.figtext(explanation_x + 0.02, explanation_y - 0.07, f"• {num_heads} attention heads operating in parallel", fontsize=9)
    plt.figtext(explanation_x + 0.02, explanation_y - 0.095, "• Each head captures different relationship patterns", fontsize=9)
    
    # Feed-Forward Network explanation
    plt.figtext(explanation_x, explanation_y - 0.13, "FFN (Feed-Forward Network):", fontsize=10, fontweight='bold')
    plt.figtext(explanation_x + 0.02, explanation_y - 0.145, "• Two fully connected layers with GELU activation", fontsize=9)
    plt.figtext(explanation_x + 0.02, explanation_y - 0.17, f"• First expands to 4× width ({hidden_size*4}), then projects back", fontsize=9)
    
    # Add note about residual connections and layer norm
    plt.figtext(explanation_x, explanation_y - 0.21, "Each transformer block also includes:", fontsize=10, fontweight='bold')
    plt.figtext(explanation_x + 0.02, explanation_y - 0.235, "• Residual connections around both MHA and FFN", fontsize=9)
    plt.figtext(explanation_x + 0.02, explanation_y - 0.26, "• Layer normalization for stable training", fontsize=9)
    
    # Legend for color codes
    legend_items = [
        {"label": "Input/Output", "color": "#E8D0AA"},
        {"label": "Processing", "color": "#D7BDE2"},
        {"label": "Embedding Layers", "color": "#85C1E9"},
        {"label": "Transformer Blocks", "color": "#ABD7EB"},
        {"label": "Dropout Layers", "color": "#F5CBA7"},
        {"label": "Normalization", "color": "#D2B4DE"},
        {"label": "Multi-Head Attention", "color": "#FFD700"},
        {"label": "Feed-Forward Network", "color": "#87CEFA"}
    ]
    
    legend_x = 0.07
    legend_y = 0.78
    for i, item in enumerate(legend_items):
        col = i % 2  # Two columns of legend items
        row = i // 2
        
        legend_box = plt.Rectangle(
            (legend_x + col * 0.25, legend_y - row * 0.03),
            0.025, 0.02,
            facecolor=item["color"],
            edgecolor='black',
            linewidth=0.5,
            transform=fig.transFigure,
            figure=fig
        )
        fig.add_artist(legend_box)
        
        plt.figtext(
            legend_x + 0.035 + col * 0.25,
            legend_y - row * 0.03,
            item["label"],
            fontsize=9
        )
    
    # Add information on model capabilities at the bottom
    y_bottom = -(len(layers) * y_spacing + 1)
    info_box = plt.Rectangle(
        (1, y_bottom),
        box_width, 3.2,
        facecolor='#F8F9F9',
        edgecolor='black',
        alpha=0.9,
        linewidth=1.5
    )
    ax.add_patch(info_box)
    
    # Capabilities title
    ax.text(
        1 + box_width/2,
        y_bottom + 2.8,
        "Qwen 2.5 Model Capabilities",
        ha='center',
        va='center',
        fontweight='bold',
        fontsize=12
    )
    
    # Capabilities list
    capabilities = [
        "Context Length: 128K tokens",
        f"Vocabulary Size: {vocab_size:,} tokens",
        f"Hidden Size: {hidden_size}",
        f"Attention Heads: {num_heads}",
        f"Number of Layers: {num_layers}",
        "Training: Instruction tuning, RLHF",
        "Supported Languages: Multilingual (focus on English & Chinese)"
    ]
    
    for i, capability in enumerate(capabilities):
        ax.text(
            1 + box_width/2,
            y_bottom + 2.3 - i * 0.35,
            capability,
            ha='center',
            va='center',
            fontsize=10
        )
    
    # Set axis limits and turn off axis
    ax.set_xlim(0, box_width + 2)
    ax.set_ylim(y_bottom - 0.5, 1)
    ax.axis('off')
    
    # Save figure
    technical_file = os.path.join(output_dir, "llm_qwen_technical_flowchart.png")
    plt.savefig(technical_file, dpi=300, bbox_inches='tight')
    print(f"LLM technical flowchart saved to {technical_file}")
    plt.close()

if __name__ == "__main__":
    visualize_llm_architecture("intermediate")  # Default to intermediate view
    # Other options:
    # visualize_llm_architecture("simple")
    # visualize_llm_architecture("technical")
