import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# SETTINGS
# ==========================================
CSV_PATH = "metrics/dfine/detr_adaptive_inner_siou/evaluation_100/confusion_matrix_raw.csv" 
FIG_SIZE = (16, 9) # Widescreen PPT format
# ==========================================

def generate_ppt_confusion_matrix():
    print(f"Loading pre-computed matrix from {CSV_PATH}...")
    
    # Read the CSV. We assume the first column contains the class names (True labels)
    df = pd.read_csv(CSV_PATH, index_col=0)
    
    # --- TRICK 1: DROP EMPTY CLASSES ---
    # If a bird never showed up in reality AND was never predicted, it's just wasting slide space.
    # Keep only rows and columns that have at least one value > 0
    df = df.loc[(df != 0).any(axis=1), (df != 0).any(axis=0)]
    
    print(f"Matrix reduced to {df.shape[0]} active True classes and {df.shape[1]} active Predicted classes.")
    
    # Get the raw numbers and class names
    cm = df.values.astype(int)
    true_classes = df.index.tolist()
    pred_classes = df.columns.tolist()
    
    # Set up the matplotlib figure
    plt.figure(figsize=FIG_SIZE)
    sns.set_theme(style="whitegrid", font_scale=1.1) 
    
    # --- TRICK 2: HIDE ZEROS ---
    # Create an annotation array that leaves 0 cells completely blank
    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            annot_labels[i, j] = str(val) if val > 0 else ""
            
    # --- TRICK 3: LOG SCALING ---
    # Log scaling prevents the background class (which has massive numbers) 
    # from washing out the colors of the tiny bird mistakes.
    log_cm = np.log1p(cm)

    # Draw the heatmap
    ax = sns.heatmap(
        log_cm, 
        annot=annot_labels,      # Use our custom blank-zero labels
        fmt="",                  
        cmap="Blues",            # Professional blue color scheme
        cbar=False,              # No colorbar needed for PPT
        linewidths=0.5,          
        linecolor='lightgray',
        xticklabels=pred_classes,
        yticklabels=true_classes
    )
    
    # Formatting the axes
    plt.title("D-FINE DETR Wildlife Detection - Confusion Matrix (Epoch 270)", fontsize=20, pad=20, weight='bold')
    plt.ylabel('Actual (Ground Truth)', fontsize=14, weight='bold')
    plt.xlabel('Predicted by Model', fontsize=14, weight='bold')
    
    # Rotate labels so they don't overlap
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # --- TRICK 4: HIGH RESOLUTION ---
    output_file = "confusion_matrix_ppt_dfine.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved file: {output_file}")
    
    plt.close()

if __name__ == "__main__":
    generate_ppt_confusion_matrix()