"""
Improved visualization functions for datasets with many classes (55+)

Features:
- Splits confusion matrix into multiple smaller subplots
- Creates separate PR curve plots per class
- Generates summary plots for top/bottom performing classes
- Interactive HTML outputs for exploration
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import pandas as pd


def plot_confusion_matrix_grid(
    confusion_matrix,
    class_names,
    output_dir,
    classes_per_plot=10,
    figsize_per_plot=(12, 10)
):
    """
    Split confusion matrix into multiple smaller, readable plots.
    
    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
        output_dir: Where to save plots
        classes_per_plot: How many classes per subplot grid
        figsize_per_plot: Figure size for each grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_classes = len(class_names)
    n_plots = (n_classes + classes_per_plot - 1) // classes_per_plot
    
    print(f"\n📊 Creating {n_plots} confusion matrix plots ({classes_per_plot} classes each)...")
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * classes_per_plot
        end_idx = min(start_idx + classes_per_plot, n_classes)
        
        # Extract subset of confusion matrix
        cm_subset = confusion_matrix[start_idx:end_idx, start_idx:end_idx]
        names_subset = class_names[start_idx:end_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        
        # Use log scale for better visibility with imbalanced data
        cm_log = np.log10(cm_subset + 1)  # +1 to avoid log(0)
        
        sns.heatmap(
            cm_log,
            annot=cm_subset,  # Show actual counts
            fmt='d',
            cmap='Blues',
            xticklabels=names_subset,
            yticklabels=names_subset,
            ax=ax,
            cbar_kws={'label': 'log10(count + 1)'}
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(
            f'Confusion Matrix (Classes {start_idx+1}-{end_idx})\n'
            f'Log Scale - Annotations show actual counts',
            fontsize=14
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f'confusion_matrix_classes_{start_idx+1}_to_{end_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")


def plot_pr_curves_per_class(
    pr_data,
    output_dir,
    classes_per_figure=6,
    figsize=(15, 10)
):
    """
    Create separate PR curve plots, grouping multiple classes per figure.
    
    Args:
        pr_data: Dict with class_id -> {'precision', 'recall', 'ap', 'class_name'}
        output_dir: Where to save plots
        classes_per_figure: How many classes to show per figure
        figsize: Figure size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out background and classes with no data
    valid_classes = {
        cid: data for cid, data in pr_data.items()
        if cid > 0 and data['ap'] is not None
    }
    
    if not valid_classes:
        print("⚠️  No valid PR curves to plot")
        return
    
    # Sort by AP (descending) for better organization
    sorted_classes = sorted(
        valid_classes.items(),
        key=lambda x: x[1]['ap'],
        reverse=True
    )
    
    n_classes = len(sorted_classes)
    n_figures = (n_classes + classes_per_figure - 1) // classes_per_figure
    
    print(f"\n📈 Creating {n_figures} PR curve plots ({classes_per_figure} classes each)...")
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * classes_per_figure
        end_idx = min(start_idx + classes_per_figure, n_classes)
        
        classes_subset = sorted_classes[start_idx:end_idx]
        
        # Create subplot grid
        n_rows = (len(classes_subset) + 2) // 3  # 3 columns
        n_cols = min(3, len(classes_subset))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        axes = axes.flatten()
        
        for idx, (class_id, data) in enumerate(classes_subset):
            ax = axes[idx]
            
            precision = data['precision']
            recall = data['recall']
            ap = data['ap']
            class_name = data['class_name']
            
            # Plot PR curve
            ax.plot(recall, precision, linewidth=2, label=f'AP={ap:.3f}')
            ax.fill_between(recall, precision, alpha=0.2)
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(f'{class_name}\n(Class {class_id})', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(classes_subset), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Precision-Recall Curves (Sorted by AP, Group {fig_idx+1}/{n_figures})',
            fontsize=16,
            y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        output_path = output_dir / f'pr_curves_group_{fig_idx+1}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")


def plot_ap_summary_bar(
    pr_data,
    output_dir,
    figsize=(14, 10),
    top_n=20
):
    """
    Create bar chart showing AP for all classes.
    
    Args:
        pr_data: Dict with class_id -> {'ap', 'class_name'}
        output_dir: Where to save plot
        figsize: Figure size
        top_n: Only show top N classes if there are many
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid classes
    valid_classes = [
        (data['class_name'], data['ap'])
        for cid, data in pr_data.items()
        if cid > 0 and data['ap'] is not None
    ]
    
    if not valid_classes:
        print("⚠️  No valid AP data to plot")
        return
    
    # Sort by AP
    valid_classes.sort(key=lambda x: x[1], reverse=True)
    
    # If too many classes, create two plots: top N and bottom N
    if len(valid_classes) > top_n * 2:
        # Top N
        top_classes = valid_classes[:top_n]
        names_top, aps_top = zip(*top_classes)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(aps_top)))
        
        bars = ax.barh(range(len(names_top)), aps_top, color=colors)
        ax.set_yticks(range(len(names_top)))
        ax.set_yticklabels(names_top, fontsize=9)
        ax.set_xlabel('Average Precision (AP)', fontsize=12)
        ax.set_title(f'Top {top_n} Classes by AP', fontsize=14)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, ap) in enumerate(zip(bars, aps_top)):
            ax.text(ap + 0.01, i, f'{ap:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / f'ap_summary_top_{top_n}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Saved: {output_path}")
        
        # Bottom N
        bottom_classes = valid_classes[-top_n:]
        names_bottom, aps_bottom = zip(*bottom_classes)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(aps_bottom)))
        
        bars = ax.barh(range(len(names_bottom)), aps_bottom, color=colors)
        ax.set_yticks(range(len(names_bottom)))
        ax.set_yticklabels(names_bottom, fontsize=9)
        ax.set_xlabel('Average Precision (AP)', fontsize=12)
        ax.set_title(f'Bottom {top_n} Classes by AP', fontsize=14)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, ap) in enumerate(zip(bars, aps_bottom)):
            ax.text(ap + 0.01, i, f'{ap:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / f'ap_summary_bottom_{top_n}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {output_path}")
        
    else:
        # Show all classes
        names, aps = zip(*valid_classes)
        
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(aps)))
        
        bars = ax.barh(range(len(names)), aps, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Average Precision (AP)', fontsize=12)
        ax.set_title('Average Precision by Class (Sorted)', fontsize=14)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, ap) in enumerate(zip(bars, aps)):
            ax.text(ap + 0.01, i, f'{ap:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / 'ap_summary_all_classes.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📊 Saved: {output_path}")


def create_summary_report(
    metrics,
    pr_data,
    output_dir
):
    """
    Create a text summary report of performance.
    
    Args:
        metrics: Dict with 'mAP' and other metrics
        pr_data: Dict with per-class data
        output_dir: Where to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter valid classes
    valid_classes = [
        (cid, data)
        for cid, data in pr_data.items()
        if cid > 0 and data['ap'] is not None
    ]
    
    # Sort by AP
    valid_classes.sort(key=lambda x: x[1]['ap'], reverse=True)
    
    report_path = output_dir / 'performance_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DETR Performance Summary\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Overall mAP: {metrics.get('mAP', 0.0):.4f}\n\n")
        
        f.write("Top 10 Best Performing Classes:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Class ID':<10} {'Class Name':<35} {'AP':<10}\n")
        f.write("-"*70 + "\n")
        
        for rank, (cid, data) in enumerate(valid_classes[:10], 1):
            f.write(f"{rank:<6} {cid:<10} {data['class_name'][:33]:<35} {data['ap']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Bottom 10 Worst Performing Classes:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Rank':<6} {'Class ID':<10} {'Class Name':<35} {'AP':<10}\n")
        f.write("-"*70 + "\n")
        
        for rank, (cid, data) in enumerate(valid_classes[-10:][::-1], 1):
            f.write(f"{rank:<6} {cid:<10} {data['class_name'][:33]:<35} {data['ap']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("All Classes (Sorted by AP):\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class ID':<10} {'Class Name':<40} {'AP':<10}\n")
        f.write("-"*70 + "\n")
        
        for cid, data in valid_classes:
            f.write(f"{cid:<10} {data['class_name'][:38]:<40} {data['ap']:.4f}\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n📄 Saved: {report_path}")


    

def save_confusion_matrix_csv(confusion_matrix, class_names, output_dir):
    """
    Saves the entire NxN confusion matrix to a CSV file with class names 
    as rows (True) and columns (Predicted).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with row and column labels
    df_cm = pd.DataFrame(
        confusion_matrix, 
        index=class_names,     # True classes (rows)
        columns=class_names    # Predicted classes (cols)
    )
    
    # Save to CSV
    csv_path = output_dir / 'confusion_matrix_raw.csv'
    
    # Index label 'True \ Predicted' makes the top-left corner of the CSV clear
    df_cm.to_csv(csv_path, index_label='True \\ Predicted')
    
    print(f"  ✓ Saved Raw Confusion Matrix CSV: {csv_path}")


# ============================================================================
# Integration with existing evaluation code
# ============================================================================

def create_readable_visualizations(metrics, pr_data, confusion_matrix, class_names, output_dir):
    """
    Main function to create all visualizations for many-class datasets.
    
    Call this instead of the old plotting functions.
    
    Args:
        metrics: Dict with 'mAP' and other metrics
        pr_data: Dict with class_id -> {'precision', 'recall', 'ap', 'class_name'}
        confusion_matrix: NxN numpy array
        class_names: List of class names
        output_dir: Where to save all plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Creating Readable Visualizations for Many-Class Dataset")
    print(f"{'='*70}")
    
    # 0. Save the raw matrix to CSV for future custom visualizations
    save_confusion_matrix_csv(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_dir=output_dir
    )
    
    # 1. Split confusion matrix into readable chunks
    plot_confusion_matrix_grid(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_dir=output_dir / 'confusion_matrices',
        classes_per_plot=10
    )
    
    # 2. Create grouped PR curves
    plot_pr_curves_per_class(
        pr_data=pr_data,
        output_dir=output_dir / 'pr_curves',
        classes_per_figure=6
    )
    
    # 3. Create AP summary bar charts
    plot_ap_summary_bar(
        pr_data=pr_data,
        output_dir=output_dir,
        top_n=20
    )
    
    # 4. Create text summary report
    create_summary_report(
        metrics=metrics,
        pr_data=pr_data,
        output_dir=output_dir
    )
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    print("This module provides readable visualization functions for many-class datasets.")
    print("Import and use create_readable_visualizations() in your evaluation code.")