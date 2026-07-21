import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional

def get_pastis_cmap(config: Dict[str, Any]) -> mcolors.ListedColormap:
    """Create a custom colormap for the PASTIS classes."""
    # Define a distinct color for each of the 20 possible classes (0-19)
    # Using a mix of tab20 and some custom colors for specific crops
    colors = [
        '#000000', # 0: Background
        '#7CFC00', # 1: Meadow (Lawn green)
        '#FFD700', # 2: Soft Winter Wheat (Gold)
        '#FFA500', # 3: Corn (Orange)
        '#DAA520', # 4: Winter Barley (Goldenrod)
        '#FFFF00', # 5: Winter Rapeseed (Yellow)
        '#BDB76B', # 6: Spring Barley (Dark Khaki)
        '#FF8C00', # 7: Sunflower (Dark Orange)
        '#800080', # 8: Grapevine (Purple)
        '#8B0000', # 9: Beet (Dark Red)
        '#CD853F', # 10: Winter Triticale (Peru)
        '#F4A460', # 11: Winter Durum Wheat (Sandy Brown)
        '#FF69B4', # 12: Fruits/Vegetables/Flowers (Hot Pink)
        '#8B4513', # 13: Potatoes (Saddle Brown)
        '#32CD32', # 14: Leguminous Fodder (Lime Green)
        '#008000', # 15: Soybeans (Green)
        '#006400', # 16: Orchard (Dark Green)
        '#D2B48C', # 17: Mixed Cereal (Tan)
        '#A0522D', # 18: Sorghum (Sienna)
        '#FFFFFF', # 19: Void (White)
    ]
    return mcolors.ListedColormap(colors)

def plot_rgb(s2_data: np.ndarray, t_idx: int = 20, ax: Optional[plt.Axes] = None, title: str = "RGB") -> None:
    """
    Plot RGB composite for a specific time step.
    Assuming B4 (Red) is index 2, B3 (Green) is index 1, B2 (Blue) is index 0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Get bands B4, B3, B2 (indices 2, 1, 0)
    rgb = s2_data[t_idx, [2, 1, 0], :, :]
    
    # Clip and normalize for visualization (simple 2-98 percentile)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(float)
    for i in range(3):
        p2, p98 = np.percentile(rgb[..., i], (2, 98))
        rgb[..., i] = np.clip((rgb[..., i] - p2) / (p98 - p2), 0, 1)
        
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis('off')

def plot_target(target_data: np.ndarray, config: Dict[str, Any], ax: Optional[plt.Axes] = None, title: str = "Target") -> None:
    """Plot the target class map with custom colormap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    cmap = get_pastis_cmap(config)
    bounds = np.arange(-0.5, 20.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(target_data[0], cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    return im

def plot_class_distribution(class_counts: Dict[int, int], config: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Plot bar chart of class pixel counts."""
    classes = config['classes']
    
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [classes.get(k, f"Class {k}") for k, v in sorted_items]
    counts = [v for k, v in sorted_items]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts, y=labels, palette="viridis")
    plt.title("Pixel Distribution per Class")
    plt.xlabel("Number of Pixels")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_temporal_profile(s2_data: np.ndarray, target_data: np.ndarray, class_idx: int, config: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Plot temporal mean profile for a specific class."""
    mask = (target_data[0] == class_idx)
    if not np.any(mask):
        print(f"Class {class_idx} not found in this patch.")
        return
        
    # Mean reflectance per band over time for this class
    # s2_data: (46, 10, 128, 128)
    # Masked mean: (46, 10)
    class_pixels = s2_data[:, :, mask] # (46, 10, N_pixels)
    mean_profile = np.mean(class_pixels, axis=2) # (46, 10)
    
    plt.figure(figsize=(12, 5))
    bands = config['dataset']['bands']
    
    # Only plot a few key bands for clarity (B3, B4, B8, B11)
    plot_bands = [1, 2, 6, 8] # Indices for Green, Red, NIR, SWIR1
    
    for b_idx in plot_bands:
        plt.plot(mean_profile[:, b_idx], label=bands[b_idx], marker='o', markersize=3)
        
    plt.title(f"Temporal Profile: {config['classes'].get(class_idx, class_idx)}")
    plt.xlabel("Temporal Observation Index")
    plt.ylabel("Mean Reflectance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def create_prediction_panel(s2_data: np.ndarray, target: np.ndarray, pred: np.ndarray, config: Dict[str, Any], patch_id: int, save_path: Optional[str] = None):
    """Create a 1x3 panel comparing RGB, Ground Truth, and Prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_rgb(s2_data, t_idx=20, ax=axes[0], title=f"Patch {patch_id} RGB (t=20)")
    
    cmap = get_pastis_cmap(config)
    bounds = np.arange(-0.5, 20.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    axes[1].imshow(target[0], cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    im = axes[2].imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
    axes[2].set_title("Model Prediction")
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot confusion matrix from a DataFrame."""
    plt.figure(figsize=(14, 12))
    
    # Normalize by row to show recall percentages
    cm_norm = cm_df.div(cm_df.sum(axis=1), axis=0).fillna(0)
    
    sns.heatmap(cm_norm, annot=False, cmap='Blues', fmt='.2f', xticklabels=cm_df.columns, yticklabels=cm_df.index)
    plt.title("Normalized Confusion Matrix (Recall)")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_per_class_f1(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot per-class F1 score bar chart."""
    # Drop summary rows if present
    classes = [c for c in metrics_df.index if str(c).replace('.','',1).isdigit() or c.isalpha() and c not in ['accuracy', 'macro avg', 'weighted avg']]
    df_clean = metrics_df.loc[classes].sort_values(by='f1-score', ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=df_clean['f1-score'], y=df_clean.index, palette="coolwarm")
    plt.title("Per-Class F1 Score")
    plt.xlabel("F1 Score")
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
