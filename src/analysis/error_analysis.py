"""
Error analysis and visualization tools.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.face_classifier import FaceClassifier
from src.data.dataset import EmbeddingDataset


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, top_n=20):
    """
    Plot confusion matrix for top N classes.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        save_path: Path to save figure
        top_n: Number of top classes to show
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Get top N classes by frequency
    unique, counts = np.unique(y_true, return_counts=True)
    top_indices = np.argsort(counts)[-top_n:][::-1]
    top_classes = unique[top_indices]
    
    # Filter confusion matrix
    cm_filtered = cm[np.ix_(top_classes, top_classes)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] if class_names else f'Class {i}' for i in top_classes],
                yticklabels=[class_names[i] if class_names else f'Class {i}' for i in top_classes])
    plt.title(f'Confusion Matrix (Top {top_n} Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_errors(model, dataloader, device, class_names=None, top_k=10):
    """
    Analyze prediction errors.
    
    Returns:
        Dictionary with error analysis results
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    errors.append({
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'confidence': probs[i][predicted[i]].item(),
                        'true_prob': probs[i][labels[i]].item()
                    })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Get top K errors
    top_errors = errors[:top_k]
    
    # Count error types
    error_counts = {}
    for error in errors:
        key = (error['true_label'], error['predicted_label'])
        error_counts[key] = error_counts.get(key, 0) + 1
    
    return {
        'total_errors': len(errors),
        'top_errors': top_errors,
        'error_counts': error_counts,
        'all_errors': errors
    }


def visualize_failure_cases(errors, class_names=None, save_path=None):
    """
    Visualize failure cases.
    """
    if len(errors['top_errors']) == 0:
        print("No errors to visualize")
        return
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, error in enumerate(errors['top_errors'][:10]):
        ax = axes[i]
        true_name = class_names[error['true_label']] if class_names else f"Class {error['true_label']}"
        pred_name = class_names[error['predicted_label']] if class_names else f"Class {error['predicted_label']}"
        
        ax.text(0.5, 0.7, f"True: {true_name}", ha='center', fontsize=10)
        ax.text(0.5, 0.5, f"Pred: {pred_name}", ha='center', fontsize=10)
        ax.text(0.5, 0.3, f"Conf: {error['confidence']:.3f}", ha='center', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"Error {i+1}", fontsize=10)
    
    plt.suptitle('Top 10 Prediction Errors', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Failure cases visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot distribution of predictions vs true labels.
    """
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    ax1.bar(range(len(unique_true)), counts_true)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Frequency')
    ax1.set_title('True Label Distribution')
    ax1.set_xticks(range(len(unique_true)))
    ax1.set_xticklabels([class_names[i] if class_names else f'C{i}' for i in unique_true], 
                        rotation=45, ha='right')
    
    # Predicted distribution
    ax2.bar(range(len(unique_pred)), counts_pred)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Predicted Label Distribution')
    ax2.set_xticks(range(len(unique_pred)))
    ax2.set_xticklabels([class_names[i] if class_names else f'C{i}' for i in unique_pred],
                        rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

