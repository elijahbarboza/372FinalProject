"""
Hyperparameter tuning using validation data.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.face_classifier import FaceClassifier
from src.data.dataset import EmbeddingDataset
from src.training.trainer import Trainer, evaluate_model


def hyperparameter_search(embeddings, labels, device, num_classes, 
                          train_loader, val_loader):
    """
    Perform hyperparameter search.
    
    Returns:
        Dictionary with best hyperparameters and results
    """
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'dropout_rate': [0.2, 0.3, 0.4],
        'hidden_dims': [[128], [256], [256, 128], [512, 256]],
        'weight_decay': [0.0, 0.0001, 0.001]
    }
    
    best_score = 0.0
    best_params = None
    all_results = []
    
    # Generate all combinations (simplified - using random search for efficiency)
    import random
    n_trials = 20  # Limit trials for efficiency
    
    print(f"Performing hyperparameter search ({n_trials} trials)...")
    print("=" * 80)
    
    for trial in range(n_trials):
        # Sample random hyperparameters
        lr = random.choice(param_grid['learning_rate'])
        dropout = random.choice(param_grid['dropout_rate'])
        hidden = random.choice(param_grid['hidden_dims'])
        wd = random.choice(param_grid['weight_decay'])
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"  LR: {lr}, Dropout: {dropout}, Hidden: {hidden}, Weight Decay: {wd}")
        
        # Create model
        model = FaceClassifier(
            input_dim=embeddings.shape[1],
            hidden_dims=hidden,
            num_classes=num_classes,
            dropout_rate=dropout,
            use_batch_norm=True
        )
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Train
        trainer = Trainer(model.to(device), device)
        trainer.train(train_loader, val_loader, optimizer, scheduler, 
                     num_epochs=15, save_path=None, early_stopping_patience=5)
        
        # Evaluate on validation set
        val_results = evaluate_model(model, val_loader, device)
        val_score = val_results['f1']  # Use F1 as the metric
        
        result = {
            'trial': trial + 1,
            'learning_rate': lr,
            'dropout_rate': dropout,
            'hidden_dims': hidden,
            'weight_decay': wd,
            'val_accuracy': val_results['accuracy'],
            'val_f1': val_results['f1'],
            'val_precision': val_results['precision'],
            'val_recall': val_results['recall']
        }
        all_results.append(result)
        
        print(f"  Val Accuracy: {val_results['accuracy']:.4f}, Val F1: {val_results['f1']:.4f}")
        
        # Update best
        if val_score > best_score:
            best_score = val_score
            best_params = {
                'learning_rate': lr,
                'dropout_rate': dropout,
                'hidden_dims': hidden,
                'weight_decay': wd
            }
            print(f"  ✓ New best configuration!")
    
    # Print summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 80)
    print(f"Best configuration:")
    print(f"  Learning Rate: {best_params['learning_rate']}")
    print(f"  Dropout Rate: {best_params['dropout_rate']}")
    print(f"  Hidden Dims: {best_params['hidden_dims']}")
    print(f"  Weight Decay: {best_params['weight_decay']}")
    print(f"  Best Val F1: {best_score:.4f}")
    print("=" * 80)
    
    return best_params, all_results


def save_tuning_results(best_params, all_results, save_path):
    """Save hyperparameter tuning results."""
    results = {
        'best_parameters': best_params,
        'all_trials': all_results
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nHyperparameter tuning results saved to {save_path}")

