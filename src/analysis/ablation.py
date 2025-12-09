"""
Ablation study script to evaluate impact of different design choices.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.face_classifier import FaceClassifier, SimpleFaceClassifier
from src.data.dataset import EmbeddingDataset
from src.training.trainer import Trainer, evaluate_model
from src.utils.player_utils import load_player_data, create_player_id_to_name_map
import face_recognition
from tqdm import tqdm


def run_ablation_study(embeddings, labels, device, num_classes, 
                       train_loader, val_loader, test_loader):
    """
    Run ablation study comparing different configurations.
    
    Returns:
        Dictionary with results for each configuration
    """
    results = {}
    
    # Baseline: Simple model without batch norm and dropout
    print("\n1. Testing: Simple model (no batch norm, no dropout)")
    model_simple = SimpleFaceClassifier(input_dim=embeddings.shape[1], num_classes=num_classes)
    optimizer = optim.Adam(model_simple.parameters(), lr=0.001)
    trainer = Trainer(model_simple.to(device), device)
    trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
    test_results = evaluate_model(model_simple, test_loader, device)
    results['simple'] = {
        'accuracy': test_results['accuracy'],
        'f1': test_results['f1'],
        'description': 'Simple model (no batch norm, no dropout)'
    }
    
    # With batch normalization
    print("\n2. Testing: Model with batch normalization")
    model_bn = FaceClassifier(
        input_dim=embeddings.shape[1],
        num_classes=num_classes,
        dropout_rate=0.0,  # No dropout
        use_batch_norm=True
    )
    optimizer = optim.Adam(model_bn.parameters(), lr=0.001)
    trainer = Trainer(model_bn.to(device), device)
    trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
    test_results = evaluate_model(model_bn, test_loader, device)
    results['with_batch_norm'] = {
        'accuracy': test_results['accuracy'],
        'f1': test_results['f1'],
        'description': 'Model with batch normalization'
    }
    
    # With dropout
    print("\n3. Testing: Model with dropout")
    model_dropout = FaceClassifier(
        input_dim=embeddings.shape[1],
        num_classes=num_classes,
        dropout_rate=0.3,
        use_batch_norm=False
    )
    optimizer = optim.Adam(model_dropout.parameters(), lr=0.001)
    trainer = Trainer(model_dropout.to(device), device)
    trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
    test_results = evaluate_model(model_dropout, test_loader, device)
    results['with_dropout'] = {
        'accuracy': test_results['accuracy'],
        'f1': test_results['f1'],
        'description': 'Model with dropout (0.3)'
    }
    
    # Full model (batch norm + dropout)
    print("\n4. Testing: Full model (batch norm + dropout)")
    model_full = FaceClassifier(
        input_dim=embeddings.shape[1],
        num_classes=num_classes,
        dropout_rate=0.3,
        use_batch_norm=True
    )
    optimizer = optim.Adam(model_full.parameters(), lr=0.001)
    trainer = Trainer(model_full.to(device), device)
    trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
    test_results = evaluate_model(model_full, test_loader, device)
    results['full_model'] = {
        'accuracy': test_results['accuracy'],
        'f1': test_results['f1'],
        'description': 'Full model (batch norm + dropout)'
    }
    
    # Different learning rates
    print("\n5. Testing: Different learning rates")
    for lr in [0.0001, 0.001, 0.01]:
        model = FaceClassifier(
            input_dim=embeddings.shape[1],
            num_classes=num_classes,
            dropout_rate=0.3,
            use_batch_norm=True
        )
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model.to(device), device)
        trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
        test_results = evaluate_model(model, test_loader, device)
        results[f'lr_{lr}'] = {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'description': f'Learning rate: {lr}'
        }
    
    return results


def compare_optimizers(model_class, embeddings, num_classes, train_loader, val_loader, 
                       test_loader, device):
    """
    Compare different optimizers.
    
    Returns:
        Dictionary with results for each optimizer
    """
    results = {}
    optimizers_config = {
        'SGD': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        'Adam': lambda params: optim.Adam(params, lr=0.001),
        'AdamW': lambda params: optim.AdamW(params, lr=0.001, weight_decay=0.01)
    }
    
    for opt_name, opt_fn in optimizers_config.items():
        print(f"\nTesting optimizer: {opt_name}")
        model = model_class(
            input_dim=embeddings.shape[1],
            num_classes=num_classes,
            dropout_rate=0.3,
            use_batch_norm=True
        )
        optimizer = opt_fn(model.parameters())
        trainer = Trainer(model.to(device), device)
        trainer.train(train_loader, val_loader, optimizer, None, num_epochs=20, save_path=None)
        test_results = evaluate_model(model, test_loader, device)
        results[opt_name] = {
            'accuracy': test_results['accuracy'],
            'f1': test_results['f1'],
            'precision': test_results['precision'],
            'recall': test_results['recall']
        }
    
    return results


def print_ablation_results(results):
    """Print ablation study results in a formatted table."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Accuracy':<15} {'F1 Score':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{result['description']:<30} {result['accuracy']:<15.4f} {result['f1']:<15.4f}")
    
    print("=" * 80)


def save_ablation_results(results, save_path):
    """Save ablation results to JSON."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {save_path}")

