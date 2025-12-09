"""
Training utilities for PyTorch models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import os


class Trainer:
    """
    Trainer class for training PyTorch models with tracking.
    """
    
    def __init__(self, model, device, criterion=nn.CrossEntropyLoss()):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: torch.device (cuda or cpu)
            criterion: Loss function
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for embeddings, labels in tqdm(dataloader, desc="Training"):
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(dataloader, desc="Validating"):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return epoch_loss, epoch_acc, precision, recall, f1
    
    def train(self, train_loader, val_loader, optimizer, scheduler, num_epochs, 
              save_path=None, early_stopping_patience=10):
        """
        Train the model with validation.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate (only if validation loader provided)
            if val_loader is not None:
                val_loss, val_acc, precision, recall, f1 = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
            else:
                # No validation - use dummy values
                val_loss = 0.0
                val_acc = 0.0
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            if scheduler:
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model (only if we have validation)
            if val_loader is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    if save_path:
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, save_path)
                        print(f"✓ Saved new best model to {save_path}")
                else:
                    patience_counter += 1
            else:
                # No validation - keep track by train loss
                if train_loss < best_val_acc or best_val_acc == 0: 
                    best_val_acc = train_loss
                    patience_counter = 0
                    if save_path:
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        }, save_path)
                        print(f"✓ Saved new best model (train loss improved) to {save_path}")
                else:
                    patience_counter += 1
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def plot_training_curves(self, save_path=None):
        """Plot training loss and accuracy curves (no validation)."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Loss curve ---
        ax1.plot(self.train_losses, label='Train Loss', color='tab:red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        # --- Accuracy curve ---
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='tab:blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"✓ Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()


def evaluate_model(model, dataloader, device, class_names=None):
    """
    Comprehensive evaluation of the model.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results

