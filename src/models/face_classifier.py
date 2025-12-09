"""
Custom PyTorch neural network for NBA player face recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceClassifier(nn.Module):
    """
    Custom neural network for classifying NBA players from face embeddings.
    
    Architecture:
    - Input: 128-dimensional face embeddings (from face_recognition)
    - Hidden layers with batch normalization and dropout
    - Output: num_classes (number of NBA players)
    """
    
    def __init__(self, input_dim=128, hidden_dims=[256, 128], num_classes=500, 
                 dropout_rate=0.3, use_batch_norm=True):
        """
        Initialize the face classifier.
        
        Args:
            input_dim: Dimension of input face embeddings (default: 128)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of NBA players to classify
            dropout_rate: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super(FaceClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability distribution over classes
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class SimpleFaceClassifier(nn.Module):
    """
    Simpler version for ablation studies.
    """
    
    def __init__(self, input_dim=128, num_classes=500):
        super(SimpleFaceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

