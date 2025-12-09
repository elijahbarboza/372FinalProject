"""
PyTorch Dataset class for NBA player face recognition.
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import face_recognition
from PIL import Image
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    """
    Dataset class for loading NBA player face images and embeddings.
    """
    
    def __init__(self, image_paths, labels, player_id_to_idx, 
                 use_embeddings=True, augment=False, image_size=(224, 224)):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of player IDs (strings)
            player_id_to_idx: Dictionary mapping player ID to class index
            use_embeddings: If True, use pre-computed embeddings; if False, load raw images
            augment: Whether to apply data augmentation
            image_size: Target image size if loading raw images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.player_id_to_idx = player_id_to_idx
        self.use_embeddings = use_embeddings
        self.augment = augment
        self.image_size = image_size
        
        # Convert labels to indices
        self.label_indices = [player_id_to_idx[label] for label in labels]
        
        # Data augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(15),  # Random rotation ±15 degrees
                transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            If use_embeddings: (embedding, label_idx)
            If not: (image_tensor, label_idx)
        """
        image_path = self.image_paths[idx]
        label_idx = self.label_indices[idx]
        
        if self.use_embeddings:
            # Load and extract embedding
            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    # Return zero embedding if no face found
                    embedding = np.zeros(128, dtype=np.float32)
                else:
                    face_encodings = face_recognition.face_encodings(image, face_locations)
                    if len(face_encodings) > 0:
                        embedding = face_encodings[0].astype(np.float32)
                    else:
                        embedding = np.zeros(128, dtype=np.float32)
                
                return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)
            except Exception as e:
                # Return zero embedding on error
                return torch.zeros(128, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)
        else:
            # Load raw image
            try:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                
                if self.augment:
                    image = self.transform(image)
                else:
                    image = self.transform(image)
                
                return image, torch.tensor(label_idx, dtype=torch.long)
            except Exception as e:
                # Return black image on error
                return torch.zeros(3, *self.image_size, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


class EmbeddingDataset(Dataset):
    """
    Simplified dataset for pre-computed embeddings (faster training).
    """
    
    def __init__(self, embeddings, labels):
        """
        Initialize with pre-computed embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) with class indices
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

