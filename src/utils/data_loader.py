"""
Utility functions for loading and preparing data.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import face_recognition
from tqdm import tqdm

from src.data.dataset import EmbeddingDataset
from src.utils.player_utils import load_player_data, create_player_id_to_name_map


def load_and_prepare_data(data_dir, players_csv, train_ratio=0.7, val_ratio=0.15, 
                          test_ratio=0.15, batch_size=32, random_seed=42):
    """
    Load images, extract embeddings, normalize, and create data loaders.
    
    Returns:
        Dictionary with all data loaders, scaler, label encoder, and metadata
    """
    # Load player data
    player_df = load_player_data(players_csv)
    player_id_to_name = create_player_id_to_name_map(players_csv)
    
    # Load images
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    image_paths = []
    labels = []
    
    for img_name in image_files:
        player_id = os.path.splitext(img_name)[0]
        if player_id in player_id_to_name:
            image_paths.append(os.path.join(data_dir, img_name))
            labels.append(player_id)
    
    # Extract embeddings
    print("Extracting face embeddings...")
    embeddings = []
    valid_paths = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                continue
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if len(face_encodings) > 0:
                embeddings.append(face_encodings[0])
                valid_paths.append(img_path)
        except Exception as e:
            continue
    
    # Filter labels
    labels = [labels[i] for i, path in enumerate(image_paths) if path in valid_paths]
    embeddings = np.array(embeddings)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    # Create dataset
    dataset = EmbeddingDataset(embeddings_normalized, label_indices)
    
    # Split
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'embeddings': embeddings_normalized,
        'labels': label_indices,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'player_id_to_name': player_id_to_name,
        'num_classes': num_classes,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test
    }

