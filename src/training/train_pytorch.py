"""
Main training script for PyTorch neural network.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.face_classifier import FaceClassifier
from src.models.baseline import evaluate_baselines
from src.data.dataset import EmbeddingDataset
from src.training.trainer import Trainer, evaluate_model
from src.utils.player_utils import load_player_data, create_player_id_to_name_map
import face_recognition
from tqdm import tqdm


# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "img")
PLAYERS_CSV = os.path.join(PROJECT_ROOT, "data", "players.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "models")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "face_classifier.pth")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
PLAYER_MAP_PATH = os.path.join(OUTPUT_DIR, "player_id_to_name.pkl")
CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001  # L2 regularization
DROPOUT_RATE = 0.3
HIDDEN_DIMS = [256, 128]

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_embeddings(image_paths):
    """Extract face embeddings from images."""
    embeddings = []
    valid_paths = []
    
    print("Extracting face embeddings...")
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
    
    return np.array(embeddings), valid_paths


def main():
    """Main training function."""
    print("=" * 60)
    print("NBA Player Face Recognition - PyTorch Training")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load player data
    print("\n1. Loading player data...")
    player_df = load_player_data(PLAYERS_CSV)
    player_id_to_name = create_player_id_to_name_map(PLAYERS_CSV)
    print(f"   Loaded {len(player_df)} players")
    
    # Load images
    print("\n2. Loading images...")
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    image_paths = []
    labels = []
    
    for img_name in image_files:
        player_id = os.path.splitext(img_name)[0]
        if player_id in player_id_to_name:
            image_paths.append(os.path.join(DATA_DIR, img_name))
            labels.append(player_id)
    
    print(f"   Found {len(image_paths)} valid images")
    
    # Extract embeddings
    print("\n3. Extracting face embeddings...")
    embeddings, valid_paths = extract_embeddings(image_paths)
    # Filter labels to match valid embeddings
    labels = [labels[i] for i, path in enumerate(image_paths) if path in valid_paths]
    
    print(f"   Extracted {len(embeddings)} embeddings")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Encode labels
    print("\n4. Encoding labels...")
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"   Number of classes: {num_classes}")
    
    # Normalize embeddings
    print("\n5. Normalizing embeddings...")
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    print("   ✓ Embeddings normalized")
    
   # Use all data for training (no split)
    print("\n6. Preparing training data...")
    dataset = EmbeddingDataset(embeddings_normalized, label_indices)
    n_total = len(dataset)

# Create single data loader for all data
    print("\n7. Creating data loader...")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"   Total samples: {n_total}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   ✓ Data loader created with shuffling")
    
    # # Evaluate baselines
    # print("\n8. Evaluating baseline models...")
    # # Get baseline data (use train/val split for baselines)
    # X_train_base = embeddings_normalized[:n_train]
    # y_train_base = label_indices[:n_train]
    # X_val_base = embeddings_normalized[n_train:n_train+n_val]
    # y_val_base = label_indices[n_train:n_train+n_val]
    
    # baseline_results = evaluate_baselines(X_train_base, y_train_base, X_val_base, y_val_base, num_classes)
    # print("   Baseline Results:")
    # for name, acc in baseline_results.items():
    #     print(f"     {name}: {acc:.4f}")
    
    # Create model
    print("\n9. Creating model...")
    model = FaceClassifier(
        input_dim=embeddings.shape[1],
        hidden_dims=HIDDEN_DIMS,
        num_classes=num_classes,
        dropout_rate=DROPOUT_RATE,
        use_batch_norm=True
    )
    print(f"   Model architecture:")
    print(f"     Input: {embeddings.shape[1]} dims")
    print(f"     Hidden: {HIDDEN_DIMS}")
    print(f"     Output: {num_classes} classes")
    print(f"     Dropout: {DROPOUT_RATE}")
    print(f"     Batch Norm: True")
    
    # Create optimizer and scheduler
    print("\n10. Setting up optimizer and scheduler...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    print(f"    Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"    Scheduler: ReduceLROnPlateau")
    
    # Train
    print("\n11. Training model on all data...")
    trainer = Trainer(model, device)

    # Train without validation (pass None for val_loader)
    train_losses, _, train_accs, _ = trainer.train(
    train_loader, None, optimizer, scheduler, NUM_EPOCHS,
    save_path=CLASSIFIER_PATH, early_stopping_patience=10)

    
    # Plot training curves
    print("\n12. Plotting training curves...")
    trainer.plot_training_curves(save_path=CURVES_PATH)
    
    # Evaluate on test set
# Evaluate on training data (since no separate test set)
    print("\n13. Evaluating model on training data...")
    model.load_state_dict(torch.load(CLASSIFIER_PATH)['model_state_dict'])
    train_results = evaluate_model(model, train_loader, device)

    print(f"\n   Training Results:")
    print(f"     Accuracy: {train_results['accuracy']:.4f}")
    print(f"     Precision: {train_results['precision']:.4f}")
    print(f"     Recall: {train_results['recall']:.4f}")
    print(f"     F1 Score: {train_results['f1']:.4f}")
    print(f"\n   Note: Model trained and evaluated on all available data")
    print(f"   since each player has only one image.")
    
    # Save models and metadata
    print("\n14. Saving models and metadata...")
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    with open(PLAYER_MAP_PATH, 'wb') as f:
        pickle.dump(player_id_to_name, f)
    
    print(f"   ✓ Model saved to {CLASSIFIER_PATH}")
    print(f"   ✓ Scaler saved to {SCALER_PATH}")
    print(f"   ✓ Label encoder saved to {LABEL_ENCODER_PATH}")
    print(f"   ✓ Training curves saved to {CURVES_PATH}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

