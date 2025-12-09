"""
Train a face recognition classifier for NBA players.
"""
import os
import sys
import cv2
import numpy as np
import pickle
import joblib
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.player_utils import load_player_data, create_player_id_to_name_map

try:
    import openface
except ImportError:
    print("Error: openface library not found. Please install it.")
    print("You may need to install OpenFace: https://github.com/cmusatyalab/openface")
    sys.exit(1)


# ----------------------------
# Configuration
# ----------------------------
# Get the project root directory (assuming this file is in src/inference/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "img")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nn4.small2.def.lua")
PLAYERS_CSV = os.path.join(PROJECT_ROOT, "data", "players.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "models")
CLASSIFIER_PATH = os.path.join(OUTPUT_DIR, "player_classifier.pkl")
PLAYER_MAP_PATH = os.path.join(OUTPUT_DIR, "player_id_to_name.pkl")
IMG_DIM = 96   # OpenFace expects 96x96 input

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Initialize the OpenFace model
# ----------------------------
print("Loading OpenFace model...")
try:
    embedder = openface.TorchNeuralNet(MODEL_PATH, IMG_DIM)
    print("OpenFace model loaded successfully!")
except Exception as e:
    print(f"Error loading OpenFace model: {e}")
    print(f"Model path: {MODEL_PATH}")
    sys.exit(1)


# ----------------------------
# Helper: extract embedding for an image
# ----------------------------
def get_embedding(image_path):
    """
    Extract face embedding from an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Face embedding vector or None if extraction fails
    """
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        return None
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    try:
        rep = embedder.forward(rgb_img)
        return rep
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# ----------------------------
# Load dataset and compute embeddings
# ----------------------------
print("\nLoading player data...")
player_df = load_player_data(PLAYERS_CSV)
player_id_to_name = create_player_id_to_name_map(PLAYERS_CSV)
print(f"Loaded data for {len(player_df)} players")

print("\nLoading images and computing embeddings...")
X, y = [], []

# Get all PNG files in data/img directory
image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
print(f"Found {len(image_files)} images")

for img_name in tqdm(image_files, desc="Processing images"):
    # Extract player ID from filename (e.g., "101108.png" -> "101108")
    player_id = os.path.splitext(img_name)[0]
    
    # Skip if player ID not in our database
    if player_id not in player_id_to_name:
        continue
    
    img_path = os.path.join(DATA_DIR, img_name)
    emb = get_embedding(img_path)
    
    if emb is not None:
        X.append(emb)
        y.append(player_id)  # Store player ID as label

if len(X) == 0:
    print("Error: No valid embeddings extracted. Please check your images and model.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)

print(f"\nExtracted {len(X)} face embeddings from {len(set(y))} unique players")

# ----------------------------
# Train/test split
# ----------------------------
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ----------------------------
# Train a linear SVM classifier
# ----------------------------
print("\nTraining SVM classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
print("Training complete!")

# ----------------------------
# Evaluate
# ----------------------------
print("\nEvaluating classifier...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save the classifier and player mapping
# ----------------------------
print("\nSaving model and mappings...")
joblib.dump(clf, CLASSIFIER_PATH)
print(f"Classifier saved to: {CLASSIFIER_PATH}")

with open(PLAYER_MAP_PATH, 'wb') as f:
    pickle.dump(player_id_to_name, f)
print(f"Player mapping saved to: {PLAYER_MAP_PATH}")

print("\nTraining complete! You can now use the predictor.")

