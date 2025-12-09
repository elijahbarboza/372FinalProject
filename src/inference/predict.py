"""
Predict NBA player from a face image.
"""
import os
import sys
import cv2
import numpy as np
import pickle
import joblib
import argparse

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.player_utils import get_player_info, load_player_data

try:
    import openface
except ImportError:
    print("Error: openface library not found. Please install it.")
    print("You may need to install OpenFace: https://github.com/cmusatyalab/openface")
    sys.exit(1)


# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nn4.small2.def.lua")
PLAYERS_CSV = os.path.join(PROJECT_ROOT, "data", "players.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "player_classifier.pkl")
PLAYER_MAP_PATH = os.path.join(MODELS_DIR, "player_id_to_name.pkl")
IMG_DIM = 96


class PlayerPredictor:
    """NBA Player Face Recognition Predictor."""
    
    def __init__(self):
        """Initialize the predictor by loading models and data."""
        print("Loading models...")
        
        # Load OpenFace embedder
        try:
            self.embedder = openface.TorchNeuralNet(MODEL_PATH, IMG_DIM)
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenFace model: {e}")
        
        # Load classifier
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Classifier not found at {CLASSIFIER_PATH}. "
                "Please run train.py first to train the model."
            )
        self.classifier = joblib.load(CLASSIFIER_PATH)
        
        # Load player mapping
        if not os.path.exists(PLAYER_MAP_PATH):
            raise FileNotFoundError(
                f"Player mapping not found at {PLAYER_MAP_PATH}. "
                "Please run train.py first."
            )
        with open(PLAYER_MAP_PATH, 'rb') as f:
            self.player_id_to_name = pickle.load(f)
        
        # Load player data for additional info
        self.player_df = load_player_data(PLAYERS_CSV)
        
        print(f"Loaded classifier for {len(self.classifier.classes_)} players")
    
    def get_embedding(self, image_path):
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
            rep = self.embedder.forward(rgb_img)
            return rep
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def predict(self, image_path, top_k=5):
        """
        Predict NBA player from an image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (player_id, player_name, confidence, player_info)
        """
        emb = self.get_embedding(image_path)
        if emb is None:
            return None
        
        # Get probabilities for all classes
        probs = self.classifier.predict_proba([emb])[0]
        
        # Get top k predictions
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            player_id = self.classifier.classes_[idx]
            confidence = probs[idx]
            player_name = self.player_id_to_name.get(player_id, "Unknown")
            player_info = get_player_info(player_id, self.player_df)
            
            results.append({
                'player_id': player_id,
                'name': player_name,
                'confidence': float(confidence),
                'info': player_info
            })
        
        return results
    
    def predict_single(self, image_path):
        """
        Get the top prediction for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple (player_id, player_name, confidence) or None
        """
        results = self.predict(image_path, top_k=1)
        if results is None or len(results) == 0:
            return None
        return results[0]


def main():
    """Command-line interface for predictions."""
    parser = argparse.ArgumentParser(description='Predict NBA player from face image')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    try:
        predictor = PlayerPredictor()
        results = predictor.predict(args.image_path, top_k=args.top_k)
        
        if results is None:
            print("Error: Could not process image")
            sys.exit(1)
        
        print(f"\nTop {len(results)} predictions:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']} (ID: {result['player_id']})")
            print(f"   Confidence: {result['confidence']:.2%}")
            if result['info']:
                info = result['info']
                print(f"   Position: {info.get('position', 'N/A')}")
                print(f"   Team/Info: {info.get('school', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
