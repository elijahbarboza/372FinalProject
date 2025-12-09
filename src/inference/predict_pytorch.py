"""
Predict NBA player from a face image using PyTorch model.
"""
import os
import sys
import torch
import numpy as np
import argparse
import joblib

try:
    import face_recognition
except ImportError:
    print("Error: face_recognition library not found. Please install it.")
    sys.exit(1)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.face_classifier import FaceClassifier
from src.utils.player_utils import get_player_info, load_player_data


# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAYERS_CSV = os.path.join(PROJECT_ROOT, "data", "players.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_classifier.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
PLAYER_MAP_PATH = os.path.join(MODELS_DIR, "player_id_to_name.pkl")


class PlayerPredictor:
    """NBA Player Face Recognition Predictor using PyTorch."""
    
    def __init__(self):
        """Initialize the predictor by loading models and data."""
        print("Loading models...")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load scaler and label encoder
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Please train the model first.")
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}. Please train the model first.")
        
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        num_classes = len(self.label_encoder.classes_)
        
        # Load model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.model = FaceClassifier(input_dim=128, num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load player data
        self.player_df = load_player_data(PLAYERS_CSV)
        with open(PLAYER_MAP_PATH, 'rb') as f:
            import pickle
            self.player_id_to_name = pickle.load(f)
        
        print(f"✓ Loaded model for {num_classes} players")
    
    def get_embedding(self, image_path):
        """Extract face embedding from an image."""
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                return None
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            if len(face_encodings) == 0:
                return None
            
            return face_encodings[0]
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def predict(self, image_path, top_k=5):
        """Predict NBA player from an image."""
        emb = self.get_embedding(image_path)
        if emb is None:
            return None
        
        # Normalize embedding
        emb_normalized = self.scaler.transform([emb])
        emb_tensor = torch.tensor(emb_normalized, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(emb_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            player_id = self.label_encoder.inverse_transform([idx])[0]
            confidence = top_probs[0][i].item()
            player_name = self.player_id_to_name.get(str(player_id), "Unknown")
            player_info = get_player_info(player_id, self.player_df)
            
            results.append({
                'player_id': str(player_id),
                'name': player_name,
                'confidence': confidence,
                'info': player_info
            })
        
        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Predict NBA player from face image')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
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
            # print(f"   Confidence: {result['confidence']:.2%}")
            if result['info']:
                info = result['info']
                print(f"   Position: {info.get('position', 'N/A')}")
                print(f"   School: {info.get('school', 'N/A')}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

