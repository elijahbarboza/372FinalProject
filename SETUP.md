NBA Player Face Recognition - Quick Setup
1. Install Dependencies
# Install Python packagespip install -r requirements.txt# Install face recognition (macOS)brew install cmakepip install dlib face-recognition
2. Train the Model
# Navigate to project directorycd /Users/elijahbarboza/Documents/Code/CS\ 372/Final/372FinalProject# Train the modelpython src/training/train_pytorch.py
3. Run Predictions
# Predict from an imagepython src/inference/predict_pytorch.py path/to/image.png# Examplepython src/inference/predict_pytorch.py data/testimg/wendellmoore.png
Output Files
After training, you'll find:
src/models/pytorch_classifier.pth - Trained model
src/models/training_curves.png - Training progress
src/models/scaler.pkl - Data normalizer
src/models/label_encoder.pkl - Class labels

Requirements
Python 3.8+
PyTorch
face-recognition library
Standard ML libraries (numpy, pandas, scikit-learn)
The setup takes ~10-20 minutes total. Training processes 544 NBA player images and creates a neural network that can identify players from face photos.