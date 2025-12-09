# NBA Player Face Recognition System

A machine learning system that identifies NBA players from face images using deep learning and computer vision techniques.

## What it Does

This project implements a complete NBA player face recognition pipeline that processes headshot images of professional basketball players. The system uses face detection and embedding extraction to create 128-dimensional feature vectors, then trains a custom PyTorch neural network to classify players by their facial features. The trained model can identify NBA players from new photos with confidence scores, making it useful for sports analytics, fan engagement applications, and automated player identification in video content.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
brew install cmake && pip install dlib face-recognition
```

### Training
```bash
python src/training/train_pytorch.py
```

### Prediction
```bash
python src/inference/predict_pytorch.py data/testimg/wendellmoore.png
```

## Video Links 
Final Demo: https://youtu.be/x9gqUnuLMO8
Technical Walkthrough: https://youtu.be/Af6EujTZ-GM

## Evaluation
The model was trained on 544 NBA player images (one image per player) and evaluated on the training set:

- **Training Accuracy**: 87.4%
- **Precision**: 0.89
- **Recall**: 0.87  
- **F1 Score**: 0.88

### Model Architecture Performance

| Component | Specification |
|-----------|---------------|
| Input Dimension | 128-dimensional face embeddings |
| Hidden Layers | 256 → 128 neurons |
| Regularization | Dropout (0.3), BatchNorm, L2 penalty |
| Training Epochs | 50 (early stopping implemented) |
| Optimizer | Adam (lr=0.001, weight_decay=0.0001) |

### Baseline Comparison

| Model | Accuracy |
|-------|----------|
| SVM (previous approach) | 72.1% |
| **Neural Network** | **87.4%** |

The neural network shows significant improvement over traditional machine learning approaches, demonstrating the value of deep learning for this computer vision task.

## Individual Contributions
I worked on this alone, so everything in the project has been contributed by me. 
