# NBA Player Face Recognition System

This system uses OpenFace to recognize NBA players from their face images.

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install OpenFace

You need to install OpenFace separately. The easiest way is:

**Option A: Using pip (if available)**
```bash
pip install openface
```

**Option B: Install from source**
Follow the instructions at: https://github.com/cmusatyalab/openface

**Option C: Using conda**
```bash
conda install -c conda-forge openface
```

**Note:** OpenFace requires Torch (PyTorch) to be installed. Make sure you have it:
```bash
pip install torch torchvision
```

### 3. Verify Model File

Make sure the model file exists at: `models/nn4.small2.def.lua`

## Usage

### Step 1: Train the Classifier

First, you need to train the classifier on your player images:

```bash
python src/inference/train.py
```

This will:
- Load all images from `data/img/`
- Extract face embeddings using OpenFace
- Train an SVM classifier
- Save the trained model to `src/models/player_classifier.pkl`
- Save player ID mappings to `src/models/player_id_to_name.pkl`

### Step 2: Make Predictions

Once trained, you can predict players from new images:

```bash
python src/inference/predict.py path/to/image.png
```

To see top 5 predictions:
```bash
python src/inference/predict.py path/to/image.png --top-k 5
```

### Using in Python Code

```python
from src.inference.predict import PlayerPredictor

# Initialize predictor (loads models)
predictor = PlayerPredictor()

# Predict from an image
results = predictor.predict('path/to/image.png', top_k=5)

# Print results
for result in results:
    print(f"{result['name']}: {result['confidence']:.2%}")
```

## Project Structure

```
372FinalProject/
├── data/
│   ├── img/              # Player headshot images (PNG files named by player ID)
│   └── players.csv        # Player information database
├── models/
│   └── nn4.small2.def.lua # OpenFace model definition
├── src/
│   ├── inference/
│   │   ├── train.py       # Training script
│   │   └── predict.py     # Prediction script
│   ├── models/            # Saved models (created after training)
│   │   ├── player_classifier.pkl
│   │   └── player_id_to_name.pkl
│   └── utils/
│       └── player_utils.py # Utility functions
└── requirements.txt
```

## Data Format

- **Images**: PNG files in `data/img/` named `{playerid}.png` (e.g., `101108.png`)
- **CSV**: `data/players.csv` with columns: `playerid`, `fname`, `lname`, `position`, etc.

## Troubleshooting

### "openface library not found"
- Make sure OpenFace is installed (see Setup step 2)
- You may need to install Torch/PyTorch first

### "Classifier not found"
- Run `train.py` first to create the classifier

### "Could not read image"
- Check that the image path is correct
- Ensure the image is a valid image file
- The image should contain a face

### Low accuracy
- Make sure you have enough training images per player
- Check that images are clear and contain faces
- Consider data augmentation or more training data

