# How to Train the Classifier

You have **two options** to train the classifier:

## Option 1: Use the Jupyter Notebook (Recommended)

1. **Open the notebook:**
   ```bash
   jupyter notebook notebooks/train_classifier.ipynb
   ```
   Or if you're using JupyterLab:
   ```bash
   jupyter lab notebooks/train_classifier.ipynb
   ```

2. **Run all cells** - The notebook is organized into logical steps that you can run cell by cell or all at once.

3. **Advantages:**
   - See intermediate results
   - Easy to debug
   - Can modify parameters interactively
   - Visual progress bars

## Option 2: Run as Python Script

1. **First, fix the numpy/scikit-learn compatibility issue:**
   ```bash
   # Option A: Downgrade numpy (recommended)
   pip install "numpy<2.0"
   
   # OR Option B: Upgrade scikit-learn
   pip install --upgrade scikit-learn
   ```

2. **Run the training script:**
   ```bash
   python src/inference/train.py
   ```

## Fixing the NumPy Compatibility Issue

You're seeing an error because NumPy 2.x removed `ComplexWarning` which older scikit-learn versions need. 

**Quick fix:**
```bash
pip install "numpy<2.0" "scikit-learn>=1.0.0"
```

Or if you prefer to upgrade scikit-learn:
```bash
pip install --upgrade scikit-learn
```

## What Happens During Training

1. **Loads player data** from `data/players.csv`
2. **Loads all images** from `data/img/` directory
3. **Extracts face embeddings** using OpenFace (this takes time!)
4. **Trains an SVM classifier** on the embeddings
5. **Evaluates** the model on a test set
6. **Saves** the trained model to `src/models/player_classifier.pkl`

## Expected Output

After training, you should see:
- Number of images processed
- Number of unique players
- Training/test set sizes
- **Accuracy score** (how well it performs)
- **Classification report** (detailed metrics)
- Confirmation that models were saved

## After Training

Once training is complete, you can use the predictor:

```bash
python src/inference/predict.py path/to/image.png
```

Or in Python:
```python
from src.inference.predict import PlayerPredictor
predictor = PlayerPredictor()
results = predictor.predict('path/to/image.png')
```

## Troubleshooting

### "openface library not found"
- Make sure OpenFace is installed
- You may need: `pip install openface` or install from source

### "No valid embeddings extracted"
- Check that images are valid PNG files
- Ensure images contain faces
- Verify the OpenFace model file exists at `models/nn4.small2.def.lua`

### Training takes a long time
- This is normal! Extracting embeddings from 500+ images can take 10-30 minutes
- The progress bar will show you how many images are left

