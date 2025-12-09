"""
Baseline models for comparison.
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn


class RandomBaseline:
    """Random classifier baseline."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def fit(self, X, y):
        """Fit the baseline (no-op for random)."""
        pass
    
    def predict(self, X):
        """Predict random classes."""
        return np.random.randint(0, self.num_classes, size=len(X))
    
    def predict_proba(self, X):
        """Predict random probabilities."""
        probs = np.random.rand(len(X), self.num_classes)
        return probs / probs.sum(axis=1, keepdims=True)


class MajorityClassBaseline:
    """Majority class baseline."""
    
    def __init__(self):
        self.majority_class = None
    
    def fit(self, X, y):
        """Fit by finding majority class."""
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
    
    def predict(self, X):
        """Predict majority class for all samples."""
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X):
        """Predict probabilities (1.0 for majority class, 0.0 for others)."""
        probs = np.zeros((len(X), len(np.unique(self.majority_class)) + 1))
        probs[:, self.majority_class] = 1.0
        return probs


class SklearnBaseline:
    """Wrapper for sklearn dummy classifiers."""
    
    def __init__(self, strategy='most_frequent'):
        """
        Initialize sklearn baseline.
        
        Args:
            strategy: 'most_frequent', 'stratified', 'uniform', 'constant'
        """
        self.clf = DummyClassifier(strategy=strategy, random_state=42)
    
    def fit(self, X, y):
        """Fit the baseline."""
        self.clf.fit(X, y)
    
    def predict(self, X):
        """Predict using baseline."""
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.clf.predict_proba(X)


def evaluate_baselines(X_train, y_train, X_test, y_test, num_classes):
    """
    Evaluate all baseline models.
    
    Returns:
        Dictionary with baseline results
    """
    results = {}
    
    # Random baseline
    random_baseline = RandomBaseline(num_classes)
    random_baseline.fit(X_train, y_train)
    y_pred_random = random_baseline.predict(X_test)
    results['random'] = accuracy_score(y_test, y_pred_random)
    
    # Majority class baseline
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(X_train, y_train)
    y_pred_majority = majority_baseline.predict(X_test)
    results['majority_class'] = accuracy_score(y_test, y_pred_majority)
    
    # Sklearn baselines
    for strategy in ['most_frequent', 'stratified', 'uniform']:
        baseline = SklearnBaseline(strategy=strategy)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        results[f'sklearn_{strategy}'] = accuracy_score(y_test, y_pred)
    
    return results

