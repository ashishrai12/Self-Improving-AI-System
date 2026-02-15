import os
import yaml
import numpy as np
from .base_model import BaseModel

class Retrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def retrain(self, base_model, X_train, y_train, feedback_X, feedback_y):
        """Retrain the base model with feedback data added to current training set."""
        # Combine with feedback
        X_combined = np.vstack([X_train, feedback_X])
        y_combined = np.hstack([y_train, feedback_y])
        # Retrain
        base_model.train(X_combined, y_combined)
        # Save with versioning, but for simplicity, overwrite for now
        base_model.save()
        return X_combined, y_combined
