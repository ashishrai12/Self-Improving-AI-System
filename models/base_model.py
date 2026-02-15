import os
import pickle
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import pandas as pd

class BaseModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.model_path = os.path.join('models', 'base_model.pkl')

    def generate_data(self):
        """Generate synthetic data for demonstration."""
        X, y = make_classification(
            n_samples=self.config['data']['n_samples'],
            n_features=self.config['data']['n_features'],
            n_classes=self.config['data']['n_classes'],
            random_state=self.config['data']['random_state']
        )
        return X, y

    def train(self, X, y):
        """Train the base model."""
        self.model = LogisticRegression(
            max_iter=self.config['training']['max_iter'],
            random_state=self.config['training']['random_state']
        )
        self.model.fit(X, y)

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)

    def save(self):
        """Save the model."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self):
        """Load the model."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
