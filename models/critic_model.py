import yaml
import numpy as np

class CriticModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.threshold = self.config['model']['critic_threshold']

    def evaluate(self, proba):
        """
        Evaluate the quality of predictions.
        Returns True if high quality (confidence >= threshold), False otherwise.
        For binary, use max proba.
        """
        confidence = np.max(proba, axis=1)
        return confidence >= self.threshold
