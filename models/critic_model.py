import yaml
import numpy as np
from .uncertainty import calculate_shannon_entropy

class CriticModel:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.threshold = self.config['model']['critic_threshold']
        # Sophisticated entropy threshold: H(Y|X) > H_max * (1 - threshold)
        # For binary, max entropy is 1.0 (at p=0.5)
        self.entropy_threshold = 1.0 - self.threshold 

    def evaluate(self, proba):
        """
        Evaluate the quality of predictions using Shannon Entropy.
        Returns True if high quality (low entropy), False otherwise.
        """
        entropy = calculate_shannon_entropy(proba)
        # Low entropy means high confidence
        return entropy <= self.entropy_threshold
