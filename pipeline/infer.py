#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base_model import BaseModel
from models.critic_model import CriticModel
import numpy as np

def main():
    config_path = 'config/config.yaml'
    base_model = BaseModel(config_path)
    base_model.load()
    critic = CriticModel(config_path)

    # Generate test data
    X, y_true = base_model.generate_data()
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    predictions = base_model.predict(X_test)
    proba = base_model.predict_proba(X_test)
    quality = critic.evaluate(proba)

    print(f"Accuracy: {np.mean(predictions == y_test)}")
    print(f"High quality predictions: {np.mean(quality)}")

if __name__ == "__main__":
    main()
