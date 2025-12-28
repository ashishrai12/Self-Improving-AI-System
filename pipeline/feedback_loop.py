#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base_model import BaseModel
from models.critic_model import CriticModel
from models.retrainer import Retrainer

def main():
    config_path = 'config/config.yaml'
    base_model = BaseModel(config_path)
    critic = CriticModel(config_path)
    retrainer = Retrainer(config_path)

    # Initial train if not exists
    if not os.path.exists(base_model.model_path):
        X, y = base_model.generate_data()
        base_model.train(X, y)
        base_model.save()

    # Generate test data
    X, y_true = base_model.generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

    feedback_X = []
    feedback_y = []

    for iteration in range(5):  # simulate 5 iterations
        base_model.load()
        predictions = base_model.predict(X_test)
        proba = base_model.predict_proba(X_test)
        quality = critic.evaluate(proba)

        # Collect feedback: misclassified or low quality
        for i in range(len(X_test)):
            if not quality[i] or predictions[i] != y_test[i]:
                feedback_X.append(X_test[i])
                feedback_y.append(y_test[i])

        print(f"Iteration {iteration}: Accuracy {np.mean(predictions == y_test)}, Feedback collected: {len(feedback_X)}")

        # Retrain if enough feedback
        if len(feedback_X) >= retrainer.config['feedback']['retrain_batch_size']:
            feedback_X = np.array(feedback_X)
            feedback_y = np.array(feedback_y)
            retrainer.retrain(base_model, feedback_X, feedback_y)
            feedback_X = []
            feedback_y = []

if __name__ == "__main__":
    main()
