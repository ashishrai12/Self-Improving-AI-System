import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base_model import BaseModel
from models.critic_model import CriticModel
from evaluation.metrics import compute_metrics
import numpy as np

def test_base_model():
    config_path = 'config/config.yaml'
    model = BaseModel(config_path)
    X, y = model.generate_data()
    model.train(X, y)
    pred = model.predict(X[:10])
    assert len(pred) == 10
    print("Base model test passed.")

def test_critic():
    config_path = 'config/config.yaml'
    critic = CriticModel(config_path)
    # H(0.98, 0.02) = 0.14 (Low entropy, High quality)
    # H(0.5, 0.5) = 1.0 (High entropy, Low quality)
    proba = np.array([[0.98, 0.02], [0.5, 0.5]])
    quality = critic.evaluate(proba)
    assert quality[0] == True
    assert quality[1] == False
    print("Critic test passed.")

def test_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    metrics = compute_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    print("Metrics test passed.")

if __name__ == "__main__":
    test_base_model()
    test_critic()
    test_metrics()
    print("All tests passed.")
