#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    # Generate simulation data split
    X, y_true = base_model.generate_data()
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)
    
    # We want a very small initial X_train to show improvement from active learning
    X_train_initial, X_pool, y_train_initial, y_pool = train_test_split(X_train_temp, y_train_temp, train_size=0.05, random_state=42)
    
    # Train initial naive model
    base_model.train(X_train_initial, y_train_initial)
    base_model.save()

    current_X_train = np.copy(X_train_initial)
    current_y_train = np.copy(y_train_initial)
    
    accuracies = []
    iterations_list = []

    for iteration in range(15):  # simulate up to 15 iterations
        base_model.load()
        predictions = base_model.predict(X_test)
        
        accuracy = np.mean(predictions == y_test)
        accuracies.append(accuracy)
        iterations_list.append(iteration)
        
        # We query the unlabelled pool
        proba_pool = base_model.predict_proba(X_pool)
        quality_pool = critic.evaluate(proba_pool)
        
        feedback_X = []
        feedback_y = []
        keep_indices = []
        
        for i in range(len(X_pool)):
            # If critic evaluates poor quality (uncertainty), we request label
            if not quality_pool[i] and len(feedback_X) < retrainer.config['feedback']['retrain_batch_size']:
                feedback_X.append(X_pool[i])
                feedback_y.append(y_pool[i])
            else:
                keep_indices.append(i)
                
        print(f"Iteration {iteration}: Accuracy {accuracy:.3f}, Feedback collected: {len(feedback_X)}")

        if len(feedback_X) > 0:
            X_pool = X_pool[keep_indices]
            y_pool = y_pool[keep_indices]
            feedback_X = np.array(feedback_X)
            feedback_y = np.array(feedback_y)
            
            # Update trainer params natively using active learning batch
            current_X_train, current_y_train = retrainer.retrain(
                base_model, current_X_train, current_y_train, feedback_X, feedback_y
            )
        else:
            print("No more low-quality predictions in the pool. Convergence reached.")
            break

    # Generate visual simulation plot
    os.makedirs('experiments', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title('Self-Improving AI: Accuracy over Feedback Loop Iterations', fontsize=14)
    plt.xlabel('Feedback Loop Iteration', fontsize=12)
    plt.ylabel('Model Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(iterations_list)
    plt.ylim(max(0, min(accuracies) - 0.05), min(1.0, max(accuracies) + 0.05))
    plt.tight_layout()
    plot_path = os.path.join('experiments', 'accuracy_plot.png')
    plt.savefig(plot_path)
    print(f"Visual simulation plot saved to {plot_path}")

if __name__ == "__main__":
    main()
