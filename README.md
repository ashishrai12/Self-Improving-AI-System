# Self-Improving AI System

## Problem Statement

This project implements a self-improving AI system that learns from its own mistakes through an automated feedback loop. The system consists of a base prediction model, a critic that evaluates output quality, and a retraining mechanism that incorporates feedback to improve performance over time. The goal is to demonstrate continuous learning in a controlled, local environment without external dependencies.

## Architecture

```
+----------------+     +----------------+     +----------------+
|  Base Model    | --> |   Critic       | --> |  Feedback      |
| (Prediction)   |     | (Evaluation)   |     |  Loop          |
+----------------+     +----------------+     +----------------+
       ^                        |                        |
       |                        v                        v
       +------------------- Retrainer -------------------+
```

- **Base Model**: A simple logistic regression model for binary classification.
- **Critic**: Evaluates prediction confidence; flags low-confidence predictions as failures.
- **Feedback Loop**: Collects failed predictions, retrains the base model with combined data.
- **Retrainer**: Handles model retraining with original + feedback data.

## How the Feedback Loop Works

1. The base model makes predictions on test data.
2. The critic evaluates each prediction's quality based on confidence scores.
3. Failed predictions (low quality or incorrect) are stored in the feedback dataset.
4. When sufficient feedback is collected, the retrainer combines original training data with feedback and retrains the model.
5. The process repeats, allowing the model to improve iteratively.

<img width="842" height="491" alt="{C60CD542-9B8B-4B80-9F3E-E607CDED5810}" src="https://github.com/user-attachments/assets/8d4963ef-491b-4cfc-9bdf-ae010533f922" />


## Measuring Improvement

Improvement is measured by tracking accuracy and other metrics across iterations. Logs are stored in `experiments/training.log`. Regression tests ensure system stability.

sample output:
Iteration 0: Accuracy 0.88
Iteration 1: Accuracy 0.89
Iteration 2: Accuracy 0.88
Iteration 3: Accuracy 0.88
Iteration 4: Accuracy 0.88
Iteration 0: Accuracy 0.88
Iteration 1: Accuracy 0.89
Iteration 2: Accuracy 0.88
Iteration 3: Accuracy 0.88
Iteration 4: Accuracy 0.88
Iteration 0: Accuracy 0.88
Iteration 1: Accuracy 0.89
Iteration 2: Accuracy 0.88
Iteration 3: Accuracy 0.88
Iteration 4: Accuracy 0.88
Iteration 0: Accuracy 0.88
Iteration 1: Accuracy 0.89
Iteration 2: Accuracy 0.88
Iteration 3: Accuracy 0.88
Iteration 4: Accuracy 0.88
Iteration 5: Accuracy 0.88
Iteration 6: Accuracy 0.88
Iteration 7: Accuracy 0.88
Iteration 8: Accuracy 0.88
Iteration 9: Accuracy 0.88


<img width="986" height="617" alt="{A704AE9D-A6DA-4981-8118-6774F7CC72C0}" src="https://github.com/user-attachments/assets/8ff17e71-3068-4c81-a10d-488da9fcaa4c" />


## How to Run the Project End-to-End

1. Ensure Python 3.8+ and required packages: `pip install scikit-learn pyyaml pandas numpy`
2. Run initial training: `python pipeline/train.py`
3. Run inference test: `python pipeline/infer.py`
4. Run feedback loop simulation: `python pipeline/feedback_loop.py`
5. Run tests: `python evaluation/regression_tests.py`

The system generates synthetic data for demonstration. For real data, place CSV files in `data/raw/` and modify `base_model.py` accordingly.
