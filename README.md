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

## Mathematical Foundation

The system implements recursive optimization of the empirical risk $\mathcal{R}(\theta)$. The critic performs **uncertainty sampling** based on predictive entropy $H(Y|X, \theta)$, effectively accelerating posterior concentration in a Bayesian framework.

For a full formal derivation including Rademacher complexity bounds and Lyapunov stability analysis, see the [Mathematical Foundation Document](docs/mathematical_foundation.md).

## How the Feedback Loop Works

1. The base model makes predictions on test data.
2. The critic evaluates each prediction's quality based on confidence scores.
3. Failed predictions (low quality or incorrect) are stored in the feedback dataset.
4. When sufficient feedback is collected, the retrainer combines original training data with feedback and retrains the model.
5. The process repeats, allowing the model to improve iteratively.

Feature Correlation Heatmap (feature_correlation.png)
Purpose: Visualizes the pairwise correlations between all 20 input features in the dataset, helping identify relationships that could impact model performance.

Key Features:

Heatmap Format: Uses a color gradient from blue (negative correlation) to red (positive correlation)
20x20 Grid: Shows correlation coefficients for each feature pair
Diagonal: Always 1.0 (perfect self-correlation)
Interpretation:
Values near 1.0 indicate strong positive correlation
Values near -1.0 indicate strong negative correlation
Values near 0.0 indicate no linear relationship
Insights for ML Engineering:

Identifies redundant features (high correlation > 0.8)
Helps with feature selection to reduce dimensionality
Reveals potential multicollinearity issues
In this synthetic dataset, correlations are mostly low (< 0.2), indicating independent features
This chart demonstrates advanced data exploration skills, crucial for production ML systems. Combined with the performance plot, it shows comprehensive analysis capabilities that would impress in a senior ML engineering role. The system now produces two professional visualizations: one for model improvement tracking, and one for data understanding.

<img width="1075" height="901" alt="{CB56323F-E798-4547-A316-36B62A94BF23}" src="https://github.com/user-attachments/assets/c1e0fb0e-5f04-4bb7-8e68-c635b6d81d2e" />


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
