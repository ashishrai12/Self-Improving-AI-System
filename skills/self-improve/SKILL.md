---
name: self-improve
description: Use this skill to interact with the Self-Improving AI System in the current workspace. Use when the user wants to run the training pipeline, run inference, evaluate the feedback loop, or execute regression tests for the python base model and critic.
---

# Self-Improving AI System Skill

This skill provides the commands and procedures needed to orchestrate the polyglot Self-Improving AI System.

## Main Scripts

All scripts must be executed from the root of the project using Python 3.8+.

- **Train the base model**: 
  ```bash
  python pipeline/train.py
  ```
  This script generates a synthetic dataset and trains a logistic regression base model.

- **Run inference (testing)**:
  ```bash
  python pipeline/infer.py
  ```
  This evaluates the trained model on a new synthetic test set.

- **Run the full feedback loop**:
  ```bash
  python pipeline/feedback_loop.py
  ```
  Runs the training, evaluation, and retrains the model on failed predictions based on the Information-Theoretic Uncertainty Sampling Critic.

- **Run the full cross-language orchestration (Python + Rust + Julia)**:
  ```bash
  ./run_all.sh
  ```
  This script acts as the master orchestrator. It runs the Python feedback loop pipeline, compiles the Rust performance core, and runs the Julia stability simulations if installed.

- **Run regression tests**:
  ```bash
  python evaluation/regression_tests.py
  ```

## Working with the Polyglot Architecture

- **Python**: Primary orchestrator (in `pipeline/`, `models/`, `evaluation/`)
- **Rust**: High-performance info-theoretic metrics (in `src_rust/`)
- **Julia**: Formal simulations (in `simulations_julia/`)

When generating code or debugging failing tests, ensure that you respect the boundaries of these languages and use the proper compilers/runners.

## Analytics & Metrics

Logs for training runs are available in `experiments/training.log`.
Visualizations (like `feature_correlation.png`) will be generated during the analysis steps. 
