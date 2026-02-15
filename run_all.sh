#!/bin/bash
# Orchestration script for the multi-language Self-Improving AI System

echo "--- 1. Running Python Pipeline (Inference & Feedback) ---"
python pipeline/feedback_loop.py

echo -e "\n--- 2. Building Rust Performance Core ---"
cd src_rust
cargo build --release
cd ..

echo -e "\n--- 3. Running Julia Stability Simulation ---"
# Check if julia is installed, if so run the analysis
if command -v julia &> /dev/null
then
    julia simulations_julia/stability_analysis.jl
else
    echo "Julia not found, skipping theoretical simulation."
fi

echo -e "\n--- Pipeline execution complete ---"
