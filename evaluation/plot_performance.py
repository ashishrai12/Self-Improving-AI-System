#!/usr/bin/env python3
import matplotlib.pyplot as plt
import re

def plot_performance():
    iterations = []
    accuracies = []
    
    with open('experiments/training.log', 'r') as f:
        for line in f:
            match = re.search(r'Iteration (\d+): Accuracy ([\d.]+)', line)
            if match:
                iterations.append(int(match.group(1)))
                accuracies.append(float(match.group(2)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Self-Improving AI System: Accuracy Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0.8, 1.0)
    plt.savefig('experiments/performance_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_performance()
