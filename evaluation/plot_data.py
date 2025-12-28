#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_data_distribution():
    # Load raw data
    df_raw = pd.read_csv('data/raw/data.csv')
    
    # Load feedback data
    df_feedback = pd.read_csv('data/feedback/feedback.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw data class distribution
    df_raw['label'].value_counts().plot(kind='bar', ax=axes[0,0], title='Raw Data Class Distribution')
    
    # Feedback data class distribution
    if not df_feedback.empty:
        df_feedback['label'].value_counts().plot(kind='bar', ax=axes[0,1], title='Feedback Data Class Distribution')
    else:
        axes[0,1].text(0.5, 0.5, 'No Feedback Data Yet', ha='center', va='center')
    
    # Feature correlation heatmap (first 10 features)
    corr = df_raw.iloc[:, :10].corr()
    sns.heatmap(corr, ax=axes[1,0], cmap='coolwarm', annot=False)
    axes[1,0].set_title('Feature Correlation (Raw Data)')
    
    # Accuracy over time (placeholder for now)
    # This would be from the log
    axes[1,1].text(0.5, 0.5, 'Performance Plot\nAvailable via\nplot_performance.py', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('experiments/data_visualization.png')
    plt.show()

if __name__ == "__main__":
    plot_data_distribution()
