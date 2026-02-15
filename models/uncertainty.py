import numpy as np

def calculate_shannon_entropy(probabilities):
    """
    Calculate the Shannon Entropy of the model's predictions.
    
    Formal Definition:
    H(Y|X) = - \sum_{y \in \mathcal{Y}} p(y|x) \log_2 p(y|x)
    
    In the context of binary classification:
    H(Y|X) = - [p \log_2(p) + (1-p) \log_2(1-p)]
    
    High entropy indicates high epistemic/aleatoric uncertainty, 
    making these samples prime candidates for the feedback loop.
    
    Args:
        probabilities (np.ndarray): N x K array of predicted probabilities.
        
    Returns:
        np.ndarray: N-dimensional array of entropy values.
    """
    # Clip probabilities to avoid log(0)
    probs = np.clip(probabilities, 1e-15, 1.0)
    entropy = -np.sum(probs * np.log2(probs), axis=1)
    return entropy

def calculate_margin_uncertainty(probabilities):
    """
    Calculate sampling margin: the difference between the top two probabilities.
    Smaller margin indicates higher uncertainty.
    """
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return margin
