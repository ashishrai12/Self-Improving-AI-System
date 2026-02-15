/// Implementation of Shannon Entropy in Rust for high-performance uncertainty estimation.
/// Formal Definition: H(Y|X) = - \sum p(y|backx) \log_2 p(y|x)
pub fn calculate_shannon_entropy(probabilities: &[f64], num_classes: usize) -> Vec<f64> {
    probabilities
        .chunks(num_classes)
        .map(|chunk| {
            chunk
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.log2())
                .sum()
        })
        .collect()
}

/// Calculate the margin uncertainty: the difference between the top two probabilities.
/// This is a common heuristic in active learning to identify samples near the decision boundary.
pub fn calculate_margin_uncertainty(probabilities: &[f64], num_classes: usize) -> Vec<f64> {
    probabilities
        .chunks(num_classes)
        .map(|chunk| {
            let mut sorted = chunk.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            if sorted.len() >= 2 {
                sorted[0] - sorted[1]
            } else {
                sorted[0]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy() {
        let probs = vec![0.5, 0.5, 0.98, 0.02];
        let entropy = calculate_shannon_entropy(&probs, 2);
        assert!((entropy[0] - 1.0).abs() < 1e-9);
        assert!(entropy[1] < 0.2);
    }
}
