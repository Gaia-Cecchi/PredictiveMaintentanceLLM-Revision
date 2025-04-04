# Cross-Validation Report for Qwen 2.5-32B

Analysis date: 2025-03-21 13:04:22

## Overview

- Total samples analyzed: 1133
- Number of anomalies in dataset: 11
- Percentage of anomalies: 0.97%
- Number of folds: 5

## Main Metrics (mean ± std)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| precision | 0.8933 | 0.1535 | 0.6667 | 1.0000 |
| recall | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| f1_score | 0.9378 | 0.0908 | 0.8000 | 1.0000 |
| auc_roc | 0.9991 | 0.0012 | 0.9978 | 1.0000 |
| specificity | 0.9982 | 0.0025 | 0.9955 | 1.0000 |

## Interpretation

The metrics show consistency across folds, indicating that the model generalizes well to unseen data. The F1-score variation (CV=0.097) is normal, suggesting a good balance between stability and adaptation.

## Conclusion

The Qwen 2.5-32B model shows overall good performance in classifying anomalies in the predictive maintenance dataset.