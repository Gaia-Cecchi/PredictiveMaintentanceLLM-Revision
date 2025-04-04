# Cross-Validation Report for Convolutional Neural Network

Analysis date: 2025-03-21 15:09:45

## Overview

- Total samples analyzed: 2185
- Number of anomalies in dataset: 11
- Percentage of anomalies: 0.50%
- Number of folds: 5

## Main Metrics (mean ± std)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| precision | 0.1300 | 0.1857 | 0.0000 | 0.4000 |
| recall | 0.2500 | 0.4330 | 0.0000 | 1.0000 |
| f1_score | 0.1643 | 0.2520 | 0.0000 | 0.5714 |
| auc_roc | 0.6516 | 0.2376 | 0.4942 | 0.9966 |
| specificity | 0.9885 | 0.0056 | 0.9794 | 0.9931 |

## Interpretation

The metrics show the model's performance across different data splits. The high variation in F1-score (CV=1.534) might indicate some instability in predictions across different data subsets.

## Conclusion

The Convolutional Neural Network model shows varying performance in classifying anomalies in the predictive maintenance dataset.