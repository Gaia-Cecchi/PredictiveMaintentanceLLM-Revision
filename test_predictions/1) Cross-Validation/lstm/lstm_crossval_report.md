# Cross-Validation Report for Long Short-Term Memory Network

Analysis date: 2025-03-21 15:12:34

## Overview

- Total samples analyzed: 2185
- Number of anomalies in dataset: 11
- Percentage of anomalies: 0.50%
- Number of folds: 5

## Main Metrics (mean ± std)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| precision | 0.0199 | 0.0274 | 0.0000 | 0.0541 |
| recall | 0.3000 | 0.4472 | 0.0000 | 1.0000 |
| f1_score | 0.0369 | 0.0507 | 0.0000 | 0.0976 |
| auc_roc | 0.6443 | 0.2368 | 0.4553 | 0.9517 |
| specificity | 0.9117 | 0.0084 | 0.9034 | 0.9215 |

## Interpretation

The metrics show the model's performance across different data splits. The high variation in F1-score (CV=1.373) might indicate some instability in predictions across different data subsets.

## Conclusion

The Long Short-Term Memory Network model shows varying performance in classifying anomalies in the predictive maintenance dataset.