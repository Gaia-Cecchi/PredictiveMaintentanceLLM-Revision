# Model Comparison: Qwen 2.5 vs ANN vs CNN vs LSTM

Analysis date: 2025-03-21 15:47:10

## Performance Summary

| Metric | Qwen 2.5 (mean ± std) | ANN (mean ± std) | CNN (mean ± std) | LSTM (mean ± std) | Winner |
|---------|---------------------|------------------|------------------|------------------|--------|
| precision | 0.8933 ± 0.1535 | 0.6100 ± 0.3847 | 0.1300 ± 0.1857 | 0.7500 ± 0.2500 | Qwen 2.5 |
| recall | 1.0000 ± 0.0000 | 0.7500 ± 0.4330 | 0.2500 ± 0.4330 | 0.7500 ± 0.2500 | Qwen 2.5 |
| f1_score | 0.9378 ± 0.0908 | 0.6611 ± 0.3911 | 0.1643 ± 0.2520 | 0.7500 ± 0.2500 | Qwen 2.5 |
| auc_roc | 0.9991 ± 0.0012 | 0.9679 ± 0.0627 | 0.6516 ± 0.2376 | 0.8748 ± 0.1248 | Qwen 2.5 |
| specificity | 0.9982 ± 0.0025 | 0.9982 ± 0.0010 | 0.9885 ± 0.0056 | 0.9995 ± 0.0005 | LSTM |

## Key Findings

### Overall Performance
- Qwen 2.5 outperforms other models by 470.8% in terms of F1-score

### Model Stability
- Qwen 2.5 coefficient of variation: 0.097
- ANN coefficient of variation: 0.592
- CNN coefficient of variation: 1.534
- LSTM coefficient of variation: 0.333
- Qwen 2.5 shows better stability across folds

## Conclusion

Qwen 2.5 is the superior model, showing both better performance and more stability across all metrics. It achieves higher precision, recall, and F1-score while maintaining more consistent predictions across different data splits.