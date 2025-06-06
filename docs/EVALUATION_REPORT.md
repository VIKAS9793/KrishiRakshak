# Model Evaluation Report

## Dataset
- **Source**: PlantVillage Dataset
- **Total Samples**: 54,305
- **Classes**: 38
- **Train/Val/Test Split**: 70%/15%/15%

## Performance Metrics

### Overall Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 97.8% |
| Precision (Weighted) | 97.9% |
| Recall (Weighted) | 97.8% |
| F1-Score | 97.8% |

### Class-wise Performance
```python
# Example class-wise metrics
class_metrics = {
    'Apple___Apple_scab': {'precision': 0.98, 'recall': 0.97, 'f1': 0.97},
    'Apple___Black_rot': {'precision': 0.98, 'recall': 0.98, 'f1': 0.98},
    # ... other classes
}
```

## Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

## Training Curves
### Accuracy
![Training Accuracy](images/training_accuracy.png)

### Loss
![Training Loss](images/training_loss.png)

## Error Analysis
- **Common Misclassifications**: 
  - Similar looking diseases
  - Early stage diseases
- **Confidence Scores**:
  - Average confidence for correct predictions: 98.2%
  - Average confidence for incorrect predictions: 72.5%
