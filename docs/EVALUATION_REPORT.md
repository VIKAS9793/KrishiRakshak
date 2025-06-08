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
| Accuracy | 91.92% |
| Precision (Weighted) | 89.93% |
| Recall (Weighted) | 92.13% |
| F1-Score | 89.68% |
| IoU | 82.93% |
| mAP | 96.89% |
| MSE | 4.8650 |
| SSIM | 0.0019 |
| PSNR | 0.8152 |

### Class-wise Performance
*(See `outputs/hackathon_demo/experiment_20250607_225122/metrics.json` for detailed class-wise accuracy and other metrics.)*

### Class-wise Accuracy Visualization
*(See `outputs/hackathon_demo/experiment_20250607_225122/class_wise_accuracy.png` for a bar chart showing accuracy per class.)*

## Confusion Matrix
*(See `outputs/hackathon_demo/experiment_20250607_225122/confusion_matrix.png` for the generated confusion matrix.)*

## Training Curves
### Accuracy
*(See `outputs/hackathon_demo/experiment_20250607_225122/training_accuracy.png` for the training accuracy curve.)*

### Loss
*(See `outputs/hackathon_demo/experiment_20250607_225122/training_loss.png` for the training loss curve.)*

## Error Analysis
- **Common Misclassifications**: 
  - Similar looking diseases
  - Early stage diseases
- **Confidence Scores**:
  - Average confidence for correct predictions: (To be extracted from logs/analysis)
  - Average confidence for incorrect predictions: (To be extracted from logs/analysis)
