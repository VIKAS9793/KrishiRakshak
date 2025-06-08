# Visualization Guide

This document provides code snippets for generating visualizations used in the reports.

## 1. Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes,
                yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/confusion_matrix.png', dpi=300, bbox_inches='tight')
```

## 2. Training Curves

```python
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/training_curves.png', dpi=300)
```

## 3. Class Distribution

```python
def plot_class_distribution(labels, class_names, title='Class Distribution'):
    plt.figure(figsize=(12, 6))
    sns.countplot(y=labels, order=class_names)
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/class_distribution.png', dpi=300)
```

## 4. Performance Metrics

```python
def plot_metrics(metrics_dict, title='Model Metrics'):
    metrics_df = pd.DataFrame(metrics_dict)
    plt.figure(figsize=(10, 6))
    metrics_df.plot(kind='bar')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/performance_metrics.png', dpi=300)
```

## 5. Memory Usage

```python
def plot_memory_usage(timestamps, cpu_usage, memory_usage):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('CPU Usage (%)', color=color)
    ax1.plot(timestamps, cpu_usage, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Memory Usage (MB)', color=color)
    ax2.plot(timestamps, memory_usage, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Resource Usage')
    fig.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/memory_usage.png', dpi=300)
```

## 6. Class-wise Accuracy

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_class_accuracy(class_accuracies, class_names, title='Class-wise Accuracy'):
    plt.figure(figsize=(14, 10))
    # Sort by accuracy for better readability
    sorted_indices = np.argsort(class_accuracies)[::-1]
    sorted_accuracies = [class_accuracies[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    sns.barplot(x=sorted_accuracies, y=sorted_names, palette='viridis')
    plt.title(title)
    plt.xlabel('Accuracy')
    plt.ylabel('Class')
    plt.xlim(0, 1) # Accuracy is between 0 and 1
    plt.tight_layout()
    plt.savefig('outputs/hackathon_demo/experiment_20250607_225122/class_wise_accuracy.png', dpi=300)
```

## Usage Example

```python
# Example usage with training history
history = {
    'accuracy': [...],
    'val_accuracy': [...],
    'loss': [...],
    'val_loss': [...]
}
plot_training_history(history)
```
