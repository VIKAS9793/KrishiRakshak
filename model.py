import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import torchmetrics
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list = None
) -> Dict[str, float]:
    """Validate model on validation set with corrected accuracy calculation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    # Metrics
    precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=model.num_classes).to(device)
    recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=model.num_classes).to(device)
    f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=model.num_classes).to(device)

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_preds.append(predicted)
            all_targets.append(targets)

            precision(predicted, targets)
            recall(predicted, targets)
            f1(predicted, targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cm = confusion_matrix(all_targets.cpu().numpy(), all_preds.cpu().numpy())
    class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        'val_loss': running_loss / len(val_loader.dataset),
        'val_accuracy': 100. * correct / total,
        'val_precision': precision.compute().item(),
        'val_recall': recall.compute().item(),
        'val_f1': f1.compute().item(),
        'class_accuracy': class_acc.tolist(),
        'confusion_matrix': cm.tolist()
    }

    if class_names is not None:
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_metrics[class_name] = {
                'accuracy': class_acc[i],
                'samples': int(cm[i].sum())
            }
        metrics['per_class'] = class_metrics

    return metrics 