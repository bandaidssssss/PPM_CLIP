import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score


def validate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            outputs, _ = model(in_tens, label, mode="test")
            probs = outputs[:, 1].detach().cpu().tolist()
            y_pred.extend(probs)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred
