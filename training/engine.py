"""One epoch; full fine-tuning with unweighted cross entropy."""
import torch
from training.validation import ClassificationMetrics


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp=False):
    model.train()
    metrics = ClassificationMetrics()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        metrics.update(logits, labels, loss.item())
    return metrics.compute()
