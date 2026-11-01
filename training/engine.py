"""One epoch; full fine-tuning with unweighted cross entropy."""
import torch
from training.validation import ClassificationMetrics


def train_epoch(model, loader, optimizer, criterion, scaler, device, amp=False):
    model.train()
    metrics = ClassificationMetrics()
    optimizer_steps = amp_skipped_steps = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler decreases its scale exactly when inf/NaN gradients make
        # it skip optimizer.step(). Equal or growing scale means success.
        # AdamW normally returns None even when it updates parameters.
        if scaler.get_scale() < scale_before:
            amp_skipped_steps += 1
        else:
            optimizer_steps += 1
        metrics.update(logits, labels, loss.item())
    return {**metrics.compute(), 'optimizer_steps': optimizer_steps,
            'amp_skipped_steps': amp_skipped_steps}
