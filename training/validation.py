"""Streaming metrics over every example, macro-F1 over all ten labels."""
import torch


class ClassificationMetrics:
    def __init__(self, classes=10):
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64)
        self.loss_sum = 0.0
        self.count = 0

    def update(self, logits, labels, loss):
        labels = labels.detach().to(device='cpu', dtype=torch.int64)
        predictions = logits.detach().argmax(1).cpu()
        n = self.confusion.shape[0]
        self.confusion += torch.bincount(labels * n + predictions, minlength=n*n).reshape(n, n)
        self.loss_sum += float(loss) * labels.numel()
        self.count += labels.numel()

    def compute(self):
        if self.count == 0:
            raise ValueError('No samples evaluated')
        matrix = self.confusion.double()
        tp = matrix.diag()
        denominator = matrix.sum(0) + matrix.sum(1)
        f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0)
        return {'loss': self.loss_sum / self.count,
                'accuracy': float(tp.sum() / self.count),
                'macro_f1': float(f1.mean()), 'per_class_f1': f1.tolist(),
                'support': matrix.sum(1).long().tolist(), 'samples': self.count}


@torch.inference_mode()
def validate(model, loader, criterion, device, amp=False):
    model.eval()
    metrics = ClassificationMetrics()
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        metrics.update(logits, labels, loss.item())
    return metrics.compute()
