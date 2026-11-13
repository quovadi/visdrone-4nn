"""Paired metrics; ASR is conditional on a correct clean prediction."""
import torch
import torch.nn.functional as F
from training.validation import ClassificationMetrics


class PairedMetrics:
    def __init__(self):
        self.clean = ClassificationMetrics()
        self.adversarial = ClassificationMetrics()
        self.clean_correct = 0
        self.successes = 0
        self.clean_correct_per_class = torch.zeros(10, dtype=torch.int64)

    def update(self, labels, clean, adversarial):
        labels, clean, adversarial = [torch.as_tensor(v, dtype=torch.int64).cpu() for v in (labels, clean, adversarial)]
        for values in (labels, clean, adversarial):
            if values.ndim != 1 or torch.any((values < 0) | (values >= 10)):
                raise ValueError('Expected labels/predictions in 0..9')
        if not labels.shape == clean.shape == adversarial.shape:
            raise ValueError('Prediction shapes disagree')
        self.clean.update(F.one_hot(clean, 10).float(), labels, 0.)
        self.adversarial.update(F.one_hot(adversarial, 10).float(), labels, 0.)
        correct = clean == labels
        self.clean_correct += int(correct.sum())
        self.successes += int((correct & (adversarial != labels)).sum())
        self.clean_correct_per_class += torch.bincount(labels[correct], minlength=10)

    def compute(self):
        clean, adv = self.clean.compute(), self.adversarial.compute()
        return {'samples': clean['samples'], 'clean_accuracy': clean['accuracy'],
                'adversarial_accuracy': adv['accuracy'],
                'asr': self.successes / self.clean_correct if self.clean_correct else None,
                'asr_defined': bool(self.clean_correct), 'attack_success_count': self.successes,
                'clean_correct_count': self.clean_correct,
                'clean_correct_per_class': self.clean_correct_per_class.tolist(),
                'clean_macro_f1': clean['macro_f1'], 'adversarial_macro_f1': adv['macro_f1'],
                'macro_f1_drop': clean['macro_f1'] - adv['macro_f1'],
                'clean_per_class_f1': clean['per_class_f1'],
                'adversarial_per_class_f1': adv['per_class_f1'], 'support': clean['support']}
