"""Independent PyTorch implementation of Andriushchenko et al. Linf Square.

Algorithm reference: max-andr/square-attack, attack.py at
 ea95eebb5aca62ec790a927b5aa985ba4e87245c. See docs/square_attack.md.
Only an RGB -> logits callable is accepted: no model introspection or autograd.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from attacks.core import check_bounds


@dataclass(frozen=True)
class SquareConfig:
    epsilon: float
    max_queries: int = 5000
    p_init: float = .05
    seed: int = 42
    name: str = 'square'

    def __post_init__(self):
        if self.name != 'square' or not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('Square requires finite RGB epsilon in [0,1]')
        if type(self.max_queries) is not int or self.max_queries < 1:
            raise ValueError('max_queries must be a positive integer, including clean')
        if not math.isfinite(self.p_init) or not 0 < self.p_init <= 1:
            raise ValueError('p_init must be in (0,1]')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Invalid seed')

    def parameters(self):
        return {**asdict(self), 'norm': 'linf', 'targeted': False, 'precision': 'fp32',
                'loss': 'true_logit_minus_max_other', 'query_convention': 'total_including_clean_and_initial',
                'reference_commit': 'ea95eebb5aca62ec790a927b5aa985ba4e87245c',
                'rng': 'numpy.PCG64_per_sample', 'resample_limit': 128}


def probability(p_init, iteration, attack_budget):
    t = int(iteration / attack_budget * 10000)
    if t <= 10:
        return p_init
    for upper, divisor in ((50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32),
                           (4000, 64), (6000, 128), (8000, 256), (10000, 512)):
        if t <= upper:
            return p_init / divisor
    return p_init


def margin(logits, labels):
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return logits.gather(1, labels[:, None]).flatten() - other.amax(1)


def proposal(original, best, epsilon, p, rng):
    c, h, w = original.shape
    side = min(max(round(math.sqrt(p * h * w)), 1), h-1, w-1)
    # Preserve reference's exclusive upper location bound (last anchor omitted).
    top, left = int(rng.integers(h-side)), int(rng.integers(w-side))
    source = original[:, top:top+side, left:left+side]
    old = best[:, top:top+side, left:left+side]
    for _ in range(128):
        signs = torch.tensor(rng.choice([-1., 1.], size=(c, 1, 1)), dtype=torch.float32, device=original.device)
        window = (source + epsilon * signs).clamp(0, 1)
        if not torch.all((window - old).abs() < 1e-7):
            break
    else:
        # Bounded equivalent escape from an unchanged patch; no extra queries.
        high, low = (source + epsilon).clamp(0, 1), (source - epsilon).clamp(0, 1)
        signs = torch.where((high-old).abs().flatten(1).amax(1) < 1e-7, -1., 1.)[:, None, None]
        window = (source + signs * epsilon).clamp(0, 1)
        if torch.equal(window, old):
            return None, side
    candidate = best.clone()
    candidate[:, top:top+side, left:left+side] = window
    return candidate, side


@torch.inference_mode()
def generate_square(scores, images, labels, config, *, sample_seeds=None, diagnostic=None):
    if (images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 2
            or len(images) == 0 or labels.shape != (len(images),) or labels.dtype != torch.int64):
        raise ValueError('Expected nonempty RGB NCHW and int64 [N] labels')
    check_bounds(images, images, 0.)
    n = len(images)
    if sample_seeds is None:
        raise ValueError('Stable per-sample seeds required; use frozen validation identities')
    seeds = sample_seeds
    if len(seeds) != n:
        raise ValueError('One seed per sample required')
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    queries = torch.zeros(n, dtype=torch.int64, device=images.device)
    accepted = [0]*n
    reasons = ['budget']*n
    first_success = [None]*n
    clean_correct = None

    def query(indices, batch, stage):
        queries[indices] += 1
        if diagnostic:
            diagnostic({'kind': 'square_query', 'positions': indices.tolist(),
                        'queries': queries[indices].tolist(), 'stage': stage})
        logits = scores(batch)
        if (logits.ndim != 2 or len(logits) != len(indices) or logits.shape[1] < 2
                or logits.dtype != torch.float32 or not torch.isfinite(logits).all()):
            raise ValueError('Score oracle must return finite FP32 [N,classes] logits')
        if torch.any(labels[indices] < 0) or torch.any(labels[indices] >= logits.shape[1]):
            raise ValueError('Labels outside oracle classes')
        predictions = logits.argmax(1)
        if clean_correct is not None:
            misclassified = indices[clean_correct[indices] & (predictions != labels[indices])]
            for i in misclassified.tolist():
                if first_success[i] is None:
                    first_success[i] = int(queries[i])
                    if diagnostic:
                        diagnostic({'kind': 'square_first_success', 'positions': [i],
                                    'first_success_query': first_success[i]})
        return predictions, margin(logits, labels[indices])

    ids = torch.arange(n, device=images.device)
    clean, best_margin = query(ids, images, 'clean')
    clean_correct = clean == labels
    prediction = clean.clone()
    best = images.detach().clone()
    active = clean == labels
    for i in range(n):
        if not active[i]:
            reasons[i] = 'clean_incorrect'
        elif config.epsilon == 0:
            reasons[i] = 'epsilon_zero'
    active &= config.epsilon > 0
    active &= queries < config.max_queries
    ids = active.nonzero().flatten()
    if len(ids):
        initial = torch.stack([(images[i] + torch.tensor(
            rngs[i].choice([-config.epsilon, config.epsilon], size=(3, 1, images.shape[3])),
            dtype=torch.float32, device=images.device)).clamp(0, 1) for i in ids.tolist()])
        check_bounds(images[ids], initial, config.epsilon)
        pred, values = query(ids, initial, 'initial_stripes')
        best[ids], prediction[ids], best_margin[ids] = initial, pred, values
    while True:
        active = (clean == labels) & (prediction == labels) & (queries < config.max_queries) & (config.epsilon > 0)
        active &= torch.tensor([reason != 'no_effective_proposal' for reason in reasons], device=images.device)
        ids = active.nonzero().flatten()
        if not len(ids):
            break
        selected, candidates, sides = [], [], []
        for i in ids.tolist():
            p = probability(config.p_init, int(queries[i])-2, config.max_queries-1)
            candidate, side = proposal(images[i], best[i], config.epsilon, p, rngs[i])
            if candidate is None:
                reasons[i] = 'no_effective_proposal'
            else:
                selected.append(i)
                candidates.append(candidate)
                sides.append(side)
        if not selected:
            continue
        ids = torch.tensor(selected, device=images.device)
        candidates = torch.stack(candidates)
        check_bounds(images[ids], candidates, config.epsilon)
        pred, values = query(ids, candidates, 'square')
        # Strict margin minimization. Exact argmax success wins tie cases.
        improve = (values < best_margin[ids]) | (pred != labels[ids])
        chosen = ids[improve]
        best[chosen], prediction[chosen], best_margin[chosen] = candidates[improve], pred[improve], values[improve]
        for j, i in enumerate(selected):
            accepted[i] += int(improve[j])
        if diagnostic:
            diagnostic({'kind': 'square_update', 'positions': selected, 'side': sides,
                        'accepted': improve.tolist(), 'candidate_margin': values.tolist()})
    successes = (clean == labels) & (prediction != labels)
    for i in range(n):
        if successes[i]:
            reasons[i] = 'success'
    distances = check_bounds(images, best, config.epsilon)
    return {'images': best, 'clean_prediction': clean, 'prediction': prediction,
            'queries': queries, 'linf': distances, 'success': successes,
            'first_success_query': first_success, 'margin': best_margin, 'accepted_updates': accepted, 'stop_reason': reasons}
