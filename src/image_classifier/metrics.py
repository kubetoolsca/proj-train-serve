from dataclasses import dataclass


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
