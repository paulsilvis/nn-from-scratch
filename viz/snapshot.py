"""Checkpoint/snapshot recording utilities shared across stages.

Convention: any component that evolves during training (a
perceptron's weights, a network's parameters, a loss trajectory) can
push snapshots to a Recorder as it trains. Downstream code in
replay.py consumes the recorded snapshots to build visualizations,
so training code never needs to know about matplotlib.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Snapshot:
    step: int
    data: Dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Collects snapshots during training for later replay/plotting."""

    def __init__(self):
        self._snapshots: List[Snapshot] = []

    def record(self, step, **data):
        self._snapshots.append(Snapshot(step=step, data=data))

    @property
    def snapshots(self):
        return list(self._snapshots)

    def __len__(self):
        return len(self._snapshots)

    def get(self, key):
        """Return the list of values for `key` across all snapshots."""
        return [s.data[key] for s in self._snapshots if key in s.data]
