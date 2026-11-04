"""Baseline transfer model: receptor-to-receptor copy with no translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass
class ReceptorToReceptorBaseline:
    """A no-translation link that copies the sender's receptor activity to the receiver."""

    sender: object | None = None
    receiver: object | None = None
    source_neurons: tuple[str, ...] = ("AWA", "AWC_L", "AWC_R")
    target_neurons: tuple[str, ...] | None = None
    signal_gain: float = 1.0

    def copy_signal(self, sender_activity: Mapping[str, float]) -> Dict[str, float]:
        """Copy the sender's activity to the target neuron set without any learned transform."""
        target_keys = self.target_neurons if self.target_neurons is not None else tuple(sender_activity.keys())
        return {
            neuron: float(sender_activity.get(neuron, 0.0)) * self.signal_gain
            for neuron in target_keys
        }

    def evaluate_transfer(self, odor_intensity: float, odor_type: str = "attractive") -> dict:
        """Run one receptor-to-receptor transfer trial using the sender and receiver models."""
        if self.sender is None or self.receiver is None:
            raise ValueError("ReceptorToReceptorBaseline requires both sender and receiver models.")

        sender_activity = self.sender.odor_response(odor_intensity, odor_type)
        sender_behavior = self.sender.behavior_from_signal(sender_activity, odor_type)
        receiver_activity = self.copy_signal(sender_activity)
        receiver_behavior = self.receiver.behavior_from_signal(receiver_activity, odor_type)

        return {
            "odor_intensity": odor_intensity,
            "odor_type": odor_type,
            "sender_activity": sender_activity,
            "sender_behavior": sender_behavior,
            "receiver_activity": receiver_activity,
            "receiver_behavior": receiver_behavior,
            "success": receiver_behavior == sender_behavior,
        }
