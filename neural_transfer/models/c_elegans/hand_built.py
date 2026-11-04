"""Hand-built worm connectivity model for Stage 1.

This model is a minimal, connectome-inspired graded network for the worm
olfactory pathway, with receptor, interneuron, and output neurons built
explicitly.  It is a fast 9-connection prototype; the Cook2019WormModel in
cook2019_connectome.py uses the real 302-neuron connectome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass
class HandBuiltWormModel:
    """A minimal connectome-style worm model for the attractive/repellent pathway."""

    receptor_gain: Dict[str, float] = field(
        default_factory=lambda: {
            "AWA": 1.0,
            "AWC_L": 0.95,
            "AWC_R": 0.9,
            "AWB": 0.75,
            "ADL": 0.65,
        }
    )
    weight_matrix: Dict[str, float] = field(
        default_factory=lambda: {
            "AWA->AIY": 0.55,
            "AWC_L->AIY": 0.45,
            "AWC_R->AIY": 0.35,
            "AWB->AIB": 0.5,
            "ADL->AIB": 0.4,
            "AIY->AVA": 0.8,
            "AIB->AVB": 0.7,
            "AIA->AVA": 0.25,
            "AIZ->AVB": 0.2,
        }
    )
    command_threshold: float = 1.1
    reverse_threshold: float = 0.8

    def receptor_activity(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        """Return receptor-level activation for the chosen odor type."""
        if odor_intensity < 0:
            raise ValueError("odor_intensity must be non-negative")

        if odor_type == "attractive":
            return {
                "AWA": self.receptor_gain["AWA"] * odor_intensity,
                "AWC_L": self.receptor_gain["AWC_L"] * odor_intensity,
                "AWC_R": self.receptor_gain["AWC_R"] * odor_intensity,
            }

        if odor_type == "repellent":
            return {
                "AWB": self.receptor_gain["AWB"] * odor_intensity,
                "ADL": self.receptor_gain["ADL"] * odor_intensity,
            }

        raise ValueError(f"Unsupported odor_type: {odor_type!r}")

    def network_state(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        """Compute the explicit graded network state for the worm odor pathway."""
        receptor_signal = self.receptor_activity(odor_intensity, odor_type)
        return self._propagate(receptor_signal, odor_type)

    def behavior_from_network(self, state: Dict[str, float], odor_type: str | None = None) -> str:
        """Translate the explicit network state into the worm's coarse movement output."""
        if odor_type is None:
            if any(key in state for key in ("AWA", "AWC_L", "AWC_R")) and not any(key in state for key in ("AWB", "ADL", "AIB")):
                odor_type = "attractive"
            elif any(key in state for key in ("AWB", "ADL", "AIB")) and not any(key in state for key in ("AWA", "AWC_L", "AWC_R")):
                odor_type = "repellent"
            else:
                odor_type = "attractive"

        if odor_type == "attractive":
            forward_drive = state.get("AVA", 0.0) + state.get("AVB", 0.0)
            return "forward" if forward_drive >= self.command_threshold else "pause"

        if odor_type == "repellent":
            reverse_drive = state.get("AVA", 0.0)
            return "reverse" if reverse_drive >= self.reverse_threshold else "pause"

        raise ValueError(f"Unsupported odor_type: {odor_type!r}")

    def odor_response(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, float]:
        """Alias for receptor_activity — interface compatibility with WormModel."""
        return self.receptor_activity(odor_intensity, odor_type)

    def _propagate(self, receptor_signal: Dict[str, float], odor_type: str) -> Dict[str, float]:
        """Propagate a receptor-layer signal through interneurons to command neurons."""
        if odor_type == "attractive":
            aiy = (
                receptor_signal.get("AWA", 0.0) * self.weight_matrix.get("AWA->AIY", 0.0)
                + receptor_signal.get("AWC_L", 0.0) * self.weight_matrix.get("AWC_L->AIY", 0.0)
                + receptor_signal.get("AWC_R", 0.0) * self.weight_matrix.get("AWC_R->AIY", 0.0)
            )
            ava = aiy * self.weight_matrix.get("AIY->AVA", 0.0)
            avb = aiy * self.weight_matrix.get("AIB->AVB", 0.0) * 0.6
            return {**receptor_signal, "AIY": aiy, "AVA": ava, "AVB": avb}

        if odor_type == "repellent":
            aib = (
                receptor_signal.get("AWB", 0.0) * self.weight_matrix.get("AWB->AIB", 0.0)
                + receptor_signal.get("ADL", 0.0) * self.weight_matrix.get("ADL->AIB", 0.0)
            )
            reverse_drive = aib * 1.5
            return {**receptor_signal, "AIB": aib, "AVA": reverse_drive}

        raise ValueError(f"Unsupported odor_type: {odor_type!r}")

    def simulate_from_receptors(
        self, receptor_signal: Dict[str, float], odor_type: str = "attractive"
    ) -> Dict[str, object]:
        """Propagate an injected receptor signal through the receiver's own network."""
        state = self._propagate(receptor_signal, odor_type)
        behavior = self.behavior_from_network(state, odor_type)
        return {
            "model_type": "hand_built_connectome",
            "source": "cook_connectome",
            "odor_type": odor_type,
            "behavior": behavior,
            "neurons": state,
            "injection": "receptor_layer",
        }

    def behavior_from_signal(self, signal: Dict[str, float], odor_type: str = "attractive") -> str:
        """Propagate an injected signal and return behavior — interface compatibility."""
        result = self.simulate_from_receptors(signal, odor_type)
        return result["behavior"]

    def simulate(self, odor_intensity: float, odor_type: str = "attractive") -> Dict[str, object]:
        """Run a single, graded time-step simulation using the explicit network state."""
        state = self.network_state(odor_intensity, odor_type)
        behavior = self.behavior_from_network(state, odor_type)
        return {
            "model_type": "hand_built_connectome",
            "source": "cook_connectome",
            "odor_type": odor_type,
            "odor_intensity": odor_intensity,
            "behavior": behavior,
            "neurons": state,
        }

    def validate_odor_responses(self, odor_intensity: float = 1.0) -> Dict[str, object]:
        """Check that attractive and repellent odors produce the expected directional movement."""
        attractive = self.simulate(odor_intensity, "attractive")
        repellent = self.simulate(odor_intensity, "repellent")
        attractive_ok = attractive["behavior"] == "forward"
        repellent_ok = repellent["behavior"] == "reverse"
        return {
            "model_type": "hand_built_connectome",
            "source": "cook_connectome",
            "odor_intensity": odor_intensity,
            "attractive": attractive,
            "repellent": repellent,
            "passes": attractive_ok and repellent_ok,
            "checks": {
                "attractive_drives_forward": attractive_ok,
                "repellent_drives_reverse": repellent_ok,
            },
        }
