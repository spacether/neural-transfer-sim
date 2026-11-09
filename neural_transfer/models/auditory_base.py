"""Base classes for auditory peripheral encoders and circuit models.

Any organism's auditory model implements these interfaces so the transfer
pipeline can read from one and write to another without knowing which
organism it's talking to.

Peripheral encoder: stimulus → sensory neuron activity.
  - Fly: JO neuron frequency tuning
  - Human: cochlear model (hair cells → auditory nerve fibers)
  - Mouse: similar to human, different frequency range

Circuit model (optional): sensory activity → downstream layers → behavior.
  - Fly: AMMC → WED/VLP → output (from FlyWire connectome)
  - Human: not needed if writing at auditory nerve (cochlear implant site)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class AuditoryEncoder(ABC):
    """Peripheral encoder: maps sound stimuli to sensory neuron activity.

    Subclasses implement the organism-specific transduction:
      - Which neurons exist and what they're called
      - How they respond to frequency content
      - Temporal dynamics (adaptation, filtering)

    The transfer pipeline reads from this layer.
    """

    @property
    @abstractmethod
    def receptor_names(self) -> List[str]:
        """Ordered list of sensory neuron names."""
        ...

    @property
    def n_receptors(self) -> int:
        return len(self.receptor_names)

    @property
    @abstractmethod
    def stimulus_profiles(self) -> Dict[str, Dict]:
        """Named stimuli available for this organism."""
        ...

    @property
    @abstractmethod
    def stimulus_valence(self) -> Dict[str, str]:
        """Behavioral category per stimulus name."""
        ...

    @abstractmethod
    def encode(self, sound_name: str, intensity: float) -> Dict[str, float]:
        """Steady-state encoding: sound → receptor activity dict."""
        ...

    @abstractmethod
    def encode_array(self, sound_name: str, intensity: float) -> np.ndarray:
        """Steady-state encoding: sound → receptor activity array (n_receptors,)."""
        ...

    @abstractmethod
    def encode_timesteps(
        self, stimulus: np.ndarray, dt: Optional[float] = None,
    ) -> np.ndarray:
        """Temporal encoding: (n_steps, n_freq_bands) → (n_steps, n_receptors).

        Includes adaptation and membrane dynamics.
        """
        ...

    @abstractmethod
    def clone_varied(self, seed: int, jitter_scale: float) -> "AuditoryEncoder":
        """Return a varied copy (individual differences in sensitivity)."""
        ...

    def validate(self) -> Dict[str, bool]:
        """Basic validation: receptors respond to stimuli."""
        checks: Dict[str, bool] = {}
        for name in list(self.stimulus_profiles.keys())[:3]:
            rates = self.encode(name, 1.0)
            checks[f"responds_to_{name}"] = max(rates.values()) > 0
        silent = self.encode(list(self.stimulus_profiles.keys())[0], 0.0)
        checks["silent_at_zero"] = max(silent.values()) < 1e-6
        return checks


class AuditoryCircuit(ABC):
    """Circuit model: propagates receptor activity through downstream layers.

    Subclasses implement the organism-specific wiring:
      - Which layers exist (AMMC/WED for fly, cochlear nucleus for vertebrate)
      - Connectivity (from connectome or model)
      - Behavioral readout

    The transfer pipeline writes to this layer.
    """

    @property
    @abstractmethod
    def bottleneck_names(self) -> List[str]:
        """Names of neurons at the bottleneck (best write target)."""
        ...

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """Names of behavioral output channels."""
        ...

    @property
    def n_bottleneck(self) -> int:
        return len(self.bottleneck_names)

    @abstractmethod
    def forward_from_receptors(
        self, receptor_signal: Dict[str, float],
    ) -> Dict:
        """Receptor dict → {behavior: str, neurons: Dict[str, float]}."""
        ...

    @abstractmethod
    def forward_from_bottleneck(
        self, bottleneck_signal: Dict[str, float],
    ) -> Dict:
        """Bottleneck dict → {behavior: str, neurons: Dict[str, float]}."""
        ...

    @abstractmethod
    def natural_response(
        self, sound_name: str, intensity: float,
    ) -> Dict[str, float]:
        """What this circuit produces when the organism hears the sound itself."""
        ...

    @abstractmethod
    def natural_behavior(
        self, sound_name: str, intensity: float,
    ) -> str:
        """Behavioral classification for a naturally heard sound."""
        ...

    @abstractmethod
    def forward_timesteps_from_receptors(
        self, receptor_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Temporal propagation from receptor time series."""
        ...

    @abstractmethod
    def forward_timesteps_from_bottleneck(
        self, bottleneck_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Temporal propagation from bottleneck time series."""
        ...
