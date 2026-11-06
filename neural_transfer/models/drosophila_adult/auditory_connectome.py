"""Connectome-derived adult Drosophila auditory model.

Split into two pluggable pieces:
  DrosophilaAuditoryEncoder — JO peripheral encoding (stimulus → 1103 JO rates)
  DrosophilaAuditoryCircuit — AMMC/WED/VLP circuit (JO rates → behavior)

AuditoryAdultModel composes both and keeps the original API.

A human or mouse model replaces the encoder (cochlear model) and optionally
the circuit (not needed if writing at the auditory nerve).

Weight matrices from FlyWire v783 (Dorkenwald et al. 2024; Schlegel et al. 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from neural_transfer.models.auditory_base import AuditoryEncoder, AuditoryCircuit


DATA_PATH = Path(__file__).resolve().parent / "auditory_connectome.npz"

FREQ_BANDS = [100, 150, 200, 250, 300, 400, 500, 700]
N_FREQ_BANDS = len(FREQ_BANDS)

# JO subtype tuning: (center_hz, bandwidth_hz, is_sound_sensitive)
# JO-A: broad vibration-sensitive, 100–900+ Hz, peak ~350 Hz
#   Kamikouchi et al. 2009 Nature; Patella & Wilson 2018 Curr Biol
# JO-B: low-frequency vibration, peak <100 Hz, responds to song at low end
#   Yorozu et al. 2009 Nature
# JO-C/E: deflection/gravity-sensitive, not frequency-tuned
#   Kamikouchi et al. 2009
# JO-D: mid-range vibration 100–200 Hz
#   Matsuo et al. 2014, cited in Ishikawa et al. 2017
JO_TUNING = {
    "A": (350, 300, True),
    "B": (100, 80, True),
    "C": (100, 50, False),
    "D": (150, 80, False),
    "E": (100, 60, False),
    "F": (100, 50, False),
    "mz": (200, 150, True),
    "unclear": (200, 150, False),
}

# Sound spectral profiles — power at each frequency band.
#
# Courtship song spectra from Arthur et al. 2013 BMC Biology:
#   pulse carrier 220–450 Hz, peak ~300 Hz (broad)
#   sine carrier 130–185 Hz, median 158 Hz (narrow)
# Agonistic sound from Jonsson et al. 2011: ~300 Hz carrier
# Flight tone measured at ~180 Hz fundamental
# Antenna mechanical resonance ~400 Hz (Göpfert & Robert 2001, 2002)
# Pure tones: standard stimuli used in Patella & Wilson 2018
SOUND_PROFILES: Dict[str, Dict[int, float]] = {
    # ── Courtship signals ────────────────────────────────────────────
    "pulse_song":       {100: 0.05, 150: 0.10, 200: 0.40, 250: 0.80, 300: 1.00, 400: 0.60, 500: 0.20, 700: 0.02},
    "sine_song":        {100: 0.15, 150: 1.00, 200: 0.40, 250: 0.05, 300: 0.02, 400: 0.00, 500: 0.00, 700: 0.00},
    "combined_song":    {100: 0.10, 150: 0.60, 200: 0.50, 250: 0.60, 300: 0.80, 400: 0.40, 500: 0.15, 700: 0.01},
    # ── Agonistic / threat signals ───────────────────────────────────
    "wing_threat":      {100: 0.10, 150: 0.15, 200: 0.30, 250: 0.60, 300: 1.00, 400: 0.50, 500: 0.15, 700: 0.02},
    "male_agonistic":   {100: 0.05, 150: 0.10, 200: 0.25, 250: 0.50, 300: 1.00, 400: 0.35, 500: 0.10, 700: 0.01},
    # ── Ecological sounds ────────────────────────────────────────────
    "flight_tone":      {100: 0.05, 150: 0.50, 200: 1.00, 250: 0.30, 300: 0.10, 400: 0.02, 500: 0.00, 700: 0.00},
    "predator_buzz":    {100: 0.02, 150: 0.05, 200: 0.15, 250: 0.30, 300: 0.60, 400: 0.90, 500: 1.00, 700: 0.50},
    "wind_gust":        {100: 0.80, 150: 0.50, 200: 0.20, 250: 0.05, 300: 0.02, 400: 0.00, 500: 0.00, 700: 0.00},
    # ── Antenna resonance (~400 Hz) ──────────────────────────────────
    "antennal_res":     {100: 0.00, 150: 0.02, 200: 0.05, 250: 0.15, 300: 0.40, 400: 1.00, 500: 0.50, 700: 0.10},
    # ── Pure tones (electrophysiology stimuli) ───────────────────────
    "tone_100":         {100: 1.00, 150: 0.30, 200: 0.05, 250: 0.00, 300: 0.00, 400: 0.00, 500: 0.00, 700: 0.00},
    "tone_150":         {100: 0.30, 150: 1.00, 200: 0.30, 250: 0.05, 300: 0.00, 400: 0.00, 500: 0.00, 700: 0.00},
    "tone_200":         {100: 0.05, 150: 0.30, 200: 1.00, 250: 0.30, 300: 0.05, 400: 0.00, 500: 0.00, 700: 0.00},
    "tone_300":         {100: 0.00, 150: 0.05, 200: 0.15, 250: 0.50, 300: 1.00, 400: 0.30, 500: 0.05, 700: 0.00},
    "tone_400":         {100: 0.00, 150: 0.00, 200: 0.05, 250: 0.15, 300: 0.30, 400: 1.00, 500: 0.30, 700: 0.05},
    "tone_500":         {100: 0.00, 150: 0.00, 200: 0.00, 250: 0.05, 300: 0.10, 400: 0.30, 500: 1.00, 700: 0.30},
    "tone_700":         {100: 0.00, 150: 0.00, 200: 0.00, 250: 0.00, 300: 0.02, 400: 0.10, 500: 0.30, 700: 1.00},
    # ── Band-pass stimuli ────────────────────────────────────────────
    "low_band":         {100: 0.80, 150: 1.00, 200: 0.80, 250: 0.20, 300: 0.00, 400: 0.00, 500: 0.00, 700: 0.00},
    "high_band":        {100: 0.00, 150: 0.00, 200: 0.00, 250: 0.20, 300: 0.50, 400: 0.80, 500: 1.00, 700: 0.80},
    # ── Controls ─────────────────────────────────────────────────────
    "silence":          {f: 0.0 for f in FREQ_BANDS},
    "white_noise":      {f: 0.5 for f in FREQ_BANDS},
}

SOUND_VALENCE: Dict[str, str] = {
    "pulse_song": "courtship", "sine_song": "courtship",
    "combined_song": "courtship",
    "wing_threat": "startle", "male_agonistic": "startle",
    "predator_buzz": "startle",
    "flight_tone": "neutral", "wind_gust": "neutral",
    "antennal_res": "neutral",
    "tone_100": "neutral", "tone_150": "neutral", "tone_200": "neutral",
    "tone_300": "neutral", "tone_400": "neutral", "tone_500": "neutral",
    "tone_700": "neutral",
    "low_band": "neutral", "high_band": "neutral",
    "silence": "neutral", "white_noise": "neutral",
}

COURTSHIP_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "courtship"]
STARTLE_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "startle"]
NEUTRAL_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "neutral"]


def _pick_sound(sound_type: str, rng: np.random.Generator | None = None) -> str:
    if rng is None:
        rng = np.random.default_rng(0)
    if sound_type == "courtship":
        return rng.choice(COURTSHIP_SOUNDS)
    elif sound_type == "startle":
        return rng.choice(STARTLE_SOUNDS)
    elif sound_type == "neutral":
        return rng.choice(NEUTRAL_SOUNDS)
    return sound_type


# ── Temporal stimulus generators ─────────────────────────────────────

def _profile_to_vec(profile: Dict[int, float]) -> np.ndarray:
    return np.array([profile.get(f, 0.0) for f in FREQ_BANDS])


def make_pulse_train(
    duration_ms: float, dt: float = 0.5, ipi_ms: float = 35.0,
    pulse_width_ms: float = 10.0, intensity: float = 1.0,
    profile: str = "pulse_song",
) -> np.ndarray:
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(profile, SOUND_PROFILES["pulse_song"]))
    stim = np.zeros((n_steps, N_FREQ_BANDS))
    pulse_samples = int(pulse_width_ms / dt)
    period_samples = int(ipi_ms / dt)
    t = 0
    while t < n_steps:
        end = min(t + pulse_samples, n_steps)
        stim[t:end] = spec * intensity
        t += period_samples
    return stim


def make_sine_song(
    duration_ms: float, dt: float = 0.5, mod_freq_hz: float = 5.0,
    intensity: float = 1.0, profile: str = "sine_song",
) -> np.ndarray:
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(profile, SOUND_PROFILES["sine_song"]))
    t_sec = np.arange(n_steps) * (dt / 1000.0)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq_hz * t_sec)
    return (envelope[:, None] * spec[None, :]) * intensity


def make_constant(
    duration_ms: float, dt: float = 0.5, sound_name: str = "pulse_song",
    intensity: float = 1.0,
) -> np.ndarray:
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(sound_name, {}))
    return np.full((n_steps, N_FREQ_BANDS), spec * intensity)


def make_stimulus_sequence(segments: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(segments, axis=0)


STIM_DURATION_MS = 500.0
STIM_DT = 0.5
STIM_AVG_WINDOW_MS = 100.0

_TEMPORAL_GENERATOR: Dict[str, str] = {
    "pulse_song": "pulse", "combined_song": "pulse", "sine_song": "sine",
    "male_agonistic": "pulse", "wing_threat": "pulse",
}


def make_temporal_stimulus(
    sound_name: str, intensity: float = 1.0,
    duration_ms: float = STIM_DURATION_MS, dt: float = STIM_DT,
) -> np.ndarray:
    gen = _TEMPORAL_GENERATOR.get(sound_name, "constant")
    if gen == "pulse":
        return make_pulse_train(duration_ms, dt, intensity=intensity, profile=sound_name)
    elif gen == "sine":
        return make_sine_song(duration_ms, dt, intensity=intensity, profile=sound_name)
    return make_constant(duration_ms, dt, sound_name=sound_name, intensity=intensity)


def temporal_avg_window(dt: float = STIM_DT) -> int:
    return max(1, int(STIM_AVG_WINDOW_MS / dt))


# ── Connectome loading ───────────────────────────────────────────────

SOUND_SUBTYPES = ("A", "B")

MIN_WV_SYN = 200

_CACHE: Dict[tuple, dict] = {}


def _load_connectome(subtypes: Optional[tuple] = None) -> dict:
    data = np.load(DATA_PATH, allow_pickle=True)
    jo_major_groups = list(data["jo_major_groups"])
    jo_cell_types = list(data["jo_cell_types"])

    w_jo_ammc = data["w_jo_ammc"]
    w_ammc_wedvlp_full = data["w_ammc_wedvlp"]
    w_ammc_ammc = data["w_ammc_ammc"]
    ammc_nt_signs = data["ammc_nt_signs"]

    if subtypes is not None:
        allowed = set(subtypes)
        jo_mask = np.array([g in allowed for g in jo_major_groups])
        jo_major_groups = [g for g, m in zip(jo_major_groups, jo_mask) if m]
        jo_cell_types = [ct for ct, m in zip(jo_cell_types, jo_mask) if m]
        w_jo_ammc = w_jo_ammc[:, jo_mask]

    wv_input = w_ammc_wedvlp_full.sum(axis=1)
    wv_mask = wv_input >= MIN_WV_SYN
    w_ammc_wedvlp = w_ammc_wedvlp_full[wv_mask]

    return {
        "jo_major_groups": jo_major_groups,
        "jo_cell_types": jo_cell_types,
        "ammc_cell_types": list(data["ammc_cell_types"]),
        "ammc_nts": list(data["ammc_nts"]),
        "ammc_nt_signs": ammc_nt_signs,
        "wv_cell_types": list(data["wedvlp_cell_types"][wv_mask]),
        "wv_nts": list(data["wedvlp_nts"][wv_mask]),
        "w_jo_ammc": w_jo_ammc,
        "w_ammc_wedvlp": w_ammc_wedvlp,
        "w_ammc_ammc": w_ammc_ammc,
    }


def _get_connectome(subtypes: Optional[tuple] = None) -> dict:
    key = subtypes
    if key not in _CACHE:
        _CACHE[key] = _load_connectome(subtypes)
    return _CACHE[key]


def _build_jo_tuning(jo_major_groups: List[str], seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_jo = len(jo_major_groups)
    tuning = np.zeros((n_jo, N_FREQ_BANDS))
    freq_arr = np.array(FREQ_BANDS, dtype=float)
    for i, group in enumerate(jo_major_groups):
        center_hz, bw_hz, _ = JO_TUNING.get(group, (180, 150, False))
        jittered_center = center_hz + rng.normal(0, bw_hz * 0.1)
        jittered_bw = bw_hz * (0.8 + 0.4 * rng.random())
        sigma = jittered_bw / 2.0
        tuning[i] = np.exp(-0.5 * ((freq_arr - jittered_center) / sigma) ** 2)
    return tuning


N_OUTPUT = 2


# ═════════════════════════════════════════════════════════════════════
#  Peripheral encoder
# ═════════════════════════════════════════════════════════════════════

class DrosophilaAuditoryEncoder(AuditoryEncoder):
    """JO neuron peripheral encoding for adult Drosophila.

    Maps frequency-band sound input to JO neuron firing rates,
    with subtype-specific tuning from FlyWire v3.1 annotations.

    subtypes: tuple of JO subtype letters to include, e.g. ("A", "B")
              for sound-sensitive only. None means all 1103 neurons.
    """

    def __init__(
        self,
        jo_tuning: Optional[np.ndarray] = None,
        jo_gain: Optional[Dict[str, float]] = None,
        tau_jo: float = 5.0,
        tau_adapt: float = 200.0,
        adapt_strength: float = 0.3,
        subtypes: Optional[tuple] = None,
    ) -> None:
        self.subtypes = subtypes
        conn = _get_connectome(subtypes)
        self._jo_cell_types = conn["jo_cell_types"]
        self._jo_major_groups = conn["jo_major_groups"]
        self._jo_tuning = jo_tuning if jo_tuning is not None else _build_jo_tuning(conn["jo_major_groups"])
        self._names = [f"{ct}_{i:04d}" for i, ct in enumerate(self._jo_cell_types)]
        self.jo_gain = jo_gain if jo_gain is not None else {n: 1.0 for n in self._names}
        self.tau_jo = tau_jo
        self.tau_adapt = tau_adapt
        self.adapt_strength = adapt_strength
        self._gain_arr = np.array([self.jo_gain.get(n, 1.0) for n in self._names])

    @property
    def receptor_names(self) -> List[str]:
        return list(self._names)

    @property
    def stimulus_profiles(self) -> Dict[str, Dict]:
        return SOUND_PROFILES

    @property
    def stimulus_valence(self) -> Dict[str, str]:
        return SOUND_VALENCE

    def encode(self, sound_name: str, intensity: float) -> Dict[str, float]:
        arr = self.encode_array(sound_name, intensity)
        return {n: float(arr[i]) for i, n in enumerate(self._names)}

    def encode_array(self, sound_name: str, intensity: float) -> np.ndarray:
        profile = SOUND_PROFILES.get(sound_name, {})
        freq_input = np.array([profile.get(f, 0.0) for f in FREQ_BANDS]) * intensity
        return np.maximum((self._jo_tuning @ freq_input) * self._gain_arr, 0.0)

    def encode_freq_input(self, freq_input: np.ndarray) -> np.ndarray:
        return np.maximum((self._jo_tuning @ freq_input) * self._gain_arr, 0.0)

    def encode_timesteps(
        self, stimulus: np.ndarray, dt: Optional[float] = None,
    ) -> np.ndarray:
        if dt is None:
            dt = 0.5
        n_steps = stimulus.shape[0]
        decay_jo = np.exp(-dt / self.tau_jo)
        onset_jo = 1.0 - decay_jo
        decay_adapt = np.exp(-dt / self.tau_adapt)
        onset_adapt = 1.0 - decay_adapt

        jo_state = np.zeros(self.n_receptors)
        adapt_state = np.zeros(self.n_receptors)
        jo_rec = np.empty((n_steps, self.n_receptors))

        for t in range(n_steps):
            drive = self.encode_freq_input(stimulus[t])
            adapt_state = decay_adapt * adapt_state + onset_adapt * drive
            adapted = np.maximum(drive - self.adapt_strength * adapt_state, 0.0)
            jo_state = decay_jo * jo_state + onset_jo * adapted
            jo_state = np.maximum(jo_state, 0.0)
            jo_rec[t] = jo_state

        return jo_rec

    def clone_varied(self, seed: int, jitter_scale: float) -> "DrosophilaAuditoryEncoder":
        rng = np.random.default_rng(seed)
        cv = 0.40 * jitter_scale
        new_gain = {}
        for n, g in self.jo_gain.items():
            if cv > 0:
                sigma = np.sqrt(np.log(1.0 + cv ** 2))
                mu = -0.5 * sigma ** 2
                factor = float(rng.lognormal(mu, sigma))
            else:
                factor = 1.0
            new_gain[n] = max(0.3, g * factor)
        return DrosophilaAuditoryEncoder(
            jo_tuning=self._jo_tuning.copy(),
            jo_gain=new_gain,
            tau_jo=self.tau_jo,
            tau_adapt=self.tau_adapt,
            adapt_strength=self.adapt_strength,
            subtypes=self.subtypes,
        )


# ═════════════════════════════════════════════════════════════════════
#  Circuit model
# ═════════════════════════════════════════════════════════════════════

class DrosophilaAuditoryCircuit(AuditoryCircuit):
    """AMMC → WED/VLP → output circuit from FlyWire v783 connectome.

    Takes JO neuron rates as input, propagates through 150 AMMC and
    ~324 WED/VLP neurons to a 2-channel behavioral output.
    """

    def __init__(
        self,
        encoder: DrosophilaAuditoryEncoder,
        w_jo_ammc: Optional[np.ndarray] = None,
        w_ammc_wedvlp: Optional[np.ndarray] = None,
        w_ammc_ammc: Optional[np.ndarray] = None,
        ammc_nt_signs: Optional[np.ndarray] = None,
        ammc_baseline: float = 2.0,
        lateral_gain: float = 0.05,
        vlp_threshold: float = 0.2,
        response_threshold: float = 0.05,
        tau_ammc: float = 10.0,
        tau_wedvlp: float = 20.0,
        tau_output: float = 50.0,
        dt: float = 0.5,
    ) -> None:
        conn = _get_connectome(encoder.subtypes)
        self.encoder = encoder
        self._w_jo_ammc = w_jo_ammc if w_jo_ammc is not None else conn["w_jo_ammc"].copy()
        self._w_ammc_wedvlp = w_ammc_wedvlp if w_ammc_wedvlp is not None else conn["w_ammc_wedvlp"].copy()
        self._w_ammc_ammc = w_ammc_ammc if w_ammc_ammc is not None else conn["w_ammc_ammc"].copy()
        self._ammc_nt_signs = ammc_nt_signs if ammc_nt_signs is not None else conn["ammc_nt_signs"].copy()
        self._ammc_cell_types = conn["ammc_cell_types"]
        self._wv_cell_types = conn["wv_cell_types"]
        self.ammc_baseline = ammc_baseline
        self.lateral_gain = lateral_gain
        self.vlp_threshold = vlp_threshold
        self.response_threshold = response_threshold
        self.tau_ammc = tau_ammc
        self.tau_wedvlp = tau_wedvlp
        self.tau_output = tau_output
        self.dt = dt
        self._ammc_names = [f"AMMC_{ct}_{i:03d}" for i, ct in enumerate(self._ammc_cell_types)]
        self._wv_names = [f"WV_{ct}_{i:04d}" for i, ct in enumerate(self._wv_cell_types)]
        self._w_output: Optional[np.ndarray] = None
        self._build_output()

    @property
    def bottleneck_names(self) -> List[str]:
        return list(self._ammc_names)

    @property
    def output_names(self) -> List[str]:
        return ["courtship_response", "startle_escape"]

    @property
    def wedvlp_names(self) -> List[str]:
        return list(self._wv_names)

    @property
    def n_ammc(self) -> int:
        return self._w_jo_ammc.shape[0]

    @property
    def n_wedvlp(self) -> int:
        return self._w_ammc_wedvlp.shape[0]

    # ── Internal forward pass ────────────────────────────────────────

    def _ammc_rates(self, jo_rates: np.ndarray) -> np.ndarray:
        w_norm = self._w_jo_ammc / (self._w_jo_ammc.max() + 1e-9)
        raw = w_norm @ jo_rates
        if self.lateral_gain > 0:
            w_lat = self._w_ammc_ammc / (self._w_ammc_ammc.max() + 1e-9)
            lateral = (w_lat * self._ammc_nt_signs[:, None]) @ raw
            raw = raw + self.lateral_gain * lateral
        return np.maximum(raw + self.ammc_baseline, 0.0)

    def _wedvlp_rates(self, ammc_rates: np.ndarray) -> np.ndarray:
        w_norm = self._w_ammc_wedvlp / (self._w_ammc_wedvlp.max() + 1e-9)
        raw = w_norm @ ammc_rates
        peak = raw.max()
        if peak < 1e-9:
            return np.zeros(self.n_wedvlp)
        normed = raw / peak
        return np.where(normed > self.vlp_threshold, normed, 0.0)

    def _threshold_wedvlp(self, wv: np.ndarray) -> np.ndarray:
        peak = wv.max()
        if peak < 1e-9:
            return np.zeros_like(wv)
        normed = wv / peak
        return np.where(normed > self.vlp_threshold, wv, 0.0)

    def _ammc_drive(self, jo_rates: np.ndarray) -> np.ndarray:
        w_norm = self._w_jo_ammc / (self._w_jo_ammc.max() + 1e-9)
        raw = w_norm @ jo_rates
        if self.lateral_gain > 0:
            w_lat = self._w_ammc_ammc / (self._w_ammc_ammc.max() + 1e-9)
            lateral = (w_lat * self._ammc_nt_signs[:, None]) @ raw
            raw = raw + self.lateral_gain * lateral
        return raw + self.ammc_baseline

    def _wedvlp_drive(self, ammc_rates: np.ndarray) -> np.ndarray:
        w_norm = self._w_ammc_wedvlp / (self._w_ammc_wedvlp.max() + 1e-9)
        return w_norm @ ammc_rates

    def _behavior_from_output(self, out: np.ndarray) -> str:
        mag = abs(out[0]) + abs(out[1])
        if mag < self.response_threshold:
            return "no_response"
        if out[0] > out[1] and out[0] > self.response_threshold:
            return "courtship_response"
        elif out[1] > out[0] and out[1] > self.response_threshold:
            return "startle_escape"
        return "no_response"

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for layer_name, names in [
            ("jo", self.encoder.receptor_names), ("ammc", self._ammc_names),
            ("wedvlp", self._wv_names), ("output", self.output_names),
        ]:
            if layer_name in layers:
                for i, n in enumerate(names):
                    d[n] = float(layers[layer_name][i])
        return d

    def _forward_arrays(self, jo_rates: np.ndarray) -> Dict[str, np.ndarray]:
        ammc = self._ammc_rates(jo_rates)
        wedvlp = self._wedvlp_rates(ammc)
        out = self._w_output @ wedvlp
        return {"jo": jo_rates, "ammc": ammc, "wedvlp": wedvlp, "output": out}

    def _forward_from_ammc_arrays(self, ammc_rates: np.ndarray) -> Dict[str, np.ndarray]:
        wedvlp = self._wedvlp_rates(ammc_rates)
        out = self._w_output @ wedvlp
        return {"ammc": ammc_rates, "wedvlp": wedvlp, "output": out}

    # ── Output weight fitting ────────────────────────────────────────

    def _temporal_wedvlp_avg(self, stimulus: np.ndarray) -> np.ndarray:
        dt = self.dt
        n_steps = stimulus.shape[0]
        d_ammc = np.exp(-dt / self.tau_ammc)
        o_ammc = 1.0 - d_ammc
        d_wv = np.exp(-dt / self.tau_wedvlp)
        o_wv = 1.0 - d_wv

        jo_rec = self.encoder.encode_timesteps(stimulus, dt)
        ammc_state = np.zeros(self.n_ammc)
        wv_state = np.zeros(self.n_wedvlp)

        avg_window = max(1, int(100.0 / dt))
        wv_accum = np.zeros(self.n_wedvlp)
        count = 0

        for t in range(n_steps):
            ammc_drive = self._ammc_drive(jo_rec[t])
            ammc_state = d_ammc * ammc_state + o_ammc * ammc_drive
            ammc_state = np.maximum(ammc_state, 0.0)
            wv_drive = self._wedvlp_drive(ammc_state)
            wv_state = d_wv * wv_state + o_wv * wv_drive
            wv_state = np.maximum(wv_state, 0.0)
            wv_out = self._threshold_wedvlp(wv_state)
            if t >= n_steps - avg_window:
                wv_accum += wv_out
                count += 1

        return wv_accum / max(count, 1)

    def _build_output(self) -> None:
        rng = np.random.default_rng(42)
        vlp_vecs, targets = [], []

        def _add(vlp: np.ndarray, valence: str) -> None:
            vlp_vecs.append(vlp)
            if valence == "courtship":
                targets.append([1.0, 0.0])
            elif valence == "startle":
                targets.append([0.0, 1.0])
            else:
                targets.append([0.0, 0.0])

        for sound in SOUND_PROFILES:
            jo = self.encoder.encode_array(sound, 1.0)
            vlp = self._wedvlp_rates(self._ammc_rates(jo))
            _add(vlp, SOUND_VALENCE.get(sound, "neutral"))

        for sound in SOUND_PROFILES:
            stim = make_constant(500.0, dt=self.dt, sound_name=sound)
            _add(self._temporal_wedvlp_avg(stim), SOUND_VALENCE.get(sound, "neutral"))

        _add(self._temporal_wedvlp_avg(make_pulse_train(500.0, dt=self.dt, ipi_ms=35.0)), "courtship")
        _add(self._temporal_wedvlp_avg(make_sine_song(500.0, dt=self.dt)), "courtship")

        X = np.array(vlp_vecs)
        Y = np.array(targets)
        if X.shape[0] > 0 and X.shape[1] > 0:
            w, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            self._w_output = w.T + rng.normal(0, 0.001, (N_OUTPUT, self.n_wedvlp))
        else:
            self._w_output = rng.normal(0, 0.01, (N_OUTPUT, self.n_wedvlp))

    # ── AuditoryCircuit interface ────────────────────────────────────

    def forward_from_receptors(self, receptor_signal: Dict[str, float]) -> Dict:
        jo_arr = np.array([receptor_signal.get(n, 0.0) for n in self.encoder.receptor_names])
        layers = self._forward_arrays(jo_arr)
        return {"behavior": self._behavior_from_output(layers["output"]),
                "neurons": self._layers_to_dict(layers)}

    def forward_from_bottleneck(self, bottleneck_signal: Dict[str, float]) -> Dict:
        ammc_arr = np.array([bottleneck_signal.get(n, 0.0) for n in self._ammc_names])
        layers = self._forward_from_ammc_arrays(ammc_arr)
        return {"behavior": self._behavior_from_output(layers["output"]),
                "neurons": self._layers_to_dict(layers)}

    def natural_response(self, sound_name: str, intensity: float) -> Dict[str, float]:
        jo = self.encoder.encode_array(sound_name, intensity)
        layers = self._forward_arrays(jo)
        return self._layers_to_dict(layers)

    def natural_behavior(self, sound_name: str, intensity: float) -> str:
        jo = self.encoder.encode_array(sound_name, intensity)
        layers = self._forward_arrays(jo)
        return self._behavior_from_output(layers["output"])

    def forward_timesteps_from_receptors(
        self, receptor_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        n_steps = receptor_timeseries.shape[0]
        d_ammc = np.exp(-dt / self.tau_ammc)
        o_ammc = 1.0 - d_ammc
        d_wv = np.exp(-dt / self.tau_wedvlp)
        o_wv = 1.0 - d_wv
        d_out = np.exp(-dt / self.tau_output)
        o_out = 1.0 - d_out

        ammc_state = np.zeros(self.n_ammc)
        wv_state = np.zeros(self.n_wedvlp)
        out_state = np.zeros(N_OUTPUT)
        ammc_rec = np.empty((n_steps, self.n_ammc))
        wv_rec = np.empty((n_steps, self.n_wedvlp))
        out_rec = np.empty((n_steps, N_OUTPUT))

        for t in range(n_steps):
            ammc_drive = self._ammc_drive(receptor_timeseries[t])
            ammc_state = d_ammc * ammc_state + o_ammc * ammc_drive
            ammc_state = np.maximum(ammc_state, 0.0)
            wv_drive = self._wedvlp_drive(ammc_state)
            wv_state = d_wv * wv_state + o_wv * wv_drive
            wv_state = np.maximum(wv_state, 0.0)
            wv_out = self._threshold_wedvlp(wv_state)
            out_state = d_out * out_state + o_out * (self._w_output @ wv_out)
            ammc_rec[t] = ammc_state
            wv_rec[t] = wv_out
            out_rec[t] = out_state

        avg_window = max(1, int(100.0 / dt))
        behavior = self._behavior_from_output(out_rec[-avg_window:].mean(axis=0))
        return {"t_ms": np.arange(n_steps) * dt, "ammc": ammc_rec,
                "wedvlp": wv_rec, "output": out_rec, "behavior": behavior}

    def forward_timesteps_from_bottleneck(
        self, bottleneck_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        n_steps = bottleneck_timeseries.shape[0]
        d_wv = np.exp(-dt / self.tau_wedvlp)
        o_wv = 1.0 - d_wv
        d_out = np.exp(-dt / self.tau_output)
        o_out = 1.0 - d_out

        wv_state = np.zeros(self.n_wedvlp)
        out_state = np.zeros(N_OUTPUT)
        wv_rec = np.empty((n_steps, self.n_wedvlp))
        out_rec = np.empty((n_steps, N_OUTPUT))

        for t in range(n_steps):
            wv_drive = self._wedvlp_drive(bottleneck_timeseries[t])
            wv_state = d_wv * wv_state + o_wv * wv_drive
            wv_state = np.maximum(wv_state, 0.0)
            wv_out = self._threshold_wedvlp(wv_state)
            out_state = d_out * out_state + o_out * (self._w_output @ wv_out)
            wv_rec[t] = wv_out
            out_rec[t] = out_state

        avg_window = max(1, int(100.0 / dt))
        behavior = self._behavior_from_output(out_rec[-avg_window:].mean(axis=0))
        return {"t_ms": np.arange(n_steps) * dt, "ammc": bottleneck_timeseries,
                "wedvlp": wv_rec, "output": out_rec, "behavior": behavior}


# ═════════════════════════════════════════════════════════════════════
#  Composed model (backward-compatible API)
# ═════════════════════════════════════════════════════════════════════

@dataclass
class AuditoryAdultModel:
    """Composed encoder + circuit with the original API.

    Access the pieces directly via .encoder and .circuit for use with
    the transfer pipeline or to swap in a different encoder (human, mouse).
    """

    ammc_baseline: float = 2.0
    lateral_gain: float = 0.05
    vlp_threshold: float = 0.2
    response_threshold: float = 0.05
    dt: float = 0.5
    tau_jo: float = 5.0
    tau_ammc: float = 10.0
    tau_wedvlp: float = 20.0
    tau_output: float = 50.0
    tau_adapt: float = 200.0
    adapt_strength: float = 0.3
    subtypes: Optional[tuple] = None

    jo_gain: Dict[str, float] = field(default_factory=dict)

    encoder: Optional[DrosophilaAuditoryEncoder] = field(default=None, repr=False)
    circuit: Optional[DrosophilaAuditoryCircuit] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            gain = self.jo_gain if self.jo_gain else None
            self.encoder = DrosophilaAuditoryEncoder(
                jo_gain=gain, tau_jo=self.tau_jo,
                tau_adapt=self.tau_adapt, adapt_strength=self.adapt_strength,
                subtypes=self.subtypes,
            )
        if not self.jo_gain:
            self.jo_gain = dict(self.encoder.jo_gain)
        if self.circuit is None:
            self.circuit = DrosophilaAuditoryCircuit(
                encoder=self.encoder,
                ammc_baseline=self.ammc_baseline, lateral_gain=self.lateral_gain,
                vlp_threshold=self.vlp_threshold, response_threshold=self.response_threshold,
                tau_ammc=self.tau_ammc, tau_wedvlp=self.tau_wedvlp,
                tau_output=self.tau_output, dt=self.dt,
            )

    # ── Delegated properties ─────────────────────────────────────────

    @property
    def jo_names(self) -> List[str]:
        return self.encoder.receptor_names

    @property
    def ammc_names(self) -> List[str]:
        return self.circuit.bottleneck_names

    @property
    def wedvlp_names(self) -> List[str]:
        return self.circuit.wedvlp_names

    @property
    def output_names(self) -> List[str]:
        return self.circuit.output_names

    @property
    def n_jo(self) -> int:
        return self.encoder.n_receptors

    @property
    def n_ammc(self) -> int:
        return self.circuit.n_ammc

    @property
    def n_wedvlp(self) -> int:
        return self.circuit.n_wedvlp

    # ── Backward-compatible internal accessors ───────────────────────

    @property
    def _w_jo_ammc(self) -> np.ndarray:
        return self.circuit._w_jo_ammc

    @property
    def _w_ammc_wedvlp(self) -> np.ndarray:
        return self.circuit._w_ammc_wedvlp

    @property
    def _w_ammc_ammc(self) -> np.ndarray:
        return self.circuit._w_ammc_ammc

    @property
    def _jo_tuning(self) -> np.ndarray:
        return self.encoder._jo_tuning

    def _jo_rates(self, sound_name: str, intensity: float) -> np.ndarray:
        return self.encoder.encode_array(sound_name, intensity)

    def _ammc_rates(self, jo_rates: np.ndarray) -> np.ndarray:
        return self.circuit._ammc_rates(jo_rates)

    def _wedvlp_rates(self, ammc_rates: np.ndarray) -> np.ndarray:
        return self.circuit._wedvlp_rates(ammc_rates)

    def _forward(self, sound_name: str, intensity: float) -> Dict[str, np.ndarray]:
        jo = self.encoder.encode_array(sound_name, intensity)
        return self.circuit._forward_arrays(jo)

    def _forward_from_ammc(self, ammc_rates: np.ndarray) -> Dict[str, np.ndarray]:
        return self.circuit._forward_from_ammc_arrays(ammc_rates)

    def _behavior_from_output(self, out: np.ndarray) -> str:
        return self.circuit._behavior_from_output(out)

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        return self.circuit._layers_to_dict(layers)

    # ── Public interface ─────────────────────────────────────────────

    def receptor_activity(self, stimulus_intensity: float, sound_type: str = "courtship") -> Dict[str, float]:
        return self.encoder.encode(_pick_sound(sound_type), stimulus_intensity)

    def network_state(self, stimulus_intensity: float, sound_type: str = "courtship") -> Dict[str, float]:
        return self.circuit.natural_response(_pick_sound(sound_type), stimulus_intensity)

    def simulate(self, stimulus_intensity: float, sound_type: str = "courtship") -> Dict:
        sound_name = _pick_sound(sound_type)
        jo = self.encoder.encode(sound_name, stimulus_intensity)
        return self.circuit.forward_from_receptors(jo)

    def simulate_from_receptors(self, receptor_signal: Dict[str, float], sound_type: str = "courtship") -> Dict:
        return self.circuit.forward_from_receptors(receptor_signal)

    def simulate_from_ammc(self, ammc_signal: Dict[str, float], sound_type: str = "courtship") -> Dict:
        return self.circuit.forward_from_bottleneck(ammc_signal)

    def sound_response(self, stimulus_intensity: float, sound_type: str = "courtship") -> Dict[str, float]:
        return self.receptor_activity(stimulus_intensity, sound_type)

    def behavior_from_signal(self, receptor_signal: Dict[str, float], sound_type: str = "courtship") -> str:
        return self.simulate_from_receptors(receptor_signal)["behavior"]

    def ammc_response(self, stimulus_intensity: float, sound_type: str = "courtship") -> Dict[str, float]:
        sound_name = _pick_sound(sound_type)
        jo = self.encoder.encode_array(sound_name, stimulus_intensity)
        ammc = self.circuit._ammc_rates(jo)
        return {n: float(ammc[i]) for i, n in enumerate(self.ammc_names)}

    def simulate_timesteps(self, stimulus: np.ndarray, dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        jo_rec = self.encoder.encode_timesteps(stimulus, dt)
        result = self.circuit.forward_timesteps_from_receptors(jo_rec, dt)
        result["jo"] = jo_rec
        return result

    def simulate_timesteps_from_ammc(self, ammc_input: np.ndarray, dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        return self.circuit.forward_timesteps_from_bottleneck(ammc_input, dt)

    def validate_sound_responses(self) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        court = self.simulate(1.0, "courtship")
        checks["courtship_response"] = court["behavior"] == "courtship_response"
        startle = self.simulate(1.0, "startle")
        checks["startle_escape"] = startle["behavior"] == "startle_escape"
        silent = self.simulate(0.0, "courtship")
        checks["silence_no_response"] = silent["behavior"] == "no_response"
        jo = self.receptor_activity(1.0, "courtship")
        checks["jo_responds"] = max(jo.values()) > 0
        pulse = make_pulse_train(500.0, dt=self.dt)
        checks["temporal_pulse_courtship"] = self.simulate_timesteps(pulse)["behavior"] == "courtship_response"
        buzz = make_constant(500.0, dt=self.dt, sound_name="predator_buzz")
        checks["temporal_buzz_startle"] = self.simulate_timesteps(buzz)["behavior"] == "startle_escape"
        silence_t = make_constant(500.0, dt=self.dt, sound_name="silence")
        checks["temporal_silence_no_response"] = self.simulate_timesteps(silence_t)["behavior"] == "no_response"
        return checks

    def clone_perturbed(
        self,
        jo_gain: Optional[Dict[str, float]] = None,
        jo_ammc_weight_noise: Optional[np.ndarray] = None,
        ammc_wed_weight_noise: Optional[np.ndarray] = None,
        wed_vlp_weight_noise: Optional[np.ndarray] = None,
    ) -> "AuditoryAdultModel":
        new_encoder = DrosophilaAuditoryEncoder(
            jo_tuning=self.encoder._jo_tuning.copy(),
            jo_gain=dict(jo_gain) if jo_gain else dict(self.encoder.jo_gain),
            tau_jo=self.encoder.tau_jo,
            tau_adapt=self.encoder.tau_adapt,
            adapt_strength=self.encoder.adapt_strength,
            subtypes=self.encoder.subtypes,
        )
        w1 = self.circuit._w_jo_ammc.copy()
        w2 = self.circuit._w_ammc_wedvlp.copy()
        w3 = self.circuit._w_ammc_ammc.copy()
        if jo_ammc_weight_noise is not None:
            w1 = np.maximum(w1 + jo_ammc_weight_noise, 0.0)
        if ammc_wed_weight_noise is not None:
            w2 = np.maximum(w2 + ammc_wed_weight_noise, 0.0)
        if wed_vlp_weight_noise is not None:
            w2 = np.maximum(w2 + wed_vlp_weight_noise, 0.0)

        new_circuit = DrosophilaAuditoryCircuit(
            encoder=new_encoder,
            w_jo_ammc=w1, w_ammc_wedvlp=w2, w_ammc_ammc=w3,
            ammc_nt_signs=self.circuit._ammc_nt_signs.copy(),
            ammc_baseline=self.circuit.ammc_baseline,
            lateral_gain=self.circuit.lateral_gain,
            vlp_threshold=self.circuit.vlp_threshold,
            response_threshold=self.circuit.response_threshold,
            tau_ammc=self.circuit.tau_ammc,
            tau_wedvlp=self.circuit.tau_wedvlp,
            tau_output=self.circuit.tau_output,
            dt=self.dt,
        )

        clone = AuditoryAdultModel.__new__(AuditoryAdultModel)
        clone.ammc_baseline = self.ammc_baseline
        clone.lateral_gain = self.lateral_gain
        clone.vlp_threshold = self.vlp_threshold
        clone.response_threshold = self.response_threshold
        clone.dt = self.dt
        clone.tau_jo = self.tau_jo
        clone.tau_ammc = self.tau_ammc
        clone.tau_wedvlp = self.tau_wedvlp
        clone.tau_output = self.tau_output
        clone.tau_adapt = self.tau_adapt
        clone.adapt_strength = self.adapt_strength
        clone.subtypes = self.subtypes
        clone.jo_gain = dict(new_encoder.jo_gain)
        clone.encoder = new_encoder
        clone.circuit = new_circuit
        return clone
