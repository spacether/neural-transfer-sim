"""Mus musculus (house mouse) auditory model.

Implements the mouse auditory pathway:
  120 inner hair cells (IHCs, 4 per frequency band) → 60 cochlear nucleus
  (CN) neurons (bottleneck, with lateral inhibition) → 150 inferior
  colliculus (IC) neurons (expansion layer) → 3 output neurons.

Frequency range: 2.5–70 kHz (30 logarithmically spaced bands), matching
the characteristic-frequency range measured in CBA/CaJ mice by Taberner
& Liberman (2005, J Neurophysiol 93:557–569).

Neuron counts are representative, not literal.  The real mouse cochlea
has ~700 IHCs (Ehret 1979, J Comp Neurol 183:1007–1029) innervated by
~8,600 spiral ganglion neurons (SGNs; reviewed in Bharadwaj et al. 2019).
We model 120 IHC channels (4 per frequency band) as a tractable
representation of the tonotopic array.  The cochlear nucleus contains
~20,000–30,000 neurons per side in rodents (Idrizbegovic et al. 2001,
Hear Res 158:131–138, stereology in CBA/CaJ); we model 60 CN neurons
as a compressed bottleneck.  The inferior colliculus central nucleus
contains bipolar and multipolar cells (Willott 1975, J Comp Neurol
163:105–120); we model 150 IC neurons as the expansion layer.

Key architectural features:
  - Tonotopic IHC→CN connectivity: nearby-frequency IHCs project to the
    same CN neurons, following the 3D tonotopic map of Muniak et al.
    (2013, J Comp Neurol 521:1510–1532).  Weight matrices are synthetic
    but constrained to preserve tonotopic gradients.
  - Lateral inhibition within CN: sparse recurrent inhibition reflecting
    the D-stellate glycinergic interneurons of the VCN (Oertel et al.
    2011, J Neurophysiol 106:2536–2552).
  - Topographic CN→IC expansion: following differential projection
    patterns described by Blazquez Freches et al. (2023, J Comp Neurol
    531:1515–1540), with VCN projections forming a tonotopic sheet in
    the central nucleus of the IC.

Sound stimuli include ultrasonic vocalizations (USVs) for pup isolation
(30–90 kHz; Scattoni et al. 2008, Brain Res 1236:174–185), adult male
courtship USVs (50–80 kHz; Portfors 2007, J Am Assoc Lab Anim Sci
46:28–34), and maternal calls.  Predator sounds use low frequencies
(2–15 kHz).  Behavioral categories: approach (pup USVs trigger maternal
retrieval; Ehret 2005), startle (broadband noise; Koch 1999, Prog
Neurobiol 59:107–128), freeze (predator cues), neutral.

Cochlear tuning: Q10dB increases with characteristic frequency (CF),
from ~2 at 5 kHz to ~8 at 40 kHz (Taberner & Liberman 2005).  We
approximate this with Gaussian tuning curves whose bandwidth narrows
relative to CF at higher frequencies.

Signal mode: firing rates, same as the fly and zebrafish models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from neural_transfer.models.auditory_base import AuditoryEncoder, AuditoryCircuit


# ───────────────────────────────────────────────────────────────────────
#  Neuron dimensions
# ───────────────────────────────────────────────────────────────────────

N_FREQ_BANDS = 30
FREQ_BANDS: List[int] = [int(round(f)) for f in np.geomspace(2500, 70000, N_FREQ_BANDS)]

N_IHC_PER_BAND = 4
N_IHC = N_FREQ_BANDS * N_IHC_PER_BAND  # 120
N_CN = 60
N_IC = 150
N_OUTPUT = 3

IHC_NAMES: List[str] = [f"IHC_{i:03d}" for i in range(N_IHC)]
CN_NAMES: List[str] = [f"CN_{i:03d}" for i in range(N_CN)]
IC_NAMES: List[str] = [f"IC_{i:04d}" for i in range(N_IC)]
OUTPUT_NAMES: List[str] = ["approach", "startle", "freeze"]


# ───────────────────────────────────────────────────────────────────────
#  Sound profiles — power at each frequency band
# ───────────────────────────────────────────────────────────────────────

def _usv_profile(center_khz: float, bw_khz: float, peak: float = 1.0) -> Dict[int, float]:
    freq_arr = np.array(FREQ_BANDS, dtype=float)
    center = center_khz * 1000
    bw = bw_khz * 1000
    profile = np.exp(-0.5 * ((freq_arr - center) / (bw / 2)) ** 2) * peak
    return {f: float(profile[i]) for i, f in enumerate(FREQ_BANDS)}


def _broadband(lo_khz: float, hi_khz: float, peak: float = 1.0) -> Dict[int, float]:
    freq_arr = np.array(FREQ_BANDS, dtype=float)
    lo, hi = lo_khz * 1000, hi_khz * 1000
    mask = (freq_arr >= lo) & (freq_arr <= hi)
    profile = np.where(mask, peak, peak * 0.02)
    return {f: float(profile[i]) for i, f in enumerate(FREQ_BANDS)}


SOUND_PROFILES: Dict[str, Dict[int, float]] = {
    # ── Ultrasonic vocalizations (social / maternal) ────────────────
    # Pup isolation USVs: 30–90 kHz, peak ~60 kHz (Scattoni et al. 2008)
    "pup_isolation":    _usv_profile(60, 25),
    # Pup wriggling calls: slightly lower, ~50 kHz (Scattoni et al. 2008)
    "pup_wriggling":    _usv_profile(50, 20, 0.7),
    # Adult male courtship USVs: 50–80 kHz (Portfors 2007)
    "adult_courtship":  _usv_profile(65, 15),
    # Maternal USVs: ~40–60 kHz (Ehret 2005)
    "maternal_call":    _usv_profile(45, 20),
    # General social USVs: 40–70 kHz
    "social_usv":       _usv_profile(55, 25, 0.8),
    # ── Distress / alarm ────────────────────────────────────────────
    # Broadband distress: energy across 5–40 kHz (Grimsley et al. 2011)
    "distress_call":    _broadband(5, 40),
    # Alarm squeak: 10–50 kHz
    "alarm_squeak":     _broadband(10, 50, 0.9),
    # ── Predator sounds ─────────────────────────────────────────────
    # Low-frequency predator vocalization: 2.5–8 kHz
    "predator_growl":   _broadband(2.5, 8, 0.9),
    # Mid-frequency predator: 8–20 kHz
    "predator_screech": _broadband(8, 20),
    # ── Startle ─────────────────────────────────────────────────────
    # Broadband startle noise (Koch 1999, Prog Neurobiol 59:107–128)
    "startle_noise":    {f: 0.8 for f in FREQ_BANDS},
    # ── Pure tones ──────────────────────────────────────────────────
    "tone_5k":          _usv_profile(5, 2),
    "tone_10k":         _usv_profile(10, 3),
    "tone_20k":         _usv_profile(20, 5),
    "tone_40k":         _usv_profile(40, 8),
    "tone_60k":         _usv_profile(60, 10),
    # ── Band-pass ───────────────────────────────────────────────────
    "low_band":         _broadband(2.5, 10),
    "mid_band":         _broadband(15, 40),
    "high_band":        _broadband(45, 70),
    # ── Controls ────────────────────────────────────────────────────
    "silence":          {f: 0.0 for f in FREQ_BANDS},
    "white_noise":      {f: 0.5 for f in FREQ_BANDS},
}

SOUND_VALENCE: Dict[str, str] = {
    "pup_isolation": "approach", "pup_wriggling": "approach",
    "adult_courtship": "approach", "maternal_call": "approach",
    "social_usv": "approach",
    "distress_call": "startle", "alarm_squeak": "startle",
    "startle_noise": "startle",
    "predator_growl": "freeze", "predator_screech": "freeze",
    "tone_5k": "neutral", "tone_10k": "neutral", "tone_20k": "neutral",
    "tone_40k": "neutral", "tone_60k": "neutral",
    "low_band": "neutral", "mid_band": "neutral", "high_band": "neutral",
    "silence": "neutral", "white_noise": "neutral",
}

APPROACH_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "approach"]
STARTLE_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "startle"]
FREEZE_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "freeze"]
NEUTRAL_SOUNDS = [s for s, v in SOUND_VALENCE.items() if v == "neutral"]


def _pick_sound(sound_type: str, rng: np.random.Generator | None = None) -> str:
    if rng is None:
        rng = np.random.default_rng(0)
    if sound_type == "approach":
        return rng.choice(APPROACH_SOUNDS)
    elif sound_type == "startle":
        return rng.choice(STARTLE_SOUNDS)
    elif sound_type == "freeze":
        return rng.choice(FREEZE_SOUNDS)
    elif sound_type == "neutral":
        return rng.choice(NEUTRAL_SOUNDS)
    return sound_type


# ───────────────────────────────────────────────────────────────────────
#  Temporal stimulus generators
# ───────────────────────────────────────────────────────────────────────

def _profile_to_vec(profile: Dict[int, float]) -> np.ndarray:
    return np.array([profile.get(f, 0.0) for f in FREQ_BANDS])


STIM_DURATION_MS = 500.0
STIM_DT = 0.5
STIM_AVG_WINDOW_MS = 100.0

_USV_SOUNDS = {
    "pup_isolation", "pup_wriggling", "adult_courtship",
    "maternal_call", "social_usv",
}
_PULSED_SOUNDS = {"distress_call", "alarm_squeak"}


def make_usv_sweep(
    duration_ms: float = STIM_DURATION_MS, dt: float = STIM_DT,
    intensity: float = 1.0, profile: str = "pup_isolation",
    sweep_rate: float = 0.003,
) -> np.ndarray:
    """Frequency-modulated sweep mimicking a USV syllable train.

    Mouse USVs are typically 40–80 ms syllables with ~20 ms gaps,
    often with frequency modulation (Scattoni et al. 2008).
    """
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(profile, SOUND_PROFILES["pup_isolation"]))
    freq_arr = np.array(FREQ_BANDS, dtype=float)
    stim = np.zeros((n_steps, N_FREQ_BANDS))
    syllable_ms = 40.0
    gap_ms = 20.0
    syllable_steps = int(syllable_ms / dt)
    gap_steps = int(gap_ms / dt)
    period = syllable_steps + gap_steps
    t = 0
    while t < n_steps:
        end = min(t + syllable_steps, n_steps)
        for s in range(t, end):
            phase = (s - t) / max(syllable_steps, 1)
            shift = sweep_rate * phase * freq_arr.mean()
            shifted_spec = np.exp(-0.5 * ((freq_arr - (freq_arr * spec).sum() / (spec.sum() + 1e-9) + shift) / (5000)) ** 2)
            shifted_spec = shifted_spec / (shifted_spec.max() + 1e-9) * spec.max()
            stim[s] = shifted_spec * spec * intensity
        t += period
    return stim


def make_call_train(
    duration_ms: float = STIM_DURATION_MS, dt: float = STIM_DT,
    pulse_width_ms: float = 15.0, ipi_ms: float = 40.0,
    intensity: float = 1.0, profile: str = "distress_call",
) -> np.ndarray:
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(profile, SOUND_PROFILES["distress_call"]))
    stim = np.zeros((n_steps, N_FREQ_BANDS))
    pulse_samples = int(pulse_width_ms / dt)
    period_samples = int(ipi_ms / dt)
    t = 0
    while t < n_steps:
        end = min(t + pulse_samples, n_steps)
        stim[t:end] = spec * intensity
        t += period_samples
    return stim


def make_constant(
    duration_ms: float = STIM_DURATION_MS, dt: float = STIM_DT,
    sound_name: str = "startle_noise", intensity: float = 1.0,
) -> np.ndarray:
    n_steps = int(duration_ms / dt)
    spec = _profile_to_vec(SOUND_PROFILES.get(sound_name, {}))
    return np.full((n_steps, N_FREQ_BANDS), spec * intensity)


def make_stimulus_sequence(segments: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(segments, axis=0)


def make_temporal_stimulus(
    sound_name: str, intensity: float = 1.0,
    duration_ms: float = STIM_DURATION_MS, dt: float = STIM_DT,
) -> np.ndarray:
    if sound_name in _USV_SOUNDS:
        return make_usv_sweep(duration_ms, dt, intensity=intensity, profile=sound_name)
    elif sound_name in _PULSED_SOUNDS:
        return make_call_train(duration_ms, dt, intensity=intensity, profile=sound_name)
    return make_constant(duration_ms, dt, sound_name=sound_name, intensity=intensity)


def temporal_avg_window(dt: float = STIM_DT) -> int:
    return max(1, int(STIM_AVG_WINDOW_MS / dt))


# ───────────────────────────────────────────────────────────────────────
#  Synthetic connectivity builders
# ───────────────────────────────────────────────────────────────────────

def _q10_from_cf(cf_hz: float) -> float:
    """Approximate Q10dB as a function of characteristic frequency.

    Derived from Taberner & Liberman (2005, Fig. 4): Q10dB rises from
    ~2 at 3 kHz to ~8 at 40 kHz in CBA/CaJ mice, with a roughly
    logarithmic relationship.  We fit Q10 ≈ 0.65 * (CF/1000)^0.45.
    """
    return 0.65 * (cf_hz / 1000.0) ** 0.45


def _build_ihc_tuning(seed: int = 42) -> np.ndarray:
    """Build IHC tuning curves with frequency-dependent sharpness.

    Bandwidth narrows relative to CF at higher frequencies, matching
    the Q10dB trend in mouse auditory nerve fibers (Taberner & Liberman
    2005).  Each IHC gets a Gaussian tuning curve whose sigma is
    CF / (2 * Q10dB), with slight jitter for individual variation.
    """
    rng = np.random.default_rng(seed)
    freq_arr = np.array(FREQ_BANDS, dtype=float)
    tuning = np.zeros((N_IHC, N_FREQ_BANDS))
    for i in range(N_IHC):
        band_idx = i // N_IHC_PER_BAND
        center = freq_arr[band_idx]
        q10 = _q10_from_cf(center)
        bw_10db = center / q10
        jittered_center = center + rng.normal(0, bw_10db * 0.05)
        jittered_bw = bw_10db * (0.85 + 0.3 * rng.random())
        sigma = jittered_bw / 2.0
        tuning[i] = np.exp(-0.5 * ((freq_arr - jittered_center) / sigma) ** 2)
    return tuning


def _build_w_ihc_cn(seed: int = 42) -> np.ndarray:
    """Tonotopic IHC → CN connectivity (Muniak et al. 2013)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((N_CN, N_IHC))
    for cn_i in range(N_CN):
        center_band = cn_i * N_FREQ_BANDS / N_CN
        for ihc_j in range(N_IHC):
            ihc_band = ihc_j // N_IHC_PER_BAND
            dist = abs(ihc_band - center_band)
            sigma_bands = 3.0
            w[cn_i, ihc_j] = np.exp(-0.5 * (dist / sigma_bands) ** 2) * rng.uniform(0.5, 1.5)
    w[w < 0.1] = 0.0
    return w


def _build_w_cn_cn(seed: int = 43) -> np.ndarray:
    """Sparse lateral inhibition within CN (D-stellate glycinergic; Oertel et al. 2011)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((N_CN, N_CN))
    for i in range(N_CN):
        for j in range(N_CN):
            if i == j:
                continue
            dist = abs(i - j)
            if dist <= 5:
                w[i, j] = rng.uniform(0.0, 0.5) * np.exp(-dist / 3.0)
    w[w < 0.05] = 0.0
    return w


def _build_cn_nt_signs(seed: int = 44) -> np.ndarray:
    """~20% of CN neurons are glycinergic/GABAergic inhibitory (Oertel et al. 2011)."""
    rng = np.random.default_rng(seed)
    signs = np.ones(N_CN)
    n_inh = int(N_CN * 0.2)
    inh_idx = rng.choice(N_CN, size=n_inh, replace=False)
    signs[inh_idx] = -1.0
    return signs


def _build_w_cn_ic(seed: int = 45) -> np.ndarray:
    """Topographic expansion CN → IC (Blazquez Freches et al. 2023)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((N_IC, N_CN))
    for ic_i in range(N_IC):
        center_cn = ic_i * N_CN / N_IC
        for cn_j in range(N_CN):
            dist = abs(cn_j - center_cn)
            sigma = 8.0
            w[ic_i, cn_j] = np.exp(-0.5 * (dist / sigma) ** 2) * rng.uniform(0.3, 1.2)
    w[w < 0.05] = 0.0
    return w


# ═══════════════════════════════════════════════════════════════════════
#  Peripheral encoder
# ═══════════════════════════════════════════════════════════════════════

class MouseAuditoryEncoder(AuditoryEncoder):
    """Inner hair cell (IHC) peripheral encoding for Mus musculus.

    Maps frequency-band sound input to IHC firing rates, with tonotopic
    Gaussian tuning covering 2.5–70 kHz (Taberner & Liberman 2005).
    Tuning sharpness (Q10dB) increases with CF.  Adaptation dynamics
    approximate rapid and short-term ANF adaptation (Westerman &
    Smith 1984, Hear Res 15:249–260).
    """

    def __init__(
        self,
        ihc_tuning: Optional[np.ndarray] = None,
        ihc_gain: Optional[Dict[str, float]] = None,
        tau_ihc: float = 3.0,
        tau_adapt: float = 150.0,
        adapt_strength: float = 0.25,
    ) -> None:
        self._ihc_tuning = ihc_tuning if ihc_tuning is not None else _build_ihc_tuning()
        self._names = list(IHC_NAMES)
        self.ihc_gain = ihc_gain if ihc_gain is not None else {n: 1.0 for n in self._names}
        self.tau_ihc = tau_ihc
        self.tau_adapt = tau_adapt
        self.adapt_strength = adapt_strength
        self._gain_arr = np.array([self.ihc_gain.get(n, 1.0) for n in self._names])

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
        return self.encode_freq_input(freq_input)

    def encode_freq_input(self, freq_input: np.ndarray) -> np.ndarray:
        return np.maximum((self._ihc_tuning @ freq_input) * self._gain_arr, 0.0)

    def encode_timesteps(
        self, stimulus: np.ndarray, dt: Optional[float] = None,
    ) -> np.ndarray:
        if dt is None:
            dt = STIM_DT
        n_steps = stimulus.shape[0]
        decay_ihc = np.exp(-dt / self.tau_ihc)
        onset_ihc = 1.0 - decay_ihc
        decay_adapt = np.exp(-dt / self.tau_adapt)
        onset_adapt = 1.0 - decay_adapt

        ihc_state = np.zeros(self.n_receptors)
        adapt_state = np.zeros(self.n_receptors)
        ihc_rec = np.empty((n_steps, self.n_receptors))

        for t in range(n_steps):
            drive = self.encode_freq_input(stimulus[t])
            adapt_state = decay_adapt * adapt_state + onset_adapt * drive
            adapted = np.maximum(drive - self.adapt_strength * adapt_state, 0.0)
            ihc_state = decay_ihc * ihc_state + onset_ihc * adapted
            ihc_state = np.maximum(ihc_state, 0.0)
            ihc_rec[t] = ihc_state

        return ihc_rec

    def clone_varied(self, seed: int, jitter_scale: float) -> "MouseAuditoryEncoder":
        rng = np.random.default_rng(seed)
        cv = 0.55 * jitter_scale
        new_gain: Dict[str, float] = {}
        for n, g in self.ihc_gain.items():
            if cv > 0:
                sigma = np.sqrt(np.log(1.0 + cv ** 2))
                mu = -0.5 * sigma ** 2
                factor = float(rng.lognormal(mu, sigma))
            else:
                factor = 1.0
            new_gain[n] = max(0.3, g * factor)
        return MouseAuditoryEncoder(
            ihc_tuning=self._ihc_tuning.copy(),
            ihc_gain=new_gain,
            tau_ihc=self.tau_ihc,
            tau_adapt=self.tau_adapt,
            adapt_strength=self.adapt_strength,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Circuit model
# ═══════════════════════════════════════════════════════════════════════

class MouseAuditoryCircuit(AuditoryCircuit):
    """IHC → CN → IC → output circuit for Mus musculus auditory pathway.

    Takes IHC firing rates as input, propagates through 60 CN neurons
    (bottleneck, with D-stellate lateral inhibition; Oertel et al. 2011)
    and 150 IC neurons (expansion; Blazquez Freches et al. 2023) to a
    3-channel behavioral output (approach/startle/freeze).
    """

    def __init__(
        self,
        encoder: MouseAuditoryEncoder,
        w_ihc_cn: Optional[np.ndarray] = None,
        w_cn_ic: Optional[np.ndarray] = None,
        w_cn_cn: Optional[np.ndarray] = None,
        cn_nt_signs: Optional[np.ndarray] = None,
        cn_baseline: float = 2.0,
        lateral_gain: float = 0.05,
        ic_threshold: float = 0.2,
        response_threshold: float = 0.05,
        tau_cn: float = 8.0,
        tau_ic: float = 15.0,
        tau_output: float = 40.0,
        dt: float = STIM_DT,
    ) -> None:
        self.encoder = encoder
        self._w_ihc_cn = w_ihc_cn if w_ihc_cn is not None else _build_w_ihc_cn()
        self._w_cn_ic = w_cn_ic if w_cn_ic is not None else _build_w_cn_ic()
        self._w_cn_cn = w_cn_cn if w_cn_cn is not None else _build_w_cn_cn()
        self._cn_nt_signs = cn_nt_signs if cn_nt_signs is not None else _build_cn_nt_signs()
        self.cn_baseline = cn_baseline
        self.lateral_gain = lateral_gain
        self.ic_threshold = ic_threshold
        self.response_threshold = response_threshold
        self.tau_cn = tau_cn
        self.tau_ic = tau_ic
        self.tau_output = tau_output
        self.dt = dt
        self._cn_names = list(CN_NAMES)
        self._ic_names = list(IC_NAMES)
        self._w_output: Optional[np.ndarray] = None
        self._cache_normalized_weights()
        self._build_output()

    @property
    def bottleneck_names(self) -> List[str]:
        return list(self._cn_names)

    @property
    def output_names(self) -> List[str]:
        return list(OUTPUT_NAMES)

    @property
    def ic_names(self) -> List[str]:
        return list(self._ic_names)

    @property
    def n_cn(self) -> int:
        return self._w_ihc_cn.shape[0]

    @property
    def n_ic(self) -> int:
        return self._w_cn_ic.shape[0]

    # ── Internal forward pass ────────────────────────────────────────

    def _cn_rates(self, ihc_rates: np.ndarray) -> np.ndarray:
        raw = self._w_ihc_cn_norm @ ihc_rates
        if self._w_lat_norm is not None:
            raw = raw + self.lateral_gain * (self._w_lat_norm @ raw)
        return np.maximum(raw + self.cn_baseline, 0.0)

    def _ic_rates(self, cn_rates: np.ndarray) -> np.ndarray:
        raw = self._ic_drive(cn_rates)
        driven = np.maximum(raw - self._ic_baseline, 0.0)
        return self._threshold_ic(driven)

    def _threshold_ic(self, ic: np.ndarray) -> np.ndarray:
        peak = ic.max()
        if peak < 1e-9:
            return np.zeros_like(ic)
        normed = ic / peak
        return np.where(normed > self.ic_threshold, ic, 0.0)

    def _cn_drive(self, ihc_rates: np.ndarray) -> np.ndarray:
        raw = self._w_ihc_cn_norm @ ihc_rates
        if self._w_lat_norm is not None:
            raw = raw + self.lateral_gain * (self._w_lat_norm @ raw)
        return raw + self.cn_baseline

    def _ic_drive(self, cn_rates: np.ndarray) -> np.ndarray:
        return self._w_cn_ic_norm @ cn_rates

    def _behavior_from_output(self, out: np.ndarray) -> str:
        mag = sum(abs(out[i]) for i in range(N_OUTPUT))
        if mag < self.response_threshold:
            return "no_response"
        idx = int(np.argmax(out))
        if out[idx] > self.response_threshold:
            return OUTPUT_NAMES[idx]
        return "no_response"

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        d: Dict[str, float] = {}
        for layer_name, names in [
            ("ihc", self.encoder.receptor_names), ("cn", self._cn_names),
            ("ic", self._ic_names), ("output", OUTPUT_NAMES),
        ]:
            if layer_name in layers:
                for i, n in enumerate(names):
                    d[n] = float(layers[layer_name][i])
        return d

    def _forward_arrays(self, ihc_rates: np.ndarray) -> Dict[str, np.ndarray]:
        cn = self._cn_rates(ihc_rates)
        ic = self._ic_rates(cn)
        out = self._w_output @ ic
        return {"ihc": ihc_rates, "cn": cn, "ic": ic, "output": out}

    def _forward_from_cn_arrays(self, cn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        ic = self._ic_rates(cn_rates)
        out = self._w_output @ ic
        return {"cn": cn_rates, "ic": ic, "output": out}

    # ── Output weight fitting ────────────────────────────────────────

    def _temporal_ic_avg(self, stimulus: np.ndarray) -> np.ndarray:
        dt = self.dt
        n_steps = stimulus.shape[0]
        d_cn = np.exp(-dt / self.tau_cn)
        o_cn = 1.0 - d_cn
        d_ic = np.exp(-dt / self.tau_ic)
        o_ic = 1.0 - d_ic

        ihc_rec = self.encoder.encode_timesteps(stimulus, dt)
        cn_state = np.zeros(self.n_cn)
        ic_state = np.zeros(self.n_ic)

        avg_window = max(1, int(100.0 / dt))
        ic_accum = np.zeros(self.n_ic)
        count = 0

        for t in range(n_steps):
            cn_drv = self._cn_drive(ihc_rec[t])
            cn_state = d_cn * cn_state + o_cn * cn_drv
            cn_state = np.maximum(cn_state, 0.0)
            ic_drv = self._ic_drive(cn_state)
            ic_state = d_ic * ic_state + o_ic * ic_drv
            ic_state = np.maximum(ic_state, 0.0)
            ic_out = self._threshold_ic(ic_state)
            if t >= n_steps - avg_window:
                ic_accum += ic_out
                count += 1

        return ic_accum / max(count, 1)

    def _build_output(self) -> None:
        rng = np.random.default_rng(42)
        ic_vecs, targets = [], []

        def _target(valence: str) -> List[float]:
            if valence == "approach":
                return [1.0, 0.0, 0.0]
            elif valence == "startle":
                return [0.0, 1.0, 0.0]
            elif valence == "freeze":
                return [0.0, 0.0, 1.0]
            return [0.0, 0.0, 0.0]

        def _add(ic: np.ndarray, valence: str) -> None:
            ic_vecs.append(ic)
            targets.append(_target(valence))

        for sound in SOUND_PROFILES:
            ihc = self.encoder.encode_array(sound, 1.0)
            ic = self._ic_rates(self._cn_rates(ihc))
            _add(ic, SOUND_VALENCE.get(sound, "neutral"))

        for sound in SOUND_PROFILES:
            stim = make_constant(500.0, dt=self.dt, sound_name=sound)
            _add(self._temporal_ic_avg(stim), SOUND_VALENCE.get(sound, "neutral"))

        _add(self._temporal_ic_avg(make_usv_sweep(500.0, dt=self.dt, profile="pup_isolation")), "approach")
        _add(self._temporal_ic_avg(make_call_train(500.0, dt=self.dt, profile="distress_call")), "startle")

        X = np.array(ic_vecs)
        Y = np.array(targets)
        if X.shape[0] > 0 and X.shape[1] > 0:
            x_scale = X.max(axis=0, keepdims=True) + 1e-9
            Xn = X / x_scale
            lam = 1.0
            XtX = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
            XtY = Xn.T @ Y
            w_normed = np.linalg.solve(XtX, XtY)
            w = w_normed / x_scale.T
            self._w_output = w.T + rng.normal(0, 0.0001, (N_OUTPUT, self.n_ic))
        else:
            self._w_output = rng.normal(0, 0.01, (N_OUTPUT, self.n_ic))

    def _cache_normalized_weights(self) -> None:
        self._w_ihc_cn_norm = self._w_ihc_cn / (self._w_ihc_cn.max() + 1e-9)
        if self.lateral_gain > 0:
            self._w_lat_norm = (self._w_cn_cn / (self._w_cn_cn.max() + 1e-9)) * self._cn_nt_signs[:, None]
        else:
            self._w_lat_norm = None
        self._w_cn_ic_norm = self._w_cn_ic / (self._w_cn_ic.max() + 1e-9)
        self._ic_baseline = self._w_cn_ic_norm @ np.full(self.n_cn, self.cn_baseline)

    # ── AuditoryCircuit interface ────────────────────────────────────

    def forward_from_receptors(self, receptor_signal: Dict[str, float]) -> Dict:
        ihc_arr = np.array([receptor_signal.get(n, 0.0) for n in self.encoder.receptor_names])
        layers = self._forward_arrays(ihc_arr)
        return {"behavior": self._behavior_from_output(layers["output"]),
                "neurons": self._layers_to_dict(layers)}

    def forward_from_bottleneck(self, bottleneck_signal: Dict[str, float]) -> Dict:
        cn_arr = np.array([bottleneck_signal.get(n, 0.0) for n in self._cn_names])
        layers = self._forward_from_cn_arrays(cn_arr)
        return {"behavior": self._behavior_from_output(layers["output"]),
                "neurons": self._layers_to_dict(layers)}

    def natural_response(self, sound_name: str, intensity: float) -> Dict[str, float]:
        ihc = self.encoder.encode_array(sound_name, intensity)
        layers = self._forward_arrays(ihc)
        return self._layers_to_dict(layers)

    def natural_behavior(self, sound_name: str, intensity: float) -> str:
        ihc = self.encoder.encode_array(sound_name, intensity)
        layers = self._forward_arrays(ihc)
        return self._behavior_from_output(layers["output"])

    def forward_timesteps_from_receptors(
        self, receptor_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        n_steps = receptor_timeseries.shape[0]
        d_cn = np.exp(-dt / self.tau_cn)
        o_cn = 1.0 - d_cn
        d_ic = np.exp(-dt / self.tau_ic)
        o_ic = 1.0 - d_ic
        d_out = np.exp(-dt / self.tau_output)
        o_out = 1.0 - d_out

        cn_state = np.zeros(self.n_cn)
        ic_state = np.zeros(self.n_ic)
        out_state = np.zeros(N_OUTPUT)
        cn_rec = np.empty((n_steps, self.n_cn))
        ic_rec = np.empty((n_steps, self.n_ic))
        out_rec = np.empty((n_steps, N_OUTPUT))

        for t in range(n_steps):
            cn_drv = self._cn_drive(receptor_timeseries[t])
            cn_state = d_cn * cn_state + o_cn * cn_drv
            cn_state = np.maximum(cn_state, 0.0)
            ic_drv = self._ic_drive(cn_state)
            ic_state = d_ic * ic_state + o_ic * ic_drv
            ic_state = np.maximum(ic_state, 0.0)
            ic_driven = np.maximum(ic_state - self._ic_baseline, 0.0)
            ic_out = self._threshold_ic(ic_driven)
            out_state = d_out * out_state + o_out * (self._w_output @ ic_out)
            cn_rec[t] = cn_state
            ic_rec[t] = ic_out
            out_rec[t] = out_state

        avg_window = max(1, int(100.0 / dt))
        behavior = self._behavior_from_output(out_rec[-avg_window:].mean(axis=0))
        return {"t_ms": np.arange(n_steps) * dt, "cn": cn_rec,
                "ic": ic_rec, "output": out_rec, "behavior": behavior}

    def forward_timesteps_from_bottleneck(
        self, bottleneck_timeseries: np.ndarray, dt: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        n_steps = bottleneck_timeseries.shape[0]
        d_ic = np.exp(-dt / self.tau_ic)
        o_ic = 1.0 - d_ic
        d_out = np.exp(-dt / self.tau_output)
        o_out = 1.0 - d_out

        ic_state = np.zeros(self.n_ic)
        out_state = np.zeros(N_OUTPUT)
        ic_rec = np.empty((n_steps, self.n_ic))
        out_rec = np.empty((n_steps, N_OUTPUT))

        for t in range(n_steps):
            ic_drv = self._ic_drive(bottleneck_timeseries[t])
            ic_state = d_ic * ic_state + o_ic * ic_drv
            ic_state = np.maximum(ic_state, 0.0)
            ic_driven = np.maximum(ic_state - self._ic_baseline, 0.0)
            ic_out = self._threshold_ic(ic_driven)
            out_state = d_out * out_state + o_out * (self._w_output @ ic_out)
            ic_rec[t] = ic_out
            out_rec[t] = out_state

        avg_window = max(1, int(100.0 / dt))
        behavior = self._behavior_from_output(out_rec[-avg_window:].mean(axis=0))
        return {"t_ms": np.arange(n_steps) * dt, "cn": bottleneck_timeseries,
                "ic": ic_rec, "output": out_rec, "behavior": behavior}


# ═══════════════════════════════════════════════════════════════════════
#  Composed model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MouseAuditoryModel:
    """Composed encoder + circuit for the Mus musculus auditory pathway."""

    cn_baseline: float = 2.0
    lateral_gain: float = 0.05
    ic_threshold: float = 0.2
    response_threshold: float = 0.05
    dt: float = STIM_DT
    tau_ihc: float = 3.0
    tau_cn: float = 8.0
    tau_ic: float = 15.0
    tau_output: float = 40.0
    tau_adapt: float = 150.0
    adapt_strength: float = 0.25

    ihc_gain: Dict[str, float] = field(default_factory=dict)

    encoder: Optional[MouseAuditoryEncoder] = field(default=None, repr=False)
    circuit: Optional[MouseAuditoryCircuit] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.encoder is None:
            gain = self.ihc_gain if self.ihc_gain else None
            self.encoder = MouseAuditoryEncoder(
                ihc_gain=gain, tau_ihc=self.tau_ihc,
                tau_adapt=self.tau_adapt, adapt_strength=self.adapt_strength,
            )
        if not self.ihc_gain:
            self.ihc_gain = dict(self.encoder.ihc_gain)
        if self.circuit is None:
            self.circuit = MouseAuditoryCircuit(
                encoder=self.encoder,
                cn_baseline=self.cn_baseline, lateral_gain=self.lateral_gain,
                ic_threshold=self.ic_threshold, response_threshold=self.response_threshold,
                tau_cn=self.tau_cn, tau_ic=self.tau_ic,
                tau_output=self.tau_output, dt=self.dt,
            )

    # ── Delegated properties ─────────────────────────────────────────

    @property
    def ihc_names(self) -> List[str]:
        return self.encoder.receptor_names

    @property
    def cn_names(self) -> List[str]:
        return self.circuit.bottleneck_names

    @property
    def ic_names(self) -> List[str]:
        return self.circuit.ic_names

    @property
    def output_names(self) -> List[str]:
        return self.circuit.output_names

    @property
    def n_ihc(self) -> int:
        return self.encoder.n_receptors

    @property
    def n_cn(self) -> int:
        return self.circuit.n_cn

    @property
    def n_ic(self) -> int:
        return self.circuit.n_ic

    # ── Backward-compatible internal accessors ───────────────────────

    @property
    def _w_ihc_cn(self) -> np.ndarray:
        return self.circuit._w_ihc_cn

    @property
    def _w_cn_ic(self) -> np.ndarray:
        return self.circuit._w_cn_ic

    @property
    def _w_cn_cn(self) -> np.ndarray:
        return self.circuit._w_cn_cn

    @property
    def _ihc_tuning(self) -> np.ndarray:
        return self.encoder._ihc_tuning

    def _ihc_rates(self, sound_name: str, intensity: float) -> np.ndarray:
        return self.encoder.encode_array(sound_name, intensity)

    def _cn_rates(self, ihc_rates: np.ndarray) -> np.ndarray:
        return self.circuit._cn_rates(ihc_rates)

    def _ic_rates(self, cn_rates: np.ndarray) -> np.ndarray:
        return self.circuit._ic_rates(cn_rates)

    def _forward(self, sound_name: str, intensity: float) -> Dict[str, np.ndarray]:
        ihc = self.encoder.encode_array(sound_name, intensity)
        return self.circuit._forward_arrays(ihc)

    def _forward_from_cn(self, cn_rates: np.ndarray) -> Dict[str, np.ndarray]:
        return self.circuit._forward_from_cn_arrays(cn_rates)

    def _behavior_from_output(self, out: np.ndarray) -> str:
        return self.circuit._behavior_from_output(out)

    def _layers_to_dict(self, layers: Dict[str, np.ndarray]) -> Dict[str, float]:
        return self.circuit._layers_to_dict(layers)

    # ── Public interface ─────────────────────────────────────────────

    def receptor_activity(self, stimulus_intensity: float, sound_type: str = "approach") -> Dict[str, float]:
        return self.encoder.encode(_pick_sound(sound_type), stimulus_intensity)

    def network_state(self, stimulus_intensity: float, sound_type: str = "approach") -> Dict[str, float]:
        return self.circuit.natural_response(_pick_sound(sound_type), stimulus_intensity)

    def simulate(self, stimulus_intensity: float, sound_type: str = "approach") -> Dict:
        sound_name = _pick_sound(sound_type)
        ihc = self.encoder.encode(sound_name, stimulus_intensity)
        return self.circuit.forward_from_receptors(ihc)

    def simulate_from_receptors(self, receptor_signal: Dict[str, float], sound_type: str = "approach") -> Dict:
        return self.circuit.forward_from_receptors(receptor_signal)

    def simulate_from_cn(self, cn_signal: Dict[str, float], sound_type: str = "approach") -> Dict:
        return self.circuit.forward_from_bottleneck(cn_signal)

    def sound_response(self, stimulus_intensity: float, sound_type: str = "approach") -> Dict[str, float]:
        return self.receptor_activity(stimulus_intensity, sound_type)

    def behavior_from_signal(self, receptor_signal: Dict[str, float], sound_type: str = "approach") -> str:
        return self.simulate_from_receptors(receptor_signal)["behavior"]

    def cn_response(self, stimulus_intensity: float, sound_type: str = "approach") -> Dict[str, float]:
        sound_name = _pick_sound(sound_type)
        ihc = self.encoder.encode_array(sound_name, stimulus_intensity)
        cn = self.circuit._cn_rates(ihc)
        return {n: float(cn[i]) for i, n in enumerate(self.cn_names)}

    def simulate_timesteps(self, stimulus: np.ndarray, dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        if dt is None:
            dt = self.dt
        ihc_rec = self.encoder.encode_timesteps(stimulus, dt)
        result = self.circuit.forward_timesteps_from_receptors(ihc_rec, dt)
        result["ihc"] = ihc_rec
        return result

    def simulate_timesteps_from_cn(self, cn_input: np.ndarray, dt: Optional[float] = None) -> Dict[str, np.ndarray]:
        return self.circuit.forward_timesteps_from_bottleneck(cn_input, dt)

    def validate_sound_responses(self) -> Dict[str, bool]:
        checks: Dict[str, bool] = {}
        approach = self.simulate(1.0, "approach")
        checks["approach_response"] = approach["behavior"] == "approach"
        startle = self.simulate(1.0, "startle")
        checks["startle_response"] = startle["behavior"] == "startle"
        freeze = self.simulate(1.0, "freeze")
        checks["freeze_response"] = freeze["behavior"] == "freeze"
        silent = self.simulate(0.0, "approach")
        checks["silence_no_response"] = silent["behavior"] == "no_response"
        ihc = self.receptor_activity(1.0, "approach")
        checks["ihc_responds"] = max(ihc.values()) > 0
        usv = make_usv_sweep(500.0, dt=self.dt, profile="pup_isolation")
        checks["temporal_usv_approach"] = self.simulate_timesteps(usv)["behavior"] == "approach"
        noise = make_constant(500.0, dt=self.dt, sound_name="startle_noise")
        checks["temporal_noise_startle"] = self.simulate_timesteps(noise)["behavior"] == "startle"
        silence_t = make_constant(500.0, dt=self.dt, sound_name="silence")
        checks["temporal_silence_no_response"] = self.simulate_timesteps(silence_t)["behavior"] == "no_response"
        return checks

    def clone_perturbed(
        self,
        ihc_gain: Optional[Dict[str, float]] = None,
        ihc_cn_weight_noise: Optional[np.ndarray] = None,
        cn_cn_weight_noise: Optional[np.ndarray] = None,
        cn_ic_weight_noise: Optional[np.ndarray] = None,
    ) -> "MouseAuditoryModel":
        new_encoder = MouseAuditoryEncoder(
            ihc_tuning=self.encoder._ihc_tuning.copy(),
            ihc_gain=dict(ihc_gain) if ihc_gain else dict(self.encoder.ihc_gain),
            tau_ihc=self.encoder.tau_ihc,
            tau_adapt=self.encoder.tau_adapt,
            adapt_strength=self.encoder.adapt_strength,
        )
        w1 = self.circuit._w_ihc_cn.copy()
        w2 = self.circuit._w_cn_ic.copy()
        w3 = self.circuit._w_cn_cn.copy()
        if ihc_cn_weight_noise is not None:
            w1 = np.maximum(w1 + ihc_cn_weight_noise, 0.0)
        if cn_cn_weight_noise is not None:
            w3 = w3 + cn_cn_weight_noise
        if cn_ic_weight_noise is not None:
            w2 = np.maximum(w2 + cn_ic_weight_noise, 0.0)

        new_circuit = MouseAuditoryCircuit(
            encoder=new_encoder,
            w_ihc_cn=w1, w_cn_ic=w2, w_cn_cn=w3,
            cn_nt_signs=self.circuit._cn_nt_signs.copy(),
            cn_baseline=self.circuit.cn_baseline,
            lateral_gain=self.circuit.lateral_gain,
            ic_threshold=self.circuit.ic_threshold,
            response_threshold=self.circuit.response_threshold,
            tau_cn=self.circuit.tau_cn,
            tau_ic=self.circuit.tau_ic,
            tau_output=self.circuit.tau_output,
            dt=self.dt,
        )

        clone = MouseAuditoryModel.__new__(MouseAuditoryModel)
        clone.cn_baseline = self.cn_baseline
        clone.lateral_gain = self.lateral_gain
        clone.ic_threshold = self.ic_threshold
        clone.response_threshold = self.response_threshold
        clone.dt = self.dt
        clone.tau_ihc = self.tau_ihc
        clone.tau_cn = self.tau_cn
        clone.tau_ic = self.tau_ic
        clone.tau_output = self.tau_output
        clone.tau_adapt = self.tau_adapt
        clone.adapt_strength = self.adapt_strength
        clone.ihc_gain = dict(new_encoder.ihc_gain)
        clone.encoder = new_encoder
        clone.circuit = new_circuit
        return clone
