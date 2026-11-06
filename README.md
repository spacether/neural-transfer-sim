# Neural Transfer Simulation

Connectome-based simulations that derive the hardware and software requirements for transferring sensory signals from one brain to another. Runs entirely in simulation — no animals, no hardware, no lab.

See [CLAUDE.md](CLAUDE.md) for the full project brief and design rationale.

## Papers

- [Olfactory paper draft](papers/olfactory_paper_draft.md) — cross-organism smell-transfer results and scaling analysis

## Subject areas

### Olfactory (smell)

The first and most complete subject area. Four organisms of increasing complexity, each using the shared transfer pipeline.

| Stage | Organism | Receptor types | Bottleneck | Signal mode | Status |
|-------|----------|---------------|------------|-------------|--------|
| 1 | *C. elegans* (worm) | 8 | none | graded | Complete |
| 2 | *Drosophila* larva | 21 ORN types | 21 glomeruli | spiking | Complete |
| 3 | Adult *Drosophila* | 50 ORN types | 50 glomeruli → 2000 KCs | spiking | Complete |
| 4 | Zebrafish | 100 OR types | 100 glomeruli → 80 mitral → 200 granule (recurrent) → 500 Dp | spiking | Complete |

**Experiments per organism:**

- **Broadcast** — direct receptor-to-receptor copy with no calibration, across brain-difference levels
- **Calibration** — ridge regression transform vs. direct broadcast on held-out odors
- **Channel sweep** — transfer quality vs. number of connected ORNs at each brain-difference level
- **Hardware boundary** — sweeps of read noise, write precision, channel dropout, and transmit delay, annotated with real technology operating points
- **PN/mitral channel sweep** (larva, adult, zebrafish) — 2D sweep of read and write projection neuron counts with calibration

**Cross-organism scaling:**

- **Scaling comparison** — channels needed, calibration cost, and broadcast quality vs. organism complexity
- **Calibration efficiency** — how odor identification accuracy grows with training set size per organism

**Cross-cutting analyses:**

- **Calibration comparison** — ridge (N²) vs. diagonal (2N) calibration across organisms
- **Connectome vs. null** — real wiring topology vs. shuffled-connectome controls
- **Sensitivity analysis** — free-parameter robustness for connectome models
- **Rerun connectome** — full connectome rerun with bootstrap CIs

**Key findings:**

- **Bottleneck architecture determines transfer requirements more than brain size.** Channel sweep curves do not collapse when normalized by bottleneck fraction — at 50% of channels connected, larva achieves 67% odor identification, zebrafish 59%, adult fly 46%.
- **Linear calibration (ridge regression) is very powerful.** It achieves near-perfect correlation and behavioral match across all organisms, which means it masks architectural differences. Calibration sample efficiency scales with dimensionality (N² parameters), making it primarily a matrix-math result rather than a biological one.
- **Hardware boundary is consistent across organisms.** Read noise and channel dropout are the binding constraints; write precision is forgiving above 4 bits; transmit delay matters only at hundreds of milliseconds.
- **The zebrafish's granule cell recurrent lateral inhibition** is the key vertebrate architectural difference from insects — it decorrelates mitral cell representations before they reach the pallium (Dp).

### Auditory (hearing)

Adult *Drosophila* auditory transfer using real FlyWire v783 connectome data (Dorkenwald et al. 2024; Schlegel et al. 2024).

| Organism | Receptor neurons | Bottleneck | Signal mode | Status |
|----------|-----------------|------------|-------------|--------|
| Adult *Drosophila* | 1103 JO neurons (359 A+B sound-sensitive) | 150 AMMC neurons → 324 WED/VLP | temporal (exact exponential integration) | Experiments running |

**Pathway:** JO neurons (mechanosensory, frequency-tuned) → 150 AMMC → 324 WED/VLP → 2 output channels (courtship response / startle-escape).

**Model architecture:** Split into `DrosophilaAuditoryEncoder` (peripheral) + `DrosophilaAuditoryCircuit` (central), implementing organism-agnostic `AuditoryEncoder`/`AuditoryCircuit` base classes. A human or mouse model replaces the encoder.

**Subtype filter:** Configurable — default to A+B sound-sensitive neurons (359), with option to include all 1103 (adds gravity/wind subtypes C–F). Both configurations pass all validation checks.

**Stimulus set:** 10 sound profiles spanning courtship songs (pulse, sine, combined), threats (wing threat, predator buzz), tones, and controls (silence, white noise, wind gust). Temporal generators: pulse trains (IPI=35ms, *D. mel* conspecific), sine song, constant.

**Temporal dynamics:** Exact exponential integration (unconditionally stable), sensory adaptation with separate adaptation variable, combined steady-state + temporal output weight fitting.

**Experiments:** Broadcast, calibration, JO channel sweep, hardware boundary, AMMC channel sweep (2D read/write). Same pipeline as olfactory, comparing at WED/VLP layer.

### Key metrics

All experiments report:
- **Behavioral match** — does the receiver produce the sender's behavioral response?
- **Correlation** (Pearson r) — how well transferred activity tracks natural response
- **MSE** — reconstruction error
- **RSA** — representational similarity (geometry preservation)
- **Odor identification** / **Sound identification** — N-way nearest-centroid classification

## Repository structure

```
neural_transfer/     importable package
  models/            per-organism brain models and individual-variation generators
    c_elegans/         hand-built + Cook2019 connectome graded worm models
    drosophila_larva/  statistical + connectome larval models, .npz data
    drosophila_adult/  olfactory + auditory connectome models, .npz data
    zebrafish/         olfactory bulb model with recurrent lateral inhibition
    null_model.py      shuffled-connectome controls for topology comparison
  hardware/          read, write, noise, and compression models
  link/              translate, encode, transmit (calibration + broadcast)
  analysis/          metrics, plotting, results I/O

experiments/         experiment modules organized by sensory modality
  olfactory/           smell-transfer experiments
    c_elegans/           broadcast, calibration, hardware boundary
    drosophila_larva/    broadcast, calibration, channel sweep, hardware boundary, PN channel sweep
    drosophila_adult/    broadcast, calibration, channel sweep, hardware boundary, PN channel sweep
    zebrafish/           broadcast, calibration, channel sweep, hardware boundary, mitral channel sweep
    scaling/             cross-organism scaling comparison + calibration efficiency
    calibration_comparison/  ridge vs diagonal calibration
    connectome_vs_null/      real connectome vs shuffled-null topology
    sensitivity_analysis/    free-parameter sensitivity
    rerun_connectome/        full connectome rerun with bootstrap CIs

papers/              write-ups and manuscripts
tests/               automated tests (pytest)
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:
```bash
python -m pytest -q
```

Run a single experiment (e.g., larva broadcast):
```bash
python -m experiments.olfactory.drosophila_larva.broadcast.run_default
```

Run all experiments for one organism:
```bash
for exp in broadcast calibration channel_sweep hardware_boundary pn_channel_sweep; do
  python -m experiments.olfactory.drosophila_larva.${exp}.run_default
done
```

Run the cross-organism scaling comparison:
```bash
python -m experiments.olfactory.scaling.run_default
python -m experiments.olfactory.scaling.run_calibration_efficiency
```

## Known limitations

See [CLAUDE.md § Known limitations](CLAUDE.md) for the full list. In brief: results are lower bounds, synaptic weights are unknown, calibration is unrealistically powerful (masks real architectural effects), and no claim is made about human transfer.
