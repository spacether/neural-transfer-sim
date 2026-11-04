# Simulated Brain-to-Brain Smell Transfer

Connectome-based simulations that derive the hardware and software requirements for transferring a smell signal from one brain to another. Runs entirely in simulation — no animals, no hardware, no lab.

See [CLAUDE.md](CLAUDE.md) for the full project brief and design rationale.

## Status

All four organisms are implemented and all experiments run end to end. Results are saved as JSON and plotted as PNG in each experiment's `results/` directory.

| Stage | Organism | Receptor types | Bottleneck | Signal mode | Status |
|-------|----------|---------------|------------|-------------|--------|
| 1 | *C. elegans* (worm) | 8 | none | graded | Complete |
| 2 | *Drosophila* larva | 21 ORN types | 21 glomeruli | spiking | Complete |
| 3 | Adult *Drosophila* | 50 ORN types | 50 glomeruli → 2000 KCs | spiking | Complete |
| 4 | Zebrafish | 100 OR types | 100 glomeruli → 80 mitral → 200 granule (recurrent) → 500 Dp | spiking | Complete |

### Experiments per organism

Each organism runs the same set of experiments through the shared pipeline:

- **Broadcast** — direct receptor-to-receptor copy with no calibration, across brain-difference levels
- **Calibration** — ridge regression transform vs. direct broadcast on held-out odors
- **Channel sweep** — transfer quality vs. number of connected ORNs at each brain-difference level
- **Hardware boundary** — sweeps of read noise, write precision, channel dropout, and transmit delay, annotated with real technology operating points
- **PN/mitral channel sweep** (larva, adult, zebrafish) — 2D sweep of read and write projection neuron counts with calibration

### Cross-organism scaling

- **Scaling comparison** — channels needed, calibration cost, and broadcast quality vs. organism complexity
- **Calibration efficiency** — how odor identification accuracy grows with training set size per organism

### Key metrics

All experiments report:
- **Behavioral match** — does the receiver produce the sender's behavioral response?
- **Correlation** (Pearson r) — how well transferred activity tracks natural response
- **MSE** — reconstruction error
- **RSA** — representational similarity (geometry preservation)
- **Odor identification** — N-way nearest-centroid classification (which specific odor, not just attractive/repellent)

## Repository structure

```
models/          per-organism brain models and individual-variation generators
  worm_connectome.py       hand-built graded worm (AWA, AWC, AWB, ADL, ASH pathway)
  cook2019_connectome.py   Cook et al. 2019 connectome via cect with CeNGEN signs
  worm_variants.py         worm individual-difference generator
  drosophila_larva.py      21-ORN → 21-PN → 72-KC → MBON larval model
  larva_variants.py        larva variant generator
  drosophila_adult.py      50-ORN → 50-PN → 2000-KC → MBON adult fly model
  adult_variants.py        adult variant generator
  zebrafish.py             100-ORN → 100-glomeruli → 80-mitral → 200-granule → 500-Dp model
  zebrafish_variants.py    zebrafish variant generator

hardware/        read, write, noise, and compression models
  noise.py                 ReadNoise, WritePrecision, ChannelDropout, TransmitDelay, HardwareChain
  technologies.py          real technology operating points for plot annotations

link/            translate, encode, transmit
  receptor_to_receptor.py  direct broadcast baseline (no translation)
  calibration.py           ridge regression calibration transform

analysis/        metrics, plotting, results I/O
  metrics.py               correlation, MSE, RSA, odor identification
  plotting.py              all experiment plot functions
  results_writer.py        JSON result serialization

experiments/     per-organism experiment directories, each with experiment.py + run_default.py
  c_elegans/               broadcast, calibration, hardware boundary
  drosophila_larva/        broadcast, calibration, channel sweep, hardware boundary, PN channel sweep
  drosophila_adult/        broadcast, calibration, channel sweep, hardware boundary, PN channel sweep
  zebrafish/               broadcast, calibration, channel sweep, hardware boundary, mitral channel sweep
  scaling/                 cross-organism scaling comparison + calibration efficiency

validation/      comparisons against published data
tests/           automated tests (pytest)
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # or: pip install numpy scipy scikit-learn matplotlib cect
```

Run tests:
```bash
python -m pytest -q
```

Run a single experiment (e.g., larva broadcast):
```bash
python -m experiments.drosophila_larva.broadcast.run_default
```

Run all experiments for one organism:
```bash
for exp in broadcast calibration channel_sweep hardware_boundary pn_channel_sweep; do
  python -m experiments.drosophila_larva.${exp}.run_default
done
```

Run the cross-organism scaling comparison:
```bash
python -m experiments.scaling.run_default
python -m experiments.scaling.run_calibration_efficiency
```

## Key findings so far

- **Bottleneck architecture determines transfer requirements more than brain size.** Channel sweep curves do not collapse when normalized by bottleneck fraction — at 50% of channels connected, larva achieves 67% odor identification, zebrafish 59%, adult fly 46%.
- **Linear calibration (ridge regression) is very powerful.** It achieves near-perfect correlation and behavioral match across all organisms, which means it masks architectural differences. Calibration sample efficiency scales with dimensionality (N² parameters), making it primarily a matrix-math result rather than a biological one.
- **Hardware boundary is consistent across organisms.** Read noise and channel dropout are the binding constraints; write precision is forgiving above 4 bits; transmit delay matters only at hundreds of milliseconds.
- **The zebrafish's granule cell recurrent lateral inhibition** is the key vertebrate architectural difference from insects — it decorrelates mitral cell representations before they reach the pallium (Dp).

## Known limitations

See [CLAUDE.md § Known limitations](CLAUDE.md) for the full list. In brief: results are lower bounds, synaptic weights are unknown, calibration is unrealistically powerful (masks real architectural effects), and no claim is made about human transfer.
