# olfactory-transfer-sim

Simulating what it takes to transfer a smell signal from one brain to another.

This repository follows the project brief in [CLAUDE.md](CLAUDE.md): a staged simulation pipeline for worm, larval fly, and adult fly olfactory transfer, with a shared architecture for read, reduce, transmit, translate, encode, write, and score stages.

## Project goal

The long-term goal is to understand how much connection quality, channel count, and calibration are needed to transfer an odor representation from a sender brain to a receiver brain in simulation. The work is intentionally scoped to smell and follows a three-stage progression:

1. Worm as the end-to-end pipeline test
2. Larval fly with a real olfactory bottleneck
3. Adult fly with convergence and expansion in the olfactory pathway

## Repository structure

The workspace is organized around the architecture described in the project brief.

### models/

Contains organism-specific brain models and the simulation primitives that define each nervous system. This is the layer where the sender and receiver brains are represented, and where model variants such as individual differences and sensitivity adjustments are introduced.

Current status:
- [models/worm.py](models/worm.py): compact graded worm model used for the first milestone

### hardware/

Stores read/write hardware models: recording devices, stimulation devices, noise models, and any signal-conditioning elements that sit between the brain and the communication channel. This module is where the project distinguishes between graded and spiking signals.

This directory is intentionally reserved for hardware abstractions rather than specific experiments.

### link/

Contains the communication path itself: direct broadcast, learned translation, encoding, and transmission logic. This is the place where the project compares a simple copy baseline to more sophisticated mapping and channel-based transfers.

Current status:
- [link/direct_broadcast.py](link/direct_broadcast.py): direct sender-to-receiver copy baseline with no translation

### experiments/

Defines runnable experiments and sweep definitions. This is the top-level execution layer that runs a single trial, a benchmark, or a search over parameters.

Current status:
- [experiments/worm_milestone.py](experiments/worm_milestone.py): first worm milestone validation and single-odor trial driver

### validation/

Holds checks that compare model output against expected biological behavior and published data. Validation is used before transfer experiments so the sender and receiver models match the organism-specific signals and expected responses.

Current status:
- [validation/odor_validation.py](validation/odor_validation.py): validates that a worm receives an attractive odor and produces the expected forward movement pattern

### analysis/

Contains metrics, summaries, and plotting logic used after a run. This is where transfer quality, noise sensitivity, calibration performance, and downstream behavioral comparisons are aggregated and visualized.

### tests/

Stores automated tests that lock in the baseline behavior for the current milestone before moving on to more advanced transfer experiments.

Current status:
- [tests/test_direct_broadcast.py](tests/test_direct_broadcast.py): verifies direct broadcast preserves sender activity and produces the expected worm reaction

## Current milestone

This repository is currently at the first milestone described in [CLAUDE.md](CLAUDE.md):

- one worm
- one attractive odor
- one expected behavior

The active baseline is a simple graded worm model that responds to an attractive odor by driving forward motion, and the direct broadcast link is implemented as the no-translation comparison case.

## Getting started

Create the project virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest numpy scipy scikit-learn matplotlib Brian2 optuna
```

Then run the validation checks:

```bash
python -m pytest -q
```

## Next development direction

The immediate next work is to expand the worm model toward the direct-broadcast baseline described in the brief, then add calibration and a learned translation layer before moving to the larval fly stage.
