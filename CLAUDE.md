# Simulated Brain-to-Brain Smell Transfer

This file is the project brief. Read it before writing code.

## Purpose and scope

This project uses connectome-based simulations to derive the hardware and software requirements for transferring a smell signal from one brain to another. It runs in simulation only: no animals, no hardware, no lab.

The scope is deliberately narrow.

- **One sense: smell.** No touch, hearing, or vision in this project.
- **Four organisms, in order.** Worm (*C. elegans*) first, then *Drosophila* larva, then adult fly, then zebrafish. Each stage must work before the next begins.
- **One question per stage.** How much connection, of what quality, and at which point in the pathway, is needed to make a receiver brain register the odor a sender brain is detecting?

Stage 1 (worm) is the pipeline test. Its job is to prove the whole chain runs end to end on the simplest available nervous system. Stages 2–4 add real olfactory bottlenecks and test how requirements change as the bottleneck grows and the architecture changes (insect vs. vertebrate).

Out of scope: any claim about human brain-to-brain communication beyond what the scaling trend across four organisms supports.

## Why these three organisms

| Organism | Receptor neurons | Glomeruli | Why included |
| --- | --- | --- | --- |
| *C. elegans* | ~10 | none | Simplest possible pipeline test; no bottleneck |
| *Drosophila* larva | ~21 | ~21 | Minimal clean bottleneck, one neuron per receptor |
| Adult *Drosophila* | ~2,600 per antenna | ~50 per side | Full bottleneck with convergence and expansion |
| Zebrafish | ~300 (100 types × 3) | ~100 | Vertebrate architecture with recurrent lateral inhibition |

## Shared pipeline architecture

One pipeline serves all three stages. Only the brain models and the choice of read/write sites change between them, so every module is written once and reused.

1. **Sender brain.** A simulated nervous system receiving an odor as input to its receptor neurons.
2. **Read.** A model of a recording device at a chosen site: which neurons it senses, how strongly as a function of distance, and what noise it adds.
3. **Reduce.** Compression of the recorded signal — spike counts in time bins for spiking neurons, downsampled voltage or band power for graded ones.
4. **Transmit.** The link itself, with adjustable delay, bandwidth ceiling, and dropout.
5. **Translate.** The learned mapping from the sender's code to the receiver's, fitted during calibration. Must support a null setting (direct broadcast, no translation) as the baseline.
6. **Encode.** Conversion from target receiver activity into a stimulation pattern, accounting for the fact that one stimulation channel drives many neurons.
7. **Write.** A model of the stimulating device at a chosen site: current or light spread, threshold, and safety ceiling.
8. **Receiver brain.** A second simulated nervous system, deliberately different from the sender's.
9. **Score.** Comparison of receiver activity and behavior against what the sender experienced.

Two design rules, built in from the start:

- **Read site and write site are chosen independently.** Reading works best late in the pathway, where the signal reflects what the sender actually detected. Writing works best early, at a bottleneck, where few channels carry everything and the receiver's own brain expands the signal. The pipeline must allow any read site paired with any write site.
- **Two signal modes.** Worm olfactory neurons are largely graded and non-spiking; fly olfactory receptor neurons spike. Spike detection and threshold counting do not apply to graded neurons, which need continuous readout and level-setting stimulation. Both modes must exist in the read, reduce, and write modules.

## Stage 1: Worm (*C. elegans*)

Stage 1 proves the pipeline runs end to end. Nothing moves to Stage 2 until a sender worm's odor detection reliably produces the matching response in a receiver worm that was never exposed to the odor.

**Circuit.** Use the whole 302-neuron connectome — at this size there is no reason to extract a subnetwork.

| Role | Neurons | Note |
| --- | --- | --- |
| Attractive odor sensing | AWA, AWC (L/R) | Primary scope for Stage 1 |
| Repellent sensing | AWB, ADL | Secondary |
| Multimodal / escape | ASH | Excluded from Stage 1 — responds to chemicals, nose touch and osmolarity through one neuron |
| Integration | AIY, AIB, AIA, AIZ | Decide continue, turn, or reverse |
| Output | AVA (backward), AVB (forward) | Behavioral readout |

**Model choice.** Graded, non-spiking neurons. Most worm neurons signal by continuous voltage change rather than action potentials, and they lack voltage-gated sodium channels entirely. Do not use a spiking configuration for convenience — it is biologically wrong and would invalidate the read and write models.

**Known limitation to document, not solve.** Each worm olfactory neuron carries many receptor types, so a single neuron's response to an odor is a blend that cannot be decomposed. The worm has no odor-specific bottleneck. Its olfactory system is closer to a low-dimensional "better or worse" channel than an identification system.

**Making two worms different.** Both worms share the canonical connectome, then diverge along a single tunable axis so results can be reported against degree of difference. In order of increasing severity:

1. Weak connections dropped, added, or resized, matching the individual variability found across eight fully mapped worms.
2. All synaptic weights jittered by a random factor.
3. Per-neuron sensitivity and time constants varied.
4. Whole-circuit neuromodulatory state changed — a hungry worm versus a fed one.
5. One worm habituated to prior odor exposure.

Every variant must pass a sanity check before use: it still reverses when touched and still moves toward food. A variant that fails is broken, not an individual.

**Experiments.**

- [x] Single worm detects an attractive odor and produces the correct behavioral response
- [x] Direct broadcast baseline: copy sender activity to receiver with no translation, measure transfer
- [x] Calibration: present shared odors to both worms, fit the sender-to-receiver transform, test on held-out odors
- [x] Hardware boundary: where noise, precision, dropout, and delay break the link, annotated with real technologies

**Success criterion.** Define before running: the receiver produces the sender's behavioral response on a stated fraction of trials, within a stated time window, on odors not used in calibration.

## Stage 2: *Drosophila* larva

Stage 2 introduces a real bottleneck: roughly 21 olfactory receptor neurons, approximately one per receptor type, each feeding its own glomerulus, with the full pathway through to the learning centre mapped.

**What changes from Stage 1.**

- **Spiking neurons.** Fly receptor neurons fire action potentials, so the spiking read and reduce modes come into use for the first time.
- **A true write target.** The glomeruli are a discrete, identifiable bottleneck — unlike the worm, there is somewhere specific to aim.
- **Clean input data.** Because the larva has about one neuron per receptor type, the published empty-neuron recordings characterise essentially its entire olfactory input layer.
- **Subnetwork extraction begins.** The larval connectome is roughly 3,000 neurons. Pull the olfactory pathway, but replace removed inputs with a tuned background drive, and validate the subnetwork against the full model on a handful of cases before running sweeps.

**New questions this stage answers.**

- Does writing at the glomerular bottleneck outperform writing at the receptor neurons or downstream, as predicted?
- How does the required channel count change when the sense gains odor-specific channels rather than a blended good/bad signal?
- Does the calibration transform generalise better or worse when each channel carries one receptor's output?

**Experiments deferred from Stage 1** (the worm lacks distinct channels and a bottleneck, making these questions meaningful only here):

- [x] Channel sweep: transfer quality versus number of connected neurons, at each level of brain difference
- [x] PN channel sweep: 2D sweep of read and write projection neuron counts with calibration

**Reuse rule.** The pipeline modules, metrics, and sweep harness carry over unchanged from Stage 1. If Stage 2 requires modifying a shared module, that is a signal the Stage 1 abstraction was wrong and should be fixed in both.

## Stage 3: Adult *Drosophila*

Roughly 2,600 receptor neurons per antenna converging onto about 50 glomeruli per side, then expanding to some 2,000 Kenyon cells in the mushroom body, where each odor activates a small, sparse subset.

**What changes from Stage 2.**

- **Convergence and expansion in one pathway.** Both operations can now be tested as read and write sites.
- **Individual variability becomes structural.** Glomerular wiring is consistent across flies; the projections into the mushroom body are not. This is the clearest test of the claim that bottlenecks are more alike between individuals than expanded layers are, and therefore better write targets.
- **The richest insect validation data.** DoOR covers most receptor types against hundreds of odors, and a published whole-brain spiking model provides both a reference implementation and a comparison point.

## Stage 4: Zebrafish (*Danio rerio*)

Stage 4 adds a vertebrate olfactory system with fundamentally different architecture: recurrent lateral inhibition via granule cells in the olfactory bulb.

**Circuit.** 100 OR types × 3 ORNs each → 100 glomeruli → 80 mitral cells → 200 granule cells (recurrent lateral inhibition) → 500 Dp (dorsal pallium) neurons → 2 output neurons.

**What changes from Stage 3.**

- **Vertebrate architecture.** The granule cell layer provides recurrent lateral inhibition between mitral cells — a computation absent from the insect pathway. This decorrelates representations before they reach downstream areas.
- **Mitral cells as the bottleneck.** Unlike insects where PNs map one-to-one from glomeruli, zebrafish mitral cells receive convergent input, creating a tighter bottleneck (80 mitral cells for 100 glomeruli).
- **30 odor profiles** spanning attractive and repellent valences, compared against the insect odor panels.

**Experiments.** All four standard experiments plus the mitral cell channel sweep (the zebrafish equivalent of the PN channel sweep).

- [x] Broadcast baseline across brain-difference levels
- [x] Calibration vs. direct broadcast
- [x] ORN channel sweep
- [x] Hardware boundary with technology annotations
- [x] Mitral cell channel sweep (2D read/write)

**Scaling result.** With all four stages complete, the scaling comparison plots requirements — channels, precision, calibration cost — against receptor-neuron count and glomerular count. That trend across four organisms (including the first vertebrate) is the project's most transferable output.

## Data sources

All required data is public. No data collection is needed at any stage.

| Stage | Wiring | Odor input data | Validation data |
| --- | --- | --- | --- |
| Worm | Cook et al. 2019 connectome (via cect); CeNGEN for neurotransmitter signs | Calcium imaging of AWA, AWC, AWB, ASH odor responses | Published whole-brain calcium recordings, including systematic single-neuron stimulation datasets |
| Larva | Larval brain connectome (2023) | Kreher et al. empty-neuron recordings of the larval receptor repertoire | Larval behavioral and imaging studies |
| Adult fly | FlyWire | DoOR database; Hallem & Carlson 2006 | Published glomerular imaging; the 2024 connectome-based whole-brain spiking model |
| Zebrafish | Published olfactory bulb circuit models | OR response profiles across 30 odors | Published glomerular and mitral cell imaging |

**Reference sources for neuron identity and function:** WormAtlas and WormBase for the worm; FlyWire and Virtual Fly Brain for the fly. Both use standard neuron naming that maps directly into simulation code.

**Caveat on worm input data.** It is the weakest of the three. The worm cannot be recorded with the empty-neuron method, and calcium imaging is a delayed, smoothed, saturating proxy for voltage that reports increases far better than decreases. Treat worm odor responses as coarse.

## Validation strategy

Validation happens once per organism, before any transfer experiments.

**Match the recording method, not the truth.** Published worm data is calcium imaging, which is not a voltage trace. Do not compare simulated membrane potential directly to recorded fluorescence. Pass simulated activity through a calcium indicator model — adding rise and decay delay, smoothing, saturation, and noise — and compare that output to the published recordings. The same discipline applies to fly spike-rate data, which is typically reported as counts over a time window rather than precise timing.

**Validation checks per stage.**

- Receptor neurons respond to the right odors, in the right direction, with roughly the right magnitude ordering
- The circuit produces the correct behavior: approach for attractants, avoidance or reversal for repellents
- Known asymmetries reproduce, such as the deliberate left/right difference between the two worm AWC neurons
- For the fly stages, the extracted subnetwork reproduces the full model's behavior on a sample of cases

**Robustness in place of certainty.** Synaptic weights are not known for any of these organisms. The connectome gives who connects to whom and roughly how many synapses, but not strength, and sign is inferred from neurotransmitter identity. Every headline result must be re-run across model variants — different weight assignments, graded versus spiking where both are plausible, with and without background neuromodulation. A conclusion that survives those variations is probably about the architecture. A conclusion that does not is about one model, and should be reported as such.

**Where validation is not needed.** The transfer experiments have ground truth for every neuron, so receiver activity can be compared to sender activity directly. The indicator model reappears only when simulating an optical read device, where it becomes part of the hardware under test.

## Target results

Three headline results. Each should be reported as a curve or a map, never a single number — requirements depend heavily on what is being transmitted.

1. **Channels needed versus brain difference.** How many neurons must be connected as the two brains diverge, from near-identical to substantially different.
2. **The hardware boundary.** A map of which combinations of channel count, stimulation precision, and noise succeed or fail, with a sensitivity ranking of which parameter matters most.
3. **Best place to connect — read and write separately.** Transfer quality for each pairing of read site and write site.

**Supporting results.**

- Calibration cost: how many shared odors are needed before the transform generalises to unseen odors, and whether a linear mapping suffices
- Information rate: distinct messages, or bits per second, under realistic conditions
- Compression limits: how far the signal can be reduced before transfer degrades
- Drift tolerance: how fast performance falls as the read and write sites shift, and how often recalibration is needed
- Direct broadcast baseline: what simple copying achieves with no translation

## Implementation notes

**Language.** Python throughout.

| Purpose | Tools |
| --- | --- |
| Spiking simulation | Brian2 (compiles to C++ behind the scenes) |
| Graded / custom models | NumPy, or PyTorch for GPU-accelerated sweeps |
| Worm model generation | Cook et al. 2019 connectome via cect, with CeNGEN neurotransmitter signs |
| Connectome access | FlyWire Python libraries |
| Analysis and calibration | NumPy, SciPy, scikit-learn |
| Parameter search | Optuna, for adaptive sweeps that focus near the success boundary |
| Plots | Matplotlib |

**Compute.** Worm simulations run in a fraction of a second each, so full sweeps finish in minutes on a MacBook Pro. The adult fly brain needs a few gigabytes and runs in seconds to minutes per trial; large sweeps take hours to days and benefit from PyTorch on Apple Silicon GPU and from parallelism across cores. Embodied simulation with a physics body model is out of scope.

**Repository structure.** Keep the pipeline modules organism-agnostic from the first commit, with organism-specific code confined to model loading and site definitions:

```
models/      per-organism brain models and individual-variation generators
hardware/    read, write, noise, and compression models
link/        translate, encode, transmit
experiments/ organized by sensory modality (e.g. experiments/olfactory/)
papers/      write-ups and manuscripts
validation/  comparisons against published data, incl. indicator models
analysis/    metrics and figures
```

**Working method.** Build one piece at a time, running it before moving on. Do not write large amounts of code before executing anything.

## Known limitations

- **Results are lower bounds.** A design that fails in simulation will almost certainly fail in reality; a design that passes is a candidate, not a guarantee.
- **Synaptic weights are unknown.** Connectomes give connectivity and synapse counts, not strengths, and sign is inferred from neurotransmitter identity — which is not always decisive, since glutamate can excite or inhibit depending on the receiving neuron's receptors.
- **Wireless chemical signaling is missing.** Worms rely heavily on neuropeptides that act between neurons with no synapse between them. Studies stimulating neurons one at a time have found real signal flow often departs from the wiring diagram.
- **Worm olfactory input data is coarse.** Responses cannot be decomposed receptor by receptor, and calcium imaging is an imperfect proxy for the underlying signal.
- **Extracted subnetworks are not the whole brain.** Removing neurons removes inputs, including inhibition and background drive.
- **No claim is made about human transfer.** The only human-facing output is the scaling trend across four organisms, presented as a trend, not an extrapolation.
- **Linear calibration is unrealistically powerful.** Ridge regression achieves near-perfect correlation across all organisms, masking architectural differences. Calibration sample efficiency scales with dimensionality (N² parameters), making it primarily a matrix-math artifact rather than a biological finding.
- **Nothing here establishes subjective equivalence.** A perfect transfer could only show that the receiver reacts to, describes, and uses the signal as the sender would.

## First milestone

**Milestone 1: one worm, one odor, one behavior.** Before any linking, two brains, or sweeps, get a single simulated worm whose AWA and AWC neurons respond to an attractive odor and whose command neurons produce the correct movement response. Everything else in the project depends on this working.

**Completed milestones.**

- [x] Set up the project directory with the module structure above
- [x] Decide the worm model source — chose Cook et al. 2019 connectome with CeNGEN neurotransmitter signs; also kept a hand-built model for comparison
- [x] Build Milestone 1 and validate it against published odor-response data
- [x] Add the individual-variation generator and confirm variants pass the sanity check
- [x] Implement the direct-broadcast baseline before building translation — it is the comparison every later result is measured against
- [x] Stage 1 complete: worm broadcast, calibration, and hardware boundary experiments all pass
- [x] Stage 2 complete: larva with 21-ORN bottleneck, channel sweep, PN channel sweep
- [x] Stage 3 complete: adult fly with 50-PN → 2000-KC expansion, all experiments
- [x] Stage 4 complete: zebrafish with vertebrate recurrent lateral inhibition architecture
- [x] Cross-organism scaling comparison across all four organisms
- [x] Replaced 2-way odor-type decoding with N-way odor identification metric
- [x] Removed receiver learning experiments (calibration handles the transform)
- [x] Calibration efficiency sweep across organisms

**Resolved questions.**

- **Calibration sample efficiency is primarily dimensional.** Samples to 90% odor ID scale ~linearly with N (ratios: 0.63, 1.43, 1.02, 0.75 for N = 8, 21, 49, 100), not with N². The sub-N² scaling comes from correlations in receptor response profiles that ridge regularization exploits. No clear biological effect beyond dimensionality — calibration efficiency does not distinguish organisms.
- **Channel sweep curves do not collapse when normalized by bottleneck fraction.** Confirmed: at 50% of channels connected (jitter=0.0), odor ID is 0.44 (larva), 0.36 (adult), 0.59 (zebrafish). The adult fly is consistently worst despite having an intermediate channel count, because its 53× KC expansion amplifies missing-channel errors. This is the strongest evidence that architecture — not dimensionality — determines transfer requirements.
- **Zebrafish granule cell lateral inhibition outperforms the adult fly at matched channel fractions** (e.g., +0.23 odor ID gap at 50% channels), and zebrafish broadcast maintains perfect behavioral match even at jitter=1.5 while the adult fly drops to 0.72 at jitter=0.75. However, the zebrafish model uses synthetic connectivity, so the granule cell effect could be an artifact of the hand-built model. This cannot be resolved without a zebrafish olfactory bulb connectome.
