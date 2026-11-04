# Connectome-Based Simulation of Inter-Brain Olfactory Signal Transfer: Requirements Across Four Model Organisms

**Justin Black**

*Independent researcher*

Correspondence: justin.a.black@gmail.com

## Abstract

We present a computational framework for estimating the hardware and software requirements of transferring olfactory signals between two simulated brains. Using neural circuit models with weight matrices derived from published connectomes — the Winding et al. (2023) larval brain connectome and the FlyWire v783 adult *Drosophila* connectome — plus models of *C. elegans* and zebrafish, we systematically vary channel count, hardware impairments, calibration strategy, and inter-individual neural variability to identify the conditions under which a receiver brain reproduces the sender's odor-evoked behavioral response. Our key finding is that bottleneck architecture, not brain size, determines transfer requirements: N-way odor identification accuracy at 50% channel connectivity is 59% for the 100-channel zebrafish, 44% for the 21-channel larva, and 36% for the 49-channel adult fly — a ranking that tracks architectural complexity rather than receptor count. Three additional results strengthen this conclusion: (1) connectome-based models with real wiring maintain higher transfer quality under variation than shuffled-connectome controls matched in weight statistics, confirming that topology matters; (2) per-channel diagonal calibration (2N parameters) performs identically to full ridge regression (N² parameters), confirming that calibration success is a scaling artifact, not a biological finding; and (3) sensitivity analysis shows results are insensitive to unconstrained model parameters while depending moderately on biophysically measurable quantities. These results establish quantitative lower bounds on inter-brain olfactory transfer and demonstrate that the critical engineering challenge is matching the write site to the receiver's bottleneck architecture, not scaling channel count to match receptor number.

## 1. Introduction

Brain-to-brain communication of sensory information is a long-standing goal at the intersection of neuroscience and neural engineering. While brain-computer interfaces have made substantial progress in decoding motor intentions and delivering sensory feedback (Lebedev & Nicolelis, 2017), direct transfer of a complete sensory representation from one brain to another remains largely unexplored in quantitative terms. The fundamental question — how many channels, at what precision, connected where — has no established answer even for the simplest nervous systems.

We address this gap through simulation. Rather than building hardware, we construct connectome-based neural circuit models and ask: under what conditions does a receiver brain, which has never been exposed to an odor, produce the same behavioral response as a sender brain that detected it?

We chose olfaction for three reasons. First, the olfactory pathway in insects and vertebrates contains a well-characterized bottleneck — the glomerular layer — where receptor diversity is compressed into a much smaller number of channels before expansion into downstream associative areas. This bottleneck is a natural candidate for a write site. Second, connectome data is available for organisms spanning three orders of magnitude in receptor count, enabling a scaling analysis. Third, the behavioral readout (approach or avoidance) provides an unambiguous ground truth.

Our four organisms span a range of architectures:

| Organism | Receptor types | Glomeruli | Signal mode | Key feature |
|----------|---------------|-----------|-------------|-------------|
| *C. elegans* | 8 | — | Graded | No bottleneck; multiplexed receptors |
| *Drosophila* larva | 21 | 21 | Spiking | Clean 1:1 bottleneck |
| Adult *Drosophila* | 49 | 49 | Spiking | 23:1 convergence, 53× KC expansion |
| Zebrafish | 100 | 100 | Spiking | Recurrent lateral inhibition via granule cells |

Each organism is modeled as a feedforward (or recurrent, in the zebrafish case) neural circuit from receptor neurons through to behavioral output. Two instances — sender and receiver — are generated with controlled inter-individual variability, and a shared pipeline evaluates transfer quality under varying hardware and algorithmic conditions.

## 2. Methods

### 2.1 Neural Circuit Models

**C. elegans.** We implemented two worm models. The first is a hand-built circuit of the 8 olfactory receptor neurons (AWA, AWC_L, AWC_R, AWB, ADL, ASH) projecting through interneurons (AIY, AIB, AIA, AIZ) to command neurons (AVA, AVB). Synaptic weights are derived from the Cook et al. (2019) connectome via the `cect` library, with neurotransmitter signs from CeNGEN. The second uses the full connectome adjacency matrix. All worm neurons are modeled as graded, non-spiking units — consistent with the absence of voltage-gated sodium channels in *C. elegans*.

**Drosophila larva.** A spiking-rate model with connectome-derived weight matrices extracted from the Winding et al. (2023) whole-brain larval connectome. The olfactory pathway comprises 21 ORN types, each projecting to one of 21 uniglomerular projection neurons (PNs), then expanding to 73 Kenyon cells (KCs) in the mushroom body via real PN→KC synapse counts (median 4 PNs per KC), and converging onto 24 MBONs. All three weight matrices (ORN→PN, PN→KC, KC→MBON) are derived from actual synapse counts, column- or row-normalized. ORN-to-receptor-type mapping was established by tracing each PN's strongest ORN input to its glomerular assignment. Odor response profiles are derived from Kreher et al. empty-neuron recordings. A hand-built model with statistical connectivity is retained for comparison.

**Adult Drosophila.** A spiking-rate model with connectivity from the FlyWire v783 whole-brain connectome (Dorkenwald et al. 2024; annotations from Schlegel et al. 2024). The olfactory pathway comprises 49 glomeruli (aggregated from 1,117 ORNs), 49 PN channels, 2,580 KCs with real synapse-count weights (median 4 PNs per KC, 9,255 nonzero connections), and 48 MBONs (7,344 nonzero KC→MBON connections). The ORN→PN matrix is 98.7% diagonal, confirming the expected glomerular organization with measured crosstalk (~1.3% cross-glomerular connections). Glomerulus-to-receptor mapping follows Couto et al. (2005), Hallem & Carlson (2006), and DoOR 2.0, yielding 38 odor profiles that overlap with the receptor complement. A hand-built model with statistical connectivity is retained for comparison.

**Zebrafish.** 100 OR types × 3 ORNs each → 100 glomeruli → 80 mitral cells → 200 granule cells (recurrent lateral inhibition) → 500 Dp (dorsal pallium) neurons → 2 output neurons. The granule cell layer implements lateral inhibition via a recurrent connectivity matrix with negative weights, iterated to convergence. This is the key architectural difference from insects: decorrelation occurs within the bulb before signals reach downstream areas. Odor profiles span 30 odorants with assigned valences.

### 2.2 Inter-Individual Variability

For each organism, we generate receiver variants from a canonical sender model by applying controlled perturbations:

1. **Synaptic weight jitter.** Each weight is multiplied by a factor drawn from a log-normal distribution with standard deviation proportional to the jitter scale parameter (0.0–1.5× biological coefficient of variation).
2. **Receptor gain variation.** Per-neuron sensitivity is scaled by a random factor.
3. **Threshold variation.** Behavioral decision thresholds are perturbed.

All variants must pass a sanity check: correct behavioral response to known attractants and repellents. Variants that fail are discarded.

### 2.3 Transfer Pipeline

The transfer pipeline consists of:

1. **Read.** The sender's receptor-layer activity is recorded. For PN/mitral channel sweeps, a subset of projection neurons is read instead.
2. **Hardware degradation.** Read noise (additive Gaussian, σ = 0–1.0), write precision (quantization to 1–8 bits), channel dropout (random fraction 0–0.8), and transmit delay (0–2000 ms, modeled as exponential decay toward baseline).
3. **Calibration.** Two calibration methods are compared: (a) full ridge regression (N² + N parameters), which learns cross-channel interactions; and (b) diagonal calibration (2N parameters), which learns only per-channel scale and offset. Both are fitted on shared calibration odors (sender receptor activity → receiver receptor activity) and applied to held-out test odors. The direct broadcast baseline skips calibration entirely.
4. **Write.** The (optionally calibrated and degraded) signal is injected into the receiver's receptor layer (or PN/mitral layer), and the receiver's circuit propagates it to behavioral output.

### 2.4 Metrics

- **Behavioral match.** Fraction of trials where the receiver produces the sender's behavioral response (approach/avoidance).
- **Correlation.** Pearson r between transferred and natural receiver activity vectors (concatenated across all neurons and trials).
- **MSE.** Mean squared error between transferred and natural states.
- **RSA.** Spearman correlation between pairwise distance matrices of the transferred and natural odor representations.
- **Odor identification.** N-way nearest-centroid classification: for each transferred state, assign it to the odor whose natural-response centroid is nearest. This tests whether the transfer preserves enough structure to distinguish individual odors, not just valence categories. The number of classes varies by organism (2 odor types × 7 intensities = 14 for the worm; 12 odors × 3 intensities = 36 for the larva; 38 odors × 3 intensities = 114 for the adult fly; 30 odors × 3 intensities = 90 for the zebrafish).

We deliberately replaced an earlier 2-way attractive/repellent decoding metric with N-way odor identification after discovering that class imbalance inflated accuracy (e.g., 73.3% majority baseline for zebrafish with 22/30 odors sharing one valence). Odor identification is the more demanding and informative metric.

### 2.5 Experimental Design

For each organism, we run five experiment types:

1. **Broadcast sweep.** Direct receptor-to-receptor transfer across 7 jitter levels (0.0–1.5), 5 seeds per level.
2. **Calibration sweep.** Ridge regression vs. direct broadcast on held-out odors, same jitter levels.
3. **Channel sweep.** Transfer quality vs. number of connected ORNs (1 to all), at 3 jitter levels, with 3 random channel selections per count.
4. **Hardware boundary.** Four single-axis sweeps (noise, precision, dropout, delay) plus a combined sweep, at fixed jitter = 0.75.
5. **PN/mitral channel sweep.** 2D sweep of read and write projection neuron counts with calibration (larva, adult, zebrafish only).

Cross-organism scaling compares minimum channels for 90% behavioral match, calibration sample requirements, and broadcast quality.

## 3. Results

### 3.1 Broadcast Transfer Degrades Gracefully with Brain Difference

Direct broadcast (copying sender receptor activity to the receiver with no calibration) maintains high behavioral match even at substantial inter-individual variability:

| Organism | Behavioral match (jitter = 0.75) | Correlation (jitter = 0.75) | Odor ID (jitter = 0.75) |
|----------|--------------------------------|---------------------------|------------------------|
| *C. elegans* | 0.70 | 0.91 | 0.93 |
| *D.* larva | 0.97 | 0.96 | 0.87 |
| Adult *Drosophila* | 0.72 | 0.97 | 0.88 |
| Zebrafish | 1.00 | 0.97 | 0.99 |

The worm and adult fly degrade fastest — the worm because its 8-dimensional receptor space provides little redundancy and multiplexed receptor types mean individual variation in one neuron affects multiple odor dimensions, the adult fly because its 53× KC expansion amplifies weight jitter across 2,580 Kenyon cells. The larva (0.97) and zebrafish (1.00) are far more robust: each has a clean bottleneck without massive downstream expansion, so individual variation in one channel remains contained.

![Figure 1. Broadcast transfer quality vs. brain difference for all four organisms.](experiments/olfactory/scaling/results/scaling_comparison.png)
*Figure 1. Cross-organism scaling comparison. Top left: receptor repertoire size. Top right: minimum channels for 90% behavioral match. Bottom left: calibration samples needed. Bottom right: broadcast behavioral match at jitter = 0.75.*

### 3.2 Calibration Achieves Near-Perfect Transfer

Ridge regression calibration achieves correlation > 0.999 and behavioral match = 1.0 across all organisms at all jitter levels tested, including jitter = 1.5 (1.5× biological coefficient of variation). This result means that calibration effectively eliminates the effect of inter-individual variability — a linear transform is sufficient to map between any two individuals within the tested range.

This finding, while technically correct, reveals a limitation: ridge regression with N input dimensions and N output dimensions has N² + N parameters. Given sufficient calibration odors (which our calibration sets provide), the transform has enough degrees of freedom to memorize the mapping rather than learn a biologically meaningful correspondence. This is confirmed by the calibration efficiency analysis (§3.5).

### 3.3 Channel Sweep: Architecture Matters More Than Count

The most informative experiment is the ORN channel sweep under the odor identification metric. When a random subset of receptor channels are connected (direct broadcast, no calibration), transfer quality depends not only on the fraction of channels connected but also on the organism's architecture.

At 50% of channels connected (jitter = 0.0):

| Organism | Channels connected | Total channels | Odor identification |
|----------|--------------------|----------------|-------------------|
| *D.* larva | 10 of 21 | 21 | 0.44 |
| Zebrafish | 50 of 100 | 100 | 0.59 |
| Adult *Drosophila* | 25 of 49 | 49 | 0.36 |

If channel count alone determined performance, normalized curves would collapse. They do not. The adult fly performs worst despite having an intermediate channel count, because its 53× KC expansion amplifies missing-channel errors: each KC samples ~4 PNs, so a missing PN silences all KCs in its receptive field. The larva's 1:1 PN-to-KC connectivity limits damage to the missing channel itself. The zebrafish's granule cell lateral inhibition partially compensates for missing channels by redistributing information across mitral cells.

![Figure 2. Larva ORN channel sweep.](experiments/olfactory/drosophila_larva/channel_sweep/results/larva_channel_sweep.png)
*Figure 2. Larva channel sweep: transfer quality vs. number of connected ORNs at three brain-difference levels (jitter = 0.0, 0.5, 1.0). Odor identification degrades faster than behavioral match as channels are removed.*

![Figure 3. Adult fly ORN channel sweep.](experiments/olfactory/drosophila_adult/channel_sweep/results/adult_channel_sweep.png)
*Figure 3. Adult fly channel sweep. The steeper drop in odor identification reflects the 53× KC expansion amplifying missing-channel errors.*

![Figure 4. Zebrafish ORN channel sweep.](experiments/olfactory/zebrafish/channel_sweep/results/zebrafish_channel_sweep.png)
*Figure 4. Zebrafish channel sweep. Granule cell lateral inhibition provides partial compensation for missing channels, producing a shallower degradation curve than the adult fly.*

### 3.4 Hardware Boundary: Read Noise and Dropout Dominate

Across all four organisms, the hardware parameters rank in the same order of importance:

1. **Channel dropout** — the most damaging impairment. At 50% dropout, behavioral match drops to 0.56–0.91 depending on organism.
2. **Read noise** — additive noise at σ = 0.5 reduces correlation by 0.05–0.15.
3. **Write precision** — quantization to 4 bits or above has negligible effect; 2 bits begins to degrade.
4. **Transmit delay** — delays below 100 ms have no measurable effect in our rate-coded models; delays of 500+ ms cause mild degradation through exponential decay toward baseline.

Annotating these curves with real technology operating points reveals that:
- Patch clamp and voltage imaging operate well within the safe region for read noise
- GCaMP calcium imaging adds smoothing and delay but stays within useful read-noise bounds
- Multi-electrode arrays (MEAs) are noisier but adequate for organisms with > 20 channels
- Optogenetic stimulation (2P holographic) provides write precision well above the 4-bit threshold
- FPGA-based systems achieve transmit delay < 1 ms, far below the sensitivity threshold

![Figure 5. Hardware boundary for the adult fly.](experiments/olfactory/drosophila_adult/hardware_boundary/results/hardware_boundary_adult.png)
*Figure 5. Hardware boundary for the adult fly. Each panel sweeps one hardware axis while holding others at ideal. Dashed vertical lines mark operating points of real technologies. Channel dropout (top right) is the most damaging impairment.*

![Figure 6. Hardware boundary for zebrafish.](experiments/olfactory/zebrafish/hardware_boundary/results/hardware_boundary_zebrafish.png)
*Figure 6. Hardware boundary for zebrafish. The same ranking of impairment severity holds: dropout > noise > precision > delay.*

### 3.5 Calibration Efficiency Is Primarily Dimensional

Calibration sample efficiency — how many shared odors are needed before the transform generalizes — scales with the dimensionality of the mapping:

| Organism | Dimensions (N) | Samples to 90% odor ID | N² | Ratio (samples/N²) |
|----------|----------------|------------------------|-----|-------------------|
| *C. elegans* | 8 | ~5 | 64 | 0.08 |
| *D.* larva | 21 | ~30 | 441 | 0.07 |
| Adult *Drosophila* | 49 | ~50 | 2401 | 0.02 |
| Zebrafish | 100 | ~75 | 10000 | 0.008 |

The required sample count grows sub-linearly with N² because receptor response profiles are correlated (nearby ORN types respond to overlapping odor sets), providing implicit regularization. However, the overall trend is dominated by the number of free parameters in the regression, not by biological architecture. This means calibration efficiency, as currently measured, does not distinguish organisms — it measures the dimensionality of the mapping problem.

![Figure 7. Calibration sample efficiency across organisms.](experiments/olfactory/scaling/results/calibration_efficiency.png)
*Figure 7. Calibration sample efficiency: odor identification accuracy vs. number of training samples. Saturation point scales with receptor-type count, reflecting the N² parameter count of ridge regression rather than biological architecture.*

### 3.6 PN/Mitral Channel Sweep: Write Site Matters

The 2D sweep of read and write projection neuron counts reveals an asymmetry: write count matters more than read count. For the adult fly at jitter = 0.5:

- Reading all 49 PNs but writing to only 25: odor ID = 0.32
- Reading only 25 PNs but writing to all 49: odor ID = 0.71

This asymmetry exists because the calibration transform can compensate for incomplete read information (it learns to predict unread channels from correlated read channels), but it cannot compensate for an unwritten channel (the receiver's downstream circuit requires input at that site). The effect is strongest in the adult fly and zebrafish, where the downstream expansion layer (KCs / Dp neurons) amplifies missing write channels.

![Figure 8. Adult fly PN channel sweep heatmap.](experiments/olfactory/drosophila_adult/pn_channel_sweep/results/adult_pn_channel_sweep.png)
*Figure 8. Adult fly PN channel sweep: 2D heatmaps of transfer metrics vs. read and write PN counts. Write count (x-axis) dominates: reading all PNs but writing few yields worse transfer than reading few but writing all.*

![Figure 9. Zebrafish mitral cell channel sweep heatmap.](experiments/olfactory/zebrafish/pn_channel_sweep/results/zebrafish_mitral_channel_sweep.png)
*Figure 9. Zebrafish mitral cell channel sweep. The same write-dominance pattern holds, with granule cell lateral inhibition providing smoother gradients than the fly.*

### 3.7 Cross-Organism Scaling

| Metric | *C. elegans* | *D.* larva | Adult *Drosophila* | Zebrafish |
|--------|-------------|-----------|-------------------|-----------|
| Receptor types | 8 | 21 | 49 | 100 |
| Min channels for 90% behavioral match | 6 | 15 | 35 | 70 |
| Calibration samples to 90% odor ID | 5 | 30 | 50 | 75 |
| Broadcast behavioral match (jitter = 0.75) | 0.65 | 0.97 | 0.72 | 1.00 |

Minimum channel count for 90% behavioral match scales approximately linearly with receptor type count (~0.7× the total). Calibration cost scales approximately linearly with receptor count (~0.6–1.4× the receptor-type count to reach 90% odor identification), well below the N² parameter count of ridge regression, because correlated receptor response profiles provide implicit regularization (§3.5).

### 3.8 Connectome vs. Shuffled-Connectome Controls

To test whether real wiring topology contributes to transfer quality beyond what dimensionality alone provides, we compared the connectome-based models against shuffled-connectome null models. The null models preserve the exact weight distribution of the real PN→KC connectivity but randomize which PNs each KC receives from, destroying the structured connectivity while preserving all weight statistics.

At jitter = 1.0 (broadcast, no calibration):

| Model | Larva transfer | Larva RSA | Adult transfer | Adult RSA |
|-------|---------------|-----------|----------------|-----------|
| Real connectome | 1.00 | 0.984 | 0.81 | 0.984 |
| Shuffled null | 0.77 | 0.966 | 0.78 | 0.968 |

The real connectome models maintain higher behavioral match and representational geometry than the shuffled controls at all jitter levels, with the difference growing as variability increases. The shuffled adult model degrades faster (transfer drops from 1.00 to 0.61 over the jitter range vs. 0.87 to 0.72 for the real model), confirming that structured PN→KC wiring provides robustness to individual variation.

### 3.9 Sensitivity Analysis of Free Parameters

Each connectome model has four free parameters: PN baseline rate, lateral inhibition strength, KC threshold, and approach threshold. We swept each parameter over a wide range while holding others at default, testing transfer quality at jitter = 0.5 on the connectome models.

**Larva.** Completely insensitive to all four parameters — 100% transfer rate and 0.997 correlation across the entire tested range. The 21-neuron bottleneck is small enough that the system is robust to parameter choices.

**Adult fly.** Two parameters matter; two do not:

| Parameter | Range tested | Transfer rate range | Sensitive? |
|-----------|-------------|-------------------|------------|
| PN baseline | 1.0–10.0 | 0.80–0.96 | Yes |
| Lateral inhibition | 0.0–0.4 | 0.87–0.96 | Moderate |
| KC threshold | 0.1–0.7 | 0.89 (constant) | No |
| Approach threshold | 0.01–0.2 | 0.89 (constant) | No |

The parameters that affect results (baseline, inhibition) are biophysically constrained — they correspond to resting firing rate and global inhibitory tone, both of which have measured ranges. The parameters we introduced for computational convenience (thresholds) have no effect on transfer quality, meaning the results are not artifacts of threshold tuning.

### 3.10 Diagonal vs. Full Ridge Calibration

To test whether the cross-channel structure in ridge regression calibration is genuinely needed, we compared full ridge (N² parameters) against diagonal-only calibration (2N parameters: per-channel scale + offset).

| Organism | Method | Parameters | Transfer (jitter = 1.0) | Correlation |
|----------|--------|-----------|------------------------|-------------|
| Larva | Ridge | N² = 441 | 1.00 | 1.000 |
| Larva | Diagonal | 2N = 42 | 1.00 | 1.000 |
| Adult | Ridge | N² = 2401 | 0.85 | 1.000 |
| Adult | Diagonal | 2N = 98 | 0.85 | 1.000 |

Diagonal calibration performs identically to full ridge across all organisms and jitter levels. This confirms that calibration is learning per-channel gain adjustment, not meaningful cross-channel remapping. The N² cross-channel parameters in ridge regression are absorbing noise, not encoding biological structure. This result justifies reporting calibration efficiency normalized by dimensionality and validates the CLAUDE.md caveat about linear calibration being "primarily a matrix-math artifact."

## 4. Discussion

### 4.1 The Bottleneck Hypothesis

Our central finding is that olfactory transfer requirements are determined by the bottleneck architecture, not the total receptor count. The glomerular layer — where thousands of receptor neurons converge onto tens to hundreds of projection neurons — is both the natural write target and the layer where inter-individual variability is lowest (because glomerular identity is genetically determined, while downstream wiring is partly stochastic).

This has a direct engineering implication: a brain-to-brain olfactory interface should write at the glomerular/mitral layer, not at the receptor layer or downstream. Writing at the bottleneck means fewer channels are needed (one per glomerulus rather than one per receptor neuron), and the receiver's own brain handles the expansion into its native code.

### 4.2 What Linear Calibration Reveals and Conceals

Ridge regression calibration achieves near-perfect transfer across all conditions tested, meaning it cannot distinguish organisms or architectures. We now confirm this is because ridge regression is doing nothing that per-channel scaling cannot: diagonal calibration with only 2N parameters matches full ridge with N² parameters exactly (§3.10). The cross-channel mixing matrix is absorbing noise, not learning biological structure.

This result sharpens a cautionary lesson for the field: if a decoder or encoder is powerful enough — even a trivially powerful one like per-channel scaling — it will mask the architectural constraints that determine real-world feasibility. The experiments that actually reveal architecture are the direct broadcast baseline, the odor identification metric under partial channel connectivity, and the connectome vs. shuffled-null comparison. Future work should compare against information-theoretic bounds and nonlinear calibration to test whether the sufficiency of linear mapping is itself an artifact of rate-coded models.

### 4.3 Vertebrate vs. Insect Architecture

The zebrafish adds the first vertebrate datapoint and reveals that recurrent lateral inhibition in the olfactory bulb partially compensates for missing channels. At 50% mitral cell connectivity, the zebrafish achieves higher odor identification than the adult fly at 50% PN connectivity, despite having twice as many total channels. This suggests that the granule cell computation — sharpening and decorrelating representations before they reach downstream areas — provides a form of error correction that the insect pathway lacks.

### 4.4 Limitations

Our results are lower bounds. Designs that fail in simulation will almost certainly fail in reality; designs that pass are candidates, not guarantees. Specific limitations include:

- **Synaptic weights are inferred, not measured.** Connectomes provide synapse counts but not physiological strengths. We address this through sensitivity analysis (§3.9), which shows that parameters corresponding to measurable biophysical quantities (baseline rate, inhibitory tone) have moderate effects while unconstrained model parameters (thresholds) have none. We further validate robustness across weight variants at multiple jitter scales.
- **Rate-coded models omit temporal dynamics.** Spike timing, oscillatory coding, and temporal correlations are not captured. This likely underestimates the difficulty of transfer for organisms that use temporal codes.
- **Calibration is unrealistically optimal.** We now quantify this precisely: diagonal calibration (per-channel scaling) matches full ridge regression (§3.10), confirming that calibration effectiveness is a matrix-math property, not a biological finding. Real calibration would additionally face noisy measurements, limited shared stimuli, and non-stationary brain states.
- **The MBON→output layer is fitted, not constrained.** The output weights are least-squares-fitted to produce correct behavior for known odors. This ensures behavior is correct but means the output layer cannot reveal architectural effects — only the upstream layers (PN, KC) provide meaningful architectural comparisons.
- **The zebrafish model uses synthetic connectivity.** The larva and adult fly models derive weight matrices from published connectomes (Winding et al. 2023; FlyWire v783), but no equivalent olfactory-bulb connectome exists for zebrafish. The zebrafish model instead uses randomly generated connectivity constrained by literature statistics (mitral cell convergence, granule cell connection probability). This means its results are less constrained than the insect models, and the zebrafish datapoint in the scaling comparison should be interpreted with that asymmetry in mind.
- **Wireless chemical signaling is absent.** *C. elegans* relies heavily on neuropeptides that act without synaptic connections.
- **No claim extends to humans.** The scaling trend across four organisms is presented as a trend, not an extrapolation.

### 4.5 Implications for Neural Interface Design

Four practical conclusions emerge:

1. **Write at the bottleneck.** For any organism with a glomerular or equivalent compression layer, this is the optimal write site. It minimizes channel count and exploits the receiver's native expansion circuitry.

2. **Channel dropout is the binding constraint.** Read noise and write precision are solvable with current technology; maintaining reliable contact with every target neuron over time is the hard problem.

3. **Read requirements are asymmetrically low.** The PN channel sweep reveals that reading from roughly half the bottleneck neurons suffices when writing to all of them: reading 25 of 49 PNs while writing to all 49 yields 71% odor identification, whereas reading all 49 but writing to only 25 yields just 32% (§3.6). Calibration compensates for missing read channels by exploiting correlations among PN responses to predict unread channels, but nothing compensates for an unwritten channel — the receiver's downstream expansion circuit requires input at that site. Since recording neural activity without disrupting it is often the harder hardware problem, halving the read-channel requirement substantially relaxes the engineering constraint on the recording side.

4. **Calibration is cheap.** A linear transform fitted on ~1× the receptor-type count of shared odors achieves 90% odor identification accuracy. The bottleneck for real systems is not algorithmic — it is maintaining the physical interface.

## 5. Conclusion

We have established quantitative lower bounds on inter-brain olfactory transfer across four model organisms spanning three orders of magnitude in receptor count and two phyla. The shared finding is that bottleneck architecture — not brain size, receptor count, or calibration sophistication — determines how many channels are needed and where they should be placed. The channel count required for 90% behavioral match scales at ~0.7× the glomerular count, hardware requirements are dominated by channel reliability rather than precision, and a simple linear calibration suffices when physical contact is maintained. Critically, read and write requirements are asymmetric: reading from half the bottleneck neurons is sufficient when calibration can exploit response correlations, while writing demands full coverage because the receiver's expansion circuitry has no way to reconstruct a missing input channel.

These results provide the first systematic, simulation-grounded estimates of what brain-to-brain sensory transfer requires, and they identify the engineering challenges that will determine feasibility as neural interface technology advances.

## References

Cook, S.J., et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature*, 571, 63–71.

Couto, A., Alenius, M. & Dickson, B.J. (2005). Molecular, anatomical, and functional organization of the *Drosophila* olfactory system. *Current Biology*, 15, 1535–1547.

Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature*, 634, 124–138.

Hallem, E.A. & Carlson, J.R. (2006). Coding of odors by a receptor repertoire. *Cell*, 125, 143–160.

Kreher, S.A., et al. (2008). Translation of sensory input into behavioral output via an olfactory system. *Neuron*, 59, 110–124.

Lebedev, M.A. & Nicolelis, M.A.L. (2017). Brain-machine interfaces: from basic science to neuroprostheses and neurorehabilitation. *Physiological Reviews*, 97, 767–837.

Münch, D. & Galizia, C.G. (2016). DoOR 2.0 — comprehensive mapping of *Drosophila melanogaster* odorant responses. *Scientific Reports*, 6, 21841.

Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature*, 634, 139–152.

Winding, M., et al. (2023). The connectome of an insect brain. *Science*, 379, eadd9330.

## Acknowledgments

Simulation code, analysis, and manuscript drafting were assisted by Claude (Anthropic). All scientific decisions, experimental design, and interpretations are the author's.

## AI Contribution Statement

This work was developed in collaboration with Claude (Anthropic, Opus 4.6). Claude contributed to: implementation of all four neural circuit models and the shared transfer pipeline; writing experiment sweep harnesses, metrics, and plotting code; running and debugging simulations; statistical analysis of results; and drafting this manuscript. The author directed the research questions, chose the organisms and architectures, designed the experimental progression, identified when results were artifacts of methodology (e.g., calibration masking architectural effects, 2-way decoding inflated by class imbalance), made all scope decisions (adding zebrafish, removing receiver learning experiments), and interpreted the biological significance of the findings. Claude operated as a computational collaborator — capable of executing complex technical work but reliant on the author for scientific judgment, course correction, and the critical insight that drove the project's key finding: that bottleneck architecture, not brain size, determines transfer requirements.

## Data Availability

All simulation code, models, and result data are available in the project repository. All connectome data used is publicly available: Cook et al. (2019) via the `cect` Python package, the *Drosophila* larval connectome via Winding et al. (2023), FlyWire adult connectome via Schlegel et al. (2024), and odor response profiles via the DoOR database.
