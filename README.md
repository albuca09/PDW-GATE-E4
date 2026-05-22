
````markdown
# PDW-GATE: Gated Attribute-Specialized Temporal Experts for Radar Pulse Deinterleaving

This repository implements **PDW-GATE**, a gated mixture-of-experts framework for radar pulse deinterleaving using Pulse Descriptor Words (PDWs). The model learns pulse-level embeddings from interleaved radar pulse sequences and applies density-based clustering to recover emitter-specific pulse groups.

The experiments are conducted on the **Turing Synthetic Radar Dataset (TSRD)**, using non-overlapping windows of 1024 pulses.

---

## Overview

Radar pulse deinterleaving aims to separate an interleaved stream of radar pulses into groups corresponding to their emitting sources. Each pulse is represented by a PDW vector containing attributes such as:

- Time of Arrival (ToA)
- Frequency
- Pulse Width
- Angle of Arrival (AoA)
- Amplitude

The proposed method combines:

1. **PDW feature engineering**
2. **Feature-specialized temporal experts**
3. **Soft gating over experts**
4. **Supervised contrastive embedding learning**
5. **HDBSCAN clustering in the learned embedding space**
6. **XAI visualization of gate weights and embedding behavior**

The best-performing configuration is referred to as:

## **PDW-GATE-E4**

where:

- **PDW** = Pulse Descriptor Word input representation
- **GATE** = gated attribute-specialized temporal experts
- **E** = engineered PDW features
- **4** = four expert branches

---

## Proposed Model

PDW-GATE learns pulse-level embeddings before clustering. Instead of applying HDBSCAN directly to raw PDWs, the model first transforms each pulse window into a discriminative embedding space.

### Pipeline

```text
Raw PDW sequence
        |
        v
PDW preprocessing / engineered features
        |
        v
Feature-specialized temporal MoE
        |
        v
Pulse-level embeddings
        |
        v
HDBSCAN clustering
        |
        v
Emitter-specific pulse clusters
````

---

## Input Format

Each TSRD `.h5` file is expected to contain:

```text
data   : [n_pulses, 5]
labels : [n_pulses, 1]
```

The PDW columns are:

| Index | Feature       |
| ----: | ------------- |
|     0 | ToA_us        |
|     1 | Frequency_MHz |
|     2 | PulseWidth_us |
|     3 | AoA_deg       |
|     4 | Amplitude_dB  |

Labels are used for training and evaluation, but **the clustering stage does not receive the true number of emitters**.

---

## Engineered PDW Features

For the best model, raw PDWs are converted into engineered features:

| Feature                    | Description                                  |
| -------------------------- | -------------------------------------------- |
| `toa_rel_norm`             | Relative normalized time of arrival          |
| `delta_toa_log`            | Log-scaled pulse interval / PRI-like feature |
| `frequency_norm`           | Normalized carrier frequency                 |
| `delta_frequency_abs_norm` | Absolute frequency variation                 |
| `pulsewidth_log`           | Log-scaled pulse width                       |
| `aoa_sin`                  | Circular AoA sine component                  |
| `aoa_cos`                  | Circular AoA cosine component                |
| `delta_aoa_abs_norm`       | Absolute angular variation                   |
| `amplitude_norm`           | Normalized amplitude                         |
| `amplitude_missing`        | Missing-amplitude indicator                  |

---

## Feature-Specialized Experts

PDW-GATE uses expert branches specialized in physically meaningful groups of PDW attributes.

For **PDW-GATE-E4**, the four experts are:

|   Expert | Feature group               |
| -------: | --------------------------- |
| Expert 1 | Time / PRI-related features |
| Expert 2 | Frequency-related features  |
| Expert 3 | Pulse-width features        |
| Expert 4 | AoA and amplitude features  |

Each expert contains:

* pointwise fully connected layers;
* temporal 1D convolution layers;
* residual normalization;
* projection to a pulse-level embedding.

A soft gating network dynamically weights the contribution of each expert for each pulse.

---

## Training Objective

The model is trained using a **supervised contrastive loss** within each 1024-pulse window.

Pulses belonging to the same emitter are pulled closer in the embedding space, while pulses from different emitters are pushed apart.

A small gate-balance regularization term is also used to avoid collapse into a single expert.

The final loss is:

```text
Loss = supervised_contrastive_loss + λ * gate_balance_loss
```

with:

```text
λ = 0.02
```

---

## Clustering

After training, embeddings are clustered using HDBSCAN.

The clustering stage does not use:

* the number of true emitters;
* the ground-truth labels;
* oracle information.

Default HDBSCAN parameters:

| Parameter                   | Value |
| --------------------------- | ----: |
| `min_cluster_size`          |     5 |
| `min_samples`               |     3 |
| `cluster_selection_method`  | `eom` |
| `cluster_selection_epsilon` |   0.0 |
| `allow_single_cluster`      | False |

---

## Experimental Setup

Main configuration:

| Parameter               |       Value |
| ----------------------- | ----------: |
| Window size             | 1024 pulses |
| Embedding dimension     |          64 |
| Hidden dimension        |         128 |
| Dropout                 |        0.15 |
| Batch size              |           4 |
| Epochs                  |          20 |
| Learning rate           |        2e-4 |
| Weight decay            |        1e-4 |
| Optimizer               |       AdamW |
| Contrastive temperature |        0.10 |
| Train windows           |        8000 |
| Validation windows      |        2000 |
| Test windows            |        5000 |

---

## Results

### Raw HDBSCAN Baseline

HDBSCAN applied directly to raw PDWs produced the following median test performance:

| Method                  | V-measure |    ARI |    AMI | Pairwise MCC |
| ----------------------- | --------: | -----: | -----: | -----------: |
| HDBSCAN raw PDWs        |    0.5226 | 0.2801 | 0.4938 |       0.3838 |
| HDBSCAN engineered PDWs |    0.5738 | 0.3516 | 0.5497 |       0.4503 |

Engineering PDW attributes improves the baseline, but performance remains limited because clustering is still performed directly in the input feature space.

---

### PDW-GATE Raw vs Engineered Features

| Model          | Experts |  V-measure |        ARI |        AMI | Pairwise MCC |
| -------------- | ------: | ---------: | ---------: | ---------: | -----------: |
| PDW-GATE-E4    |       4 | **0.9584** | **0.9985** | **0.9567** |   **0.9985** |
| PDW-GATE-E3    |       3 |     0.9561 |     0.9984 |     0.9543 |       0.9984 |
| PDW-GATE-E5    |       5 |     0.9545 |     0.9982 |     0.9529 |       0.9982 |
| PDW-GATE-E2    |       2 |     0.9511 |     0.9982 |     0.9491 |       0.9982 |
| Raw-PDW-GATE-2 |       2 |     0.9484 |     0.9979 |     0.9464 |       0.9979 |
| Raw-PDW-GATE-4 |       4 |     0.9385 |     0.9970 |     0.9364 |       0.9970 |
| Raw-PDW-GATE-3 |       3 |     0.9338 |     0.9954 |     0.9313 |       0.9954 |
| Raw-PDW-GATE-5 |       5 |     0.9223 |     0.9954 |     0.9194 |       0.9954 |

The best model is **PDW-GATE-E4**, which combines engineered PDW features with four physically motivated expert branches.

---

## Key Findings

* Direct HDBSCAN on raw PDWs provides a useful baseline but struggles with complex emitter separation.
* Engineered PDW features improve HDBSCAN performance.
* Learned embeddings substantially improve deinterleaving performance.
* Attribute-specialized experts improve the structure of the embedding space.
* The four-expert engineered configuration provides the best trade-off.
* The model achieves high clustering quality without using the true number of emitters at inference time.

---

## Repository Structure

Suggested organization:

```text
.
├── TSRD_GE.ipynb
├── README.md
├── requirements.txt
├── outputs/
│   ├── hdbscan_raw_vs_engineered/
│   ├── benchmark_moe_pdw_deinterleaving_sweep/
│   └── benchmark_moe_pdw_raw_vs_engineered_sweep/
└── figures/
```

The main implementation is currently contained in:

```text
TSRD_GE.ipynb
```

---

## Installation

Create a Python environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib h5py scipy pillow scikit-learn torch tqdm tabulate hdbscan huggingface_hub imageio
```

Alternatively, create a `requirements.txt` file with:

```text
numpy
pandas
matplotlib
h5py
scipy
pillow
scikit-learn
torch
tqdm
tabulate
hdbscan
huggingface_hub
imageio
```

Then run:

```bash
pip install -r requirements.txt
```

---

## Dataset

This repository uses the **Turing Synthetic Radar Dataset (TSRD)**.

The notebook downloads the dataset from Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
    repo_type="dataset",
    local_dir=r"D:\Fusion\turing_synthetic_radar_dataset",
    token=TOKEN,
    max_workers=2,
)
```

Expected local structure:

```text
D:\Fusion\turing_synthetic_radar_dataset
├── archive
│   ├── train
│   ├── validation
│   └── test
└── scan
```

Update the `ROOT` variable in the notebook if your dataset is stored elsewhere:

```python
ROOT = Path(r"D:\Fusion\turing_synthetic_radar_dataset")
```

---

## Running the Experiments

Open and run:

```text
TSRD_GE.ipynb
```

The notebook contains the following stages:

1. Dataset download
2. File inspection
3. Exploratory data analysis
4. Source/emitter counting
5. HDBSCAN baseline
6. Raw vs engineered HDBSCAN comparison
7. PDW-GATE feature-specialized MoE sweep
8. Raw vs engineered PDW-GATE comparison
9. XAI visualization and video export

---

## Output Files

The experiments generate CSV tables, plots, model checkpoints, and XAI visualizations.

Main output folders:

```text
benchmark_moe_pdw_deinterleaving_sweep/
benchmark_moe_pdw_raw_vs_engineered_sweep/
hdbscan_raw_vs_engineered/
reference_methodology_hdbscan/
```

Typical outputs include:

```text
comparison_all_results.csv
comparison_sorted_by_test_v_measure.csv
comparison_sorted_by_test_pairwise_mcc.csv
summary_result.csv
history.csv
best_model.pt
test_window_metrics.csv
feature_groups.json
config.json
```

---

## Explainability

The notebook includes an XAI module for analyzing the trained MoE model.

It exports:

* gate-weight visualizations;
* embedding-space cluster diagnostics;
* detected compact regions;
* per-window metrics;
* video visualizations of model behavior across pulse windows.

The XAI module helps inspect which expert branches are emphasized by the gate for different radar pulse patterns.

---

## Citation

If you use this repository, please cite the TSRD dataset and this implementation.

```bibtex
@misc{pdwgate2026,
  title  = {PDW-GATE: Gated Attribute-Specialized Temporal Experts for Radar Pulse Deinterleaving},
  author = {Luis Paulo Guedes},
  year   = {2026},
  note   = {GitHub repository}
}
```

---

## License

Add your preferred license here.

Suggested options:

* MIT License for open research code;
* Apache-2.0 License for permissive use with patent grant;
* restricted/custom license if dataset or institutional constraints apply.

---

## Acknowledgments

This work uses the Turing Synthetic Radar Dataset for radar pulse deinterleaving experiments. The proposed PDW-GATE model builds on the idea of combining physically meaningful PDW feature groups, temporal neural experts, supervised contrastive learning, and density-based clustering.

```
```
