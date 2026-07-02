[![Release](https://img.shields.io/github/v/release/rfvasile/linalg-zero?no-cache)](https://img.shields.io/github/v/release/rfvasile/linalg-zero?no-cache)
[![Build status](https://img.shields.io/github/actions/workflow/status/rfvasile/linalg-zero/main.yml?branch=main)](https://github.com/rfvasile/linalg-zero/actions/workflows/main.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Linalg-Zero

Check out the [poster](docs/poster.pdf), the [paper](docs/report.pdf) and the [demo](https://www.youtube.com/watch?v=Dxc3yTr-AE0). The model is also deployed on [HuggingFace Spaces](https://huggingface.co/spaces/rfvasile/linalg-zero).

![image](https://github.com/user-attachments/assets/b7019c34-8dcf-45a3-830e-050a822e9ff0)

## Overview

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#main-phases">Main Phases</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quickstart">Quickstart</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#artifacts">Artifacts</a></li>
    <li><a href="#reproducibility">Reproducibility</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

This repository offers tools for generating a linear algebra dataset and training an open-source base model (i.e. `Qwen2.5-3B`), aiming to explore planning and tool use using SFT and RL.

The project is designed as follows:

- `linalg_zero/`: contains the scripts to train models as well as generate synthetic data:
    - `generate.py`: generates the linear algebra dataset and splits.
    - `distillation.py`: runs the distillation pipeline to create multi-turn tool-use data.
    - `sft_train.py`: performs a simple SFT of a model on a dataset.
    - `grpo_train.py`: trains a model with GRPO on a given dataset.
- `Makefile`: contains commands for the dataset generation and common training workflows.

## Main Phases

We use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, but the project phases are:

* Step 1: generate a linear algebra dataset with controlled difficulty and tool-call metadata.
* Step 2: distill multi-turn tool-use data from a teacher model.
* Step 3: train the base model using SFT teaching the tool-calling format.
* Step 4: traing using GSPO to improve model robustness, using a curriculum.


## Installation

We use `uv` as the dependency management tool.
First, install `uv` by following the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

To run the experiments:

* For generation/distillation: `make install-data-gen`
* For SFT: `make install-sft`
* For RL: `make install-grpo`

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

## Quickstart

After installing the dependencies, execute the following commands:

```shell
# Phase 1: Generate dataset
uv run python linalg_zero/generate.py --dataset_name rfvasile/linalgzero --push_dataset

# Phase 2: Distillation (setup once)
cp linalg_zero/config/distillation/env.example.sh env.sh
# Edit env.sh to set HF_TOKEN and ARGILLA_API_KEY.
source env.sh

# Terminal A
uv run python linalg_zero/distillation/launch_server.py --config linalg_zero/config/distillation/vllm_qwen3_32b.yaml

# Terminal B (new terminal; source env.sh again)
source env.sh
uv run python linalg_zero/distillation.py --config linalg_zero/config/distillation/vllm_qwen3_32b.yaml

# Phase 3: SFT
uv run python linalg_zero/sft_train.py --config linalg_zero/config/sft/qwen2.5-3B/lora.yaml

# Phase 4: GRPO
uv run python linalg_zero/grpo_train.py --config-name runpod.yaml
```

Training requires the dataset to follow the strict OpenAI tool-calling format (see [link](https://huggingface.co/docs/trl/en/dataset_formats#tool-calling)). We provide scripts to prepare and validate the data accordingly:

- `linalg_zero/`
    - `sft/scripts/prepare_dataset.py`: prepares the SFT dataset.
    - `grpo/scripts/prepare_dataset.py`: prepares and validates the GRPO dataset.

## Results

We provide a recipe to encourage planning and tool-use capabilities using the [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) pre-trained model.

The resulting checkpoints: [Linalg-Zero-SFT](https://huggingface.co/rfvasile/LinalgZero-SFT) and [Linalg-Zero-GRPO](https://huggingface.co/rfvasile/LinalgZero-GRPO). The resulting accuracy is shown below:


| Metric             | LinAlgZero-SFT | LinAlgZero-GRPO |
|--------------------|----------------|-----------------|
| Optimal Trajectory | 89.87%         | 90.26%          |
| Correctness        | 91.86%         | 92.63%          |
| Format Validity    | 96.15%         | 96.66%          |
| Tool Success       | 100.00%        | 100.00%         |

### Artifacts

For a complete list of deliverables checkout page 66 in our [final report](./docs/report.pdf).

| Artifact | Link |
|---|---|
| SFT checkpoint | [rfvasile/LinalgZero-SFT](https://huggingface.co/rfvasile/LinalgZero-SFT) |
| GRPO checkpoint | [rfvasile/LinAlgZero-GRPO](https://huggingface.co/rfvasile/LinAlgZero-GRPO) |
| Base dataset | [rfvasile/linalgzero](https://huggingface.co/datasets/rfvasile/linalgzero) |
| Distilled dataset | [rfvasile/linalgzero-distilled](https://huggingface.co/datasets/rfvasile/linalgzero-distilled-clean) |
| SFT dataset | [rfvasile/linalgzero-sft](https://huggingface.co/datasets/rfvasile/linalgzero-sft) |
| GRPO dataset | [rfvasile/linalgzero-grpo](https://huggingface.co/datasets/rfvasile/linalgzero-grpo) |

## Cost
- **Total:** ~$75 using a mix of cloud GPUs and local training.
  - **Distillation:** H100 80GB on [Runpod](https://www.runpod.io/) with [Qwen/Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8); 15 hours at $2.39/hr (~$25).
  - **SFT:** Local 24GB RTX 4090 with [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).
  - **GRPO:** RTX 6000 Ada on [Runpod](https://www.runpod.io/), improving on the SFT baseline; 57 hours at $0.77/hr (~$50).

## Acknowledgements
- We leverage the [distilabel](https://github.com/argilla-io/distilabel) library.
- The RL experiments are based on [ART](https://github.com/OpenPipe/ART).

## Citation

If you find this project is useful in your own work, please consider citing as follows:

```bibtex
@misc{linalg-zero,
    title = {Linalg-Zero: Distilling Neurosymbolic Reasoning for Linear Algebra in Small Language Models},
    url = {https://github.com/rfvasile/linalg-zero},
    author = {{Razvan F. Vasile}},
    month = {March},
    year = {2026}
}
```
