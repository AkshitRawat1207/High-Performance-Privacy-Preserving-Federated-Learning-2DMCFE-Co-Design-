# 2DMCFE: Privacy-Preserving Federated Learning via Algorithmic & Cryptographic Co-Design

> Compressing dual-mode multi-client functional encryption overhead from a linear O(n) bottleneck down to **<1.7%** of total training time on complex color-image benchmarks.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-enabled-EE4C2C.svg)](https://pytorch.org/)
[![gmpy2](https://img.shields.io/badge/gmpy2-modular%20arithmetic-orange.svg)](https://gmpy2.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Privacy](https://img.shields.io/badge/privacy%20check-100%25%20PASS-brightgreen.svg)](#privacy-guarantees)
[![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)](#)

---

## Table of Contents

- [Overview](#overview)
- [Core Innovation](#core-innovation)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Empirical Performance Benchmarks](#empirical-performance-benchmarks)
- [Privacy Guarantees](#privacy-guarantees)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**2DMCFE** (Dual-Mode Decentralised Multi-Client Functional Encryption) is a research framework for **Privacy-Preserving Federated Learning (PPFL)**. It targets a fundamental scalability problem in cryptographic federated learning: as the number of clients `n` grows, the cryptographic overhead of secure aggregation traditionally scales **linearly (O(n))**, making cryptography-heavy PPFL impractical for anything beyond toy models on grayscale datasets.

This project addresses that bottleneck not with a single cryptographic trick, but through **co-design** — jointly re-architecting the neural network topology, the client-side gradient pipeline, and the underlying functional encryption scheme so that each layer of the system reduces the burden on the others.

The result is a system that maintains competitive global model accuracy while keeping cryptographic overhead to a small, near-constant fraction of total training time, even on multi-channel color datasets such as **SVHN** and **CIFAR-100**.

## Core Innovation

Traditional 2DMCFE deployments suffer from:

1. **Parameter explosion** — `Flatten()`-based CNN heads scale weight counts with image resolution and channel depth, inflating the size of every encrypted vector.
2. **Aggregator-visible sparsity** — naive top-k gradient sparsification leaks *which* coordinates were selected, weakening the FE security model.
3. **Modular overflow** — fixed quantization scales over a 61-bit Mersenne prime field overflow when gradient magnitudes fluctuate across rounds.
4. **Unvectorized cryptographic math** — pure Python modular arithmetic loops dominate wall-clock time at scale.

2DMCFE resolves each of these at the source, rather than patching the encryption layer in isolation — this is the algorithmic/cryptographic **co-design** referenced in the project name.

## Key Features

| Feature | Problem Solved | Mechanism |
|---|---|---|
| **Network Parameter Compaction** | Parameter explosion on color images | Replaces `Flatten()` with `GlobalAveragePooling2D()` + `BatchNormalization()`, shrinking the model to **623,818 parameters** |
| **Client-Side Secure Top-k Sparsification** | Aggregator inference of selection masks | Clients zero out low-magnitude gradient noise **locally**, then transmit uniform, full-length vectors — the aggregator never observes which indices were sparsified |
| **Dynamic Adaptive Quantization (Qₜ)** | Field overflow over `P = 2⁶¹ − 1` | Rescales quantization step size every round based on peak gradient magnitude `G_max`, keeping all values inside the Mersenne prime field |
| **Vectorized Parallel Engine** | Slow item-by-item Python crypto loops | C-optimized vectorized NumPy + `gmpy2` modular arithmetic, distributed across cores via `ThreadPoolExecutor` |

## Architecture

### System Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              CENTRAL SERVER                              │
│                                                                            │
│   ┌────────────────────┐        ┌───────────────────────────────────┐    │
│   │  Aggregator (blind  │◄───────┤   two_dmcfe.py                    │    │
│   │  to selection mask) │        │   Dual-mode mask switching        │    │
│   │                     │        │   δ=1 (masked) / δ=0 (unmasked)   │    │
│   └─────────┬───────────┘        └───────────────────────────────────┘    │
│             │ encrypted, uniform-length gradient vectors                  │
└─────────────┼──────────────────────────────────────────────────────────┘
              │
      ┌───────┴────────┬────────────────┬ ... ┬────────────────┐
      ▼                ▼                ▼      ▼                ▼
┌───────────┐   ┌───────────┐   ┌───────────┐        ┌───────────┐
│ Client 1  │   │ Client 2  │   │ Client 3  │  ...   │ Client N  │
│           │   │           │   │           │        │           │
│ ┌───────┐ │   │ ┌───────┐ │   │ ┌───────┐ │        │ ┌───────┐ │
│ │Local  │ │   │ │Local  │ │   │ │Local  │ │        │ │Local  │ │
│ │Train  │ │   │ │Train  │ │   │ │Train  │ │        │ │Train  │ │
│ │(SGD + │ │   │ │(SGD + │ │   │ │(SGD + │ │        │ │(SGD + │ │
│ │Mom.)  │ │   │ │Mom.)  │ │   │ │Mom.)  │ │        │ │Mom.)  │ │
│ └───┬───┘ │   │ └───┬───┘ │   │ └───┬───┘ │        │ └───┬───┘ │
│     ▼     │   │     ▼     │   │     ▼     │        │     ▼     │
│ ┌───────┐ │   │ ┌───────┐ │   │ ┌───────┐ │        │ ┌───────┐ │
│ │Secure │ │   │ │Secure │ │   │ │Secure │ │        │ │Secure │ │
│ │Top-k  │ │   │ │Top-k  │ │   │ │Top-k  │ │        │ │Top-k  │ │
│ │Sparse │ │   │ │Sparse │ │   │ │Sparse │ │        │ │Sparse │ │
│ └───┬───┘ │   │ └───┬───┘ │   │ └───┬───┘ │        │ └───┬───┘ │
│     ▼     │   │     ▼     │   │     ▼     │        │     ▼     │
│ ┌───────┐ │   │ ┌───────┐ │   │ ┌───────┐ │        │ ┌───────┐ │
│ │Qₜ     │ │   │ │Qₜ     │ │   │ │Qₜ     │ │        │ │Qₜ     │ │
│ │Quant. │ │   │ │Quant. │ │   │ │Quant. │ │        │ │Quant. │ │
│ └───┬───┘ │   │ └───┬───┘ │   │ └───┬───┘ │        │ └───┬───┘ │
│     ▼     │   │     ▼     │   │     ▼     │        │     ▼     │
│ ┌───────┐ │   │ ┌───────┐ │   │ ┌───────┐ │        │ ┌───────┐ │
│ │2DMCFE │ │   │ │2DMCFE │ │   │ │2DMCFE │ │        │ │2DMCFE │ │
│ │Encrypt│ │   │ │Encrypt│ │   │ │Encrypt│ │        │ │Encrypt│ │
│ └───────┘ │   │ └───────┘ │   │ └───────┘ │        │ └───────┘ │
└───────────┘   └───────────┘   └───────────┘        └───────────┘
      ▲                                                     ▲
      └──────────────── M of N clients sampled ─────────────┘
                         per round (M ≤ N)
```

### Per-Client Gradient Encoding Pipeline

```
Raw Local Gradient
       │
       ▼
┌─────────────────────┐
│ Top-k Sparsification │   locally zeroes low-magnitude coords,
│  (local, private)     │   full-length vector preserved
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ Dynamic Quantization │   scale = f(G_max, round t)
│      (Qₜ engine)      │   maps ℝ → ℤ_P
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  gmpy2 Vectorized     │   modular exponentiation / multiplication
│  Modular Arithmetic   │   over P = 2⁶¹ − 1, ThreadPoolExecutor
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│  2DMCFE Ciphertext    │   dual-mode mask δ applied
└──────────┬───────────┘
           ▼
      → Central Aggregator
```

## Empirical Performance Benchmarks

All experiments were conducted with the vectorized crypto engine enabled, over `P = 2⁶¹ − 1`.

| Dataset | Clients (N, M) | Local Epochs (E) | Global Accuracy | Total Training Time | Crypto Overhead |
|---|---|---|---|---|---|
| **MNIST** | N=12, M=12 | E=2 | **98.30%** | ~4,616 s | **5.9%** |
| **SVHN** | N=12, M=12 | E=5 | **93.05%** | ~37,942 s | **1.7%** |
| **CIFAR-100** | N=12, M=10 | E=5 | **61.20%** | ~86,181 s | **0.9%** |

**Key takeaway:** cryptographic overhead *decreases* as a proportion of total runtime on larger, more complex datasets — the co-designed pipeline amortizes fixed cryptographic costs against a growing local-training workload, rather than compounding them.

## Privacy Guarantees

| Check | Result |
|---|---|
| Dual-mode mask isolation (δ=1) | ✅ Active for **49 consecutive intermediate rounds** |
| Final-round unmasking (δ=0) | ✅ Triggered only at round **50** |
| Aggregator visibility into sparsification mask | ✅ **Blind** — never observes client selection indices |
| Overall privacy validation | ✅ **100% PASS** |

During training, individual client updates remain cryptographically masked (`δ=1`) across all intermediate rounds, so the aggregator can only ever combine — never inspect — individual contributions. Only the final aggregated result is unmasked (`δ=0`) at the last round.

## Repository Structure

```
.
├── cnn_model.py          # Network topologies (GlobalAveragePooling2D, BatchNormalization)
├── crypto_utils.py        # Mersenne prime field math, gmpy2 bindings, PRF hash generators
├── data_loader.py         # Multi-dataset splitting, non-IID/IID partitioning, shard caching
├── dmcfe.py                # Decentralised inner-product functional encryption routines
├── gradient_encoder.py    # Fixed-point dynamic quantization (Qₜ) engine + vector conversion
├── ppfl_cnn.py             # Federated orchestrator: local training, momentum SGD, Top-k, threading
├── two_dmcfe.py            # Dual-mode functional encryption scheme (δ=0 / δ=1 mask switching)
├── run_ppfl_cnn.py         # Entry point: hyperparameters, experiment runner, logging, plots
└── requirements.txt        # Python package dependencies
```

| File | Responsibility |
|---|---|
| `cnn_model.py` | Defines compact CNN topologies used across all datasets |
| `crypto_utils.py` | Low-level modular arithmetic, hashing, and field operations |
| `data_loader.py` | Dataset ingestion and IID / non-IID client partitioning |
| `dmcfe.py` | Core (single-mode) decentralised functional encryption primitives |
| `gradient_encoder.py` | Converts floating-point gradients to quantized field elements |
| `ppfl_cnn.py` | Orchestrates the end-to-end federated training loop |
| `two_dmcfe.py` | Extends `dmcfe.py` with dual-mode (δ) mask switching logic |
| `run_ppfl_cnn.py` | CLI entry point for running configured experiments |

## Installation

**Requirements:** Python 3.8+, PyTorch, NumPy, `gmpy2`, Matplotlib

```bash
# Clone the repository
git clone https://github.com/<your-username>/2dmcfe.git
cd 2dmcfe

# (Recommended) Create an isolated environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `gmpy2` requires the GMP, MPFR, and MPC C libraries to be installed at the system level before `pip install` will succeed (e.g. `apt install libgmp-dev libmpfr-dev libmpc-dev` on Debian/Ubuntu).

## Usage

Run experiments via `run_ppfl_cnn.py`, which handles hyperparameter configuration, benchmark logging, and plot generation.

### MNIST

```bash
python run_ppfl_cnn.py --dataset mnist --N 12 --M 12 --T 20 --k 50000 --batch_size 64
```

### SVHN

```bash
python run_ppfl_cnn.py --dataset svhn --N 12 --M 12 --T 50 --k 0 --batch_size 256 --local_epochs 5
```

### CIFAR-100

```bash
python run_ppfl_cnn.py --dataset cifar100 --N 12 --M 10 --T 50 --k 0 --batch_size 64 --local_epochs 5
```

## Configuration Reference

| Flag | Description |
|---|---|
| `--dataset` | Target dataset: `mnist`, `svhn`, or `cifar100` |
| `--N` | Total number of registered clients |
| `--M` | Number of clients sampled per round (M ≤ N) |
| `--T` | Total number of federated communication rounds |
| `--k` | Top-k sparsification threshold (`0` disables sparsification) |
| `--batch_size` | Local mini-batch size per client |
| `--local_epochs` | Number of local epochs (`E`) per client per round |

## Roadmap

- [ ] Extend co-design pipeline to Transformer-based client architectures
- [ ] Support asynchronous / straggler-tolerant aggregation rounds
- [ ] GPU-accelerated `gmpy2`-equivalent modular arithmetic backend
- [ ] Formal security proof writeup for the client-side Top-k masking scheme

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue first to discuss significant changes before submitting a pull request.

## License

Distributed under the MIT License. See `LICENSE` for details.
