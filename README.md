# Generative Sequential Recommendation with Semantic IDs

A complete pipeline for generative sequential recommendation using LLMs and RQ-VAE Semantic IDs, featuring multi-task SFT training, GRPO reinforcement learning, and constrained beam search decoding.

## Overview

This project implements a generative recommendation system that models item recommendation as a sequence generation task. Items are represented as discrete **Semantic IDs (SIDs)** via Residual Quantization VAE (RQ-VAE), and a fine-tuned LLM generates the next item's SID given a user's interaction history.

### Key Features

- **RQ-VAE Semantic IDs**: 3-layer residual quantization (256 codebook per layer) maps item embeddings to discrete token sequences `<a_X><b_Y><c_Z>`
- **Multi-task SFT**: Joint training on sequence recommendation, SID-title alignment, and cross-modal fusion tasks
- **GRPO Reinforcement Learning**: Group Relative Policy Optimization further improves recommendation quality (+8.1% HR@10)
- **Prefix Trie Constrained Decoding**: Ensures all generated SIDs are valid via logits masking
- **Catalog-Size Scaling Law**: Empirical power-law relationship `HR@10(N) = 3.30 × N^(-0.59) + 0.021` (R² = 0.995)

## Architecture

```
User History → [SID Tokens] → LLM (Qwen2.5-0.5B) → Constrained Beam Search → Recommended Item SID
                                    ↑
                          Multi-task SFT + GRPO RL
```

### Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `phase1_data_loading.py` | Load Amazon Reviews 2023 data, 5-core filtering |
| 2 | `phase2_subset_construction.py` | Nested subset construction (S1⊂S2⊂S2.5⊂S3) via log-binning stratified sampling |
| 3 | `phase3_sequence_construction.py` | User sequence building, leave-last-out split, C=10 per-item capping |
| 4 | `phase4_sid_generation.py` | RQ-VAE training and SID assignment using FAISS ResidualQuantizer |
| 5 | `phase5_train.py` | Multi-task SFT training (3 tasks, ConcatDataset) |
| 6 | `phase6_rl_train.py` | GRPO reinforcement learning training |
| Eval | `eval/run_eval.py` | Constrained beam search evaluation (HR@10, NDCG@10) |

## Results

### SFT Baseline vs GRPO RL (N=5K, valid set)

| Model | Beam=20 | Beam=50 | Beam=100 |
|-------|---------|---------|----------|
| SFT Baseline (HR@10) | 0.0628 | 0.0581 | 0.0577 |
| **GRPO RL (HR@10)** | **0.0649** | **0.0628** | **0.0594** |
| Improvement | +3.3% | **+8.1%** | +2.9% |

### Catalog-Size Scaling (SFT, Beam=50)

| Catalog Size N | HR@10 | NDCG@10 |
|----------------|-------|---------|
| 5K | 0.0441 | 0.0213 |
| 20K | 0.0318 | 0.0170 |
| 50K | 0.0260 | 0.0140 |
| 100K | 0.0266 | 0.0151 |

**Key Finding**: HR@10 follows a power-law decay with catalog size: `HR@10(N) = 3.30 × N^(-0.59) + 0.021`, with decay rate significantly higher than model-size scaling (Kaplan β≈0.076), indicating that catalog expansion is the primary scaling challenge for generative recommendation.

## Model & Data

- **Base Model**: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Embedding Model**: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (for RQ-VAE input)
- **Dataset**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Cell Phones & Accessories category
- **Vocabulary Extension**: +768 SID tokens (256 × 3 layers)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `trl>=0.15.0`, `peft>=0.7.0`, `faiss-cpu`, `datasets`

### Data Processing

```bash
python phase1_data_loading.py
python phase2_subset_construction.py
python phase3_sequence_construction.py
python phase4_sid_generation.py
```

### SFT Training

```bash
python phase5_train.py --config train/configs/M_N_5K.json
```

### GRPO RL Training

```bash
python phase6_rl_train.py --config train/configs/RL_M_N_5K.json
```

### Evaluation

```bash
python eval/run_eval.py \
  --model_dir train/runs/M_N_5K/final_checkpoint \
  --test_file data/sequences/S1/test.csv \
  --index_file data/sequences/S1/S1.index.json \
  --beam 50
```

## Project Structure

```
├── phase1_data_loading.py          # Data loading & 5-core filtering
├── phase2_subset_construction.py   # Nested subset construction
├── phase3_sequence_construction.py # Sequence building & C=10 capping
├── phase4_sid_generation.py        # RQ-VAE SID generation
├── phase5_train.py                 # Multi-task SFT training
├── phase6_rl_train.py              # GRPO RL training
├── grpo_trainer.py                 # GRPORecTrainer implementation
├── eval/
│   ├── run_eval.py                 # Single model evaluation
│   └── run_all_eval.py             # Batch evaluation
├── train/configs/                  # Training configurations
├── data/                           # Data files (not tracked)
├── embeddings/                     # RQ-VAE config & SID mappings
├── trie/                           # Prefix trie for constrained decoding
├── analysis/                       # Results & visualization
├── project_summary/                # Detailed documentation (Chinese)
└── reference/minionerec/           # Reference implementation from MiniOneRec
```

## GRPO Implementation Details

The GRPO trainer (`grpo_trainer.py`) implements Group Relative Policy Optimization adapted for recommendation:

- **Group Sampling**: Each prompt generates `G=4` candidate SIDs via `RepeatRandomSampler`
- **Advantage**: Group-normalized reward `A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-4)`
- **KL Penalty**: f-divergence form with `β=0.04`, reference model on separate GPU
- **Reward Functions**: Rule reward (exact match), ranking reward (NDCG-aware), hierarchy reward (partial SID match)
- **Constrained Decoding**: `prefix_allowed_tokens_fn` ensures valid SID generation during RL exploration

## Constrained Decoding

All generated SIDs are guaranteed to be valid through prefix trie constrained decoding:

1. Build a hash dictionary mapping token ID prefixes to allowed next tokens
2. At each generation step, mask invalid tokens to `-inf` in logits
3. Only 3 decoding steps needed (one per SID layer), minimal latency overhead

## References

- [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) — Reference implementation for generative recommendation with RL
- Kaplan et al., 2020 — "Scaling Laws for Neural Language Models"
- Hoffmann et al., 2022 — "Training Compute-Optimal Large Language Models" (Chinchilla)

## License

This project is for academic research purposes.
