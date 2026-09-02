# ByteChunk

**When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages**

*Accepted at EMNLP 2026 Main Conference*

[Sanjeev Kumar](https://cse.iitb.ac.in/~sanjeev)<sup>1,2</sup>, Atsuki Yamaguchi<sup>2</sup>, Nikolaos Aletras<sup>2</sup>

<sup>1</sup>CSE, IIT Bombay &nbsp; <sup>2</sup>School of Computer Sciecne, University of Sheffield

---

## Overview

Subword tokenizers fragment low-resource Indic languages — 97.7% of Bhojpuri words are split into multiple subword tokens. ByteChunk bypasses tokenization entirely by processing raw UTF-8 bytes and learning word-aligned chunks from 500K Hindi sentences, transferring zero-shot to Bhojpuri, Magahi, Sanskrit, Marathi, and Urdu.

### Key Results (Zero-Shot POS Tagging)

| Language | Subword | H-Net | Gain |
|----------|---------|-------|------|
| Bhojpuri | 43.8 | 54.4 | +10.6 |
| Marathi  | 39.1 | 46.3 | +7.2  |
| Magahi   | 41.0 | 53.1 | +12.1 |
| Sanskrit | 21.6 | 30.6 | +9.0  |
| Urdu     | 38.8 | 52.1 | +13.3 |
| **Avg**  | 36.9 | **47.3** | **+10.4** |

---

## Installation

```bash
conda create -n bytechunk python=3.10
conda activate bytechunk

pip install torch==2.1.0 transformers==4.40.0
pip install mamba-ssm causal-conv1d
pip install conllu datasets tqdm numpy
```

---

## Repository Structure
```
ByteChunk/
├── configs/
│   ├── __init__.py
│   └── default.py
├── models/
│   ├── __init__.py
│   ├── hnet_encoder.py
│   ├── routing.py
│   ├── chunking.py
│   ├── smoothing.py
│   ├── upsample.py
│   ├── hnet_loss.py
│   └── load_hnet.py
├── scripts/
│   ├── __init__.py
│   ├── pretrain.py
│   ├── eval_chunk_pos.py
│   ├── eval_chunk_ner.py
│   ├── eval_chunk_sentiment.py
│   ├── eval_cpt_pos.py
│   ├── eval_cpt_ner.py
│   ├── eval_cpt_sentiment.py
│   ├── qwen_subword_pos_baseline.py
│   ├── qwen_subword_ner_baseline.py
│   ├── qwen_subword_sentiment_baseline.py
│   └── cpt_baseline.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── optimizer.py
│   └── checkpoint.py
├── data/
│   ├── __init__.py
│   ├── pretrain_dataset.py
│   ├── pos_dataset.py
│   └── sentiment_dataset.py
├── tasks/
│   ├── __init__.py
│   └── task_heads.py
├── backbone/
│   ├── __init__.py
│   └── lm_head.py
├── utils/
│   ├── __init__.py
│   └── logging.py
├── README.md
└── LICENSE
```

---

## Data

### Pretraining data
Download FineWeb-2 Hindi (500K sentences):
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-2',
                  name='hin_Deva', split='train',
                  streaming=True)
# save first 500K sentences to data/fineweb_hindi/
"
```

### POS supervision data
Download UD Hindi HDTb treebank:
```bash
# From https://universaldependencies.org
# Place at: data/ud_hindi_treebank/hi_hdtb-ud-train.conllu
```

### Evaluation data
| Language | Task | Source |
|----------|------|--------|
| Bhojpuri | POS | UDAPI Bhojpuri treebank |
| Marathi | POS, NER, Sentiment | UD Marathi UFAL |
| Magahi | POS | UDAPI Magahi treebank |
| Sanskrit | POS, NER | Sanskrit Treebank |
| Urdu | POS, NER, Sentiment | UD Urdu UDTB |

---

## Pretraining

Set the backbone in `configs/default.py`:

```python
MODEL_ID   = "Qwen/Qwen3-1.7B"  # or Qwen/Qwen3-4B, google/gemma-3-4b-pt, google/gemma-3-12b-pt
HIDDEN_SIZE = 2560               # 2560 for 1.7B/4B, 3840 for 12B
PRETRAIN_SAVE_DIR = "runs_qwen/hnet_pretrain_pos_guided"
```

Run pretraining:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.pretrain
```

Key hyperparameters (from `configs/default.py`):

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Batch size | 2 × 4 grad accum |
| Max sequence length | 2048 bytes |
| Target boundary rate ρ* | 0.056 |
| λ_ratio | 0.8 |
| λ_align | 0.3 |
| λ_POS | 0.3 |
| POS supervision frequency | every 10 AR steps |
| Local Mamba layers | 1 |
| Unfrozen backbone layers | 1 (first + last) |

---

## Evaluation

### POS Tagging

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_pos \
    --checkpoint runs_qwen/hnet_pretrain_pos_guided/best.pt \
    --test_file data/bhojpuri_test.conllu
```

### NER

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_ner \
    --checkpoint runs_qwen/hnet_pretrain_pos_guided/best.pt \
    --test_file data/urdu_ner_test.conllu
```

### Sentiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_sentiment \
    --checkpoint runs_qwen/hnet_pretrain_pos_guided/best.pt \
    --lang marathi
```

### Subword Baselines

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen_subword_pos_baseline \
    --model Qwen/Qwen3-1.7B \
    --test_file data/bhojpuri_test.conllu
```

---

## Method

H-Net processes raw UTF-8 bytes through:

1. **Backbone-initialized byte embeddings** — each byte embedding initialized as the average of backbone subword embeddings containing that byte
2. **Local Mamba encoder** — contextualises byte representations (1 layer, d_state=16)
3. **Dynamic chunking** — router computes boundary probability p_t = 0.5(1 − cos(k_t, q_{t−1})); boundary placed when p_t ≥ τ
4. **Chunk alignment loss** — pulls chunk embeddings toward precomputed backbone subword targets (centroid fix for SentencePiece models)
5. **Task-guided pretraining** — POS supervision every 10 AR steps on Hindi UD treebank; head discarded after pretraining
6. **Frozen backbone LM** — processes word-level chunk embeddings (first + last layers unfrozen)

---

## Citation

```bibtex
@misc{kumar2026tokenizersfailbytelevelchunking,
      title={When Tokenizers Fail: Byte-Level Chunking for Zero-Shot Transfer to Low-Resource Languages}, 
      author={Sanjeev Kumar and Atsuki Yamaguchi and Nikolaos Aletras},
      year={2026},
      eprint={2608.27658},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.27658}, 
}
```

---

## License

MIT License. This code may not be used to train large language models or any commercial AI systems without explicit written permission from the authors.

---

*Work done during a visiting PhD at the University of Sheffield.*
