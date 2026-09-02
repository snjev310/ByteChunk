# configs/default.py
"""
Central configuration for H-Net.
All scripts import from here — no magic numbers scattered in code.
"""
import torch

N_LOCAL_LAYERS = 1
# Reproducibility

SEED = 42

# Backbone
MODEL_ID ="Qwen/Qwen3-1.7B"#"Qwen/Qwen3-4B"#"google/gemma-3-4b-pt"# #"google/gemma-3-12b-pt" #"Qwen/Qwen3-4B"#"google/gemma-3-4b-pt"
HIDDEN_SIZE = 2560 #3840 #2560 #1536 #4096          # LLaMA hidden dim — also byte embedding dim'
BYTE_VOCAB_SIZE = 256

# Data / Batching
MAX_LENGTH        = 2048
BATCH_SIZE        = 2
GRAD_ACCUM_STEPS  = 4
NUM_EPOCHS        = 1

# Optimizer
LR             = 5e-5
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 0.5
WARMUP_RATIO   = 0.05

# H-Net dynamic chunking
TARGET_RATIO  = 0.056    #Hindi/Devanagari: 3 bytes/char → 0.056 gives ~18 bytes = ~6 chars = ~1 word#0.125    # desired boundary rate  (1 boundary per 8 bytes)
LAMBDA_RATIO  = 0.8      # ratio loss weight  (original 5.0 was too aggressive)
BOUNDARY_THRESH = 0.3    # hard boundary threshold
LAMBDA_ALIGN = 0.0 #0.3 #0.0 #0.01     # alignment loss weight 
N_LOCAL_LAYERS     = 1
N_UNFREEZE_LAYERS  = 1

# LoRA (pretraining)
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# LoRA (fine-tuning — lighter)
FT_LORA_R       = 8
FT_LORA_ALPHA   = 32
FT_LORA_DROPOUT = 0.05

# Task-specific
# MT
MT_MAX_SRC_LEN    = 512
MT_MAX_TGT_LEN    = 512
MT_NUM_EPOCHS     = 10
MT_LR             = 1e-4

# POS tagging
POS_NUM_EPOCHS    = 20
POS_LR            = 2e-4

# Sentiment
SENTIMENT_NUM_EPOCHS = 10
SENTIMENT_LR         = 2e-4
SENTIMENT_POOL       = "mean"   # "mean" | "cls" | "last"

# Checkpoints
# PRETRAIN_SAVE_DIR = "runs_qwen3_4b/hnet_pretrain_pos_guided"
MT_SAVE_DIR       = "runs_gemma/hnet_mt"
POS_SAVE_DIR      = "runs_gemma/hnet_pos"
SENTIMENT_SAVE_DIR = "runs_gemma/hnet_sentiment"

# Runtime
DTYPE  = torch.bfloat16
# DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenization
EOS_ID = 1
PAD_ID = 0

# Task-guided pretraining
LAMBDA_POS        = 0.3
POS_BATCH_FREQ    = 10
POS_CONLL_PATH    = "data/ud_hindi_treebank/hi_hdtb-ud-train.conllu"
PRETRAIN_SAVE_DIR = "runs_cpt/Qwen_Qwen3-1.7B"#"runs_qwen/hnet_pretrain_no_align"#"runs_qwen3_4b/hnet_pretrain_pos_guided"

