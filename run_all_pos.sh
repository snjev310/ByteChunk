# #!/bin/bash
# # run_all_pos.sh
# # Runs all POS tagging evaluations for H-Net and Qwen subword baseline
# # across 6 Indic languages: Hindi, Bhojpuri, Marathi, Sanskrit, Magahi, Urdu
# #
# # Usage: bash run_all_pos.sh
# # Logs: logs/pos_*.log

# set -e
# mkdir -p logs

# echo "========================================================"
# echo "Starting all POS evaluations: $(date)"
# echo "========================================================"

# # ── Download missing treebanks ────────────────────────────────────────────────
# echo ""
# echo "[0] Downloading treebanks..."

# mkdir -p data/ud_sanskrit data/ud_magahi data/ud_urdu data/ud_braj data/ud_nepali

# wget -q -P data/ud_sanskrit \
#   https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-UFAL/master/sa_ufal-ud-test.conllu \
#   2>/dev/null || echo "  Sanskrit: already exists or failed"

# wget -q -P data/ud_magahi \
#   https://raw.githubusercontent.com/UniversalDependencies/UD_Magahi-MGTB/dev/mag_mgtb-ud-test.conllu \
#   2>/dev/null || echo "  Magahi: already exists or failed"

# wget -q -P data/ud_urdu \
#   https://raw.githubusercontent.com/UniversalDependencies/UD_Urdu-UDTB/master/ur_udtb-ud-test.conllu \
#   2>/dev/null || echo "  Urdu: already exists or failed"

# # Verify all data files exist
# echo ""
# echo "Data files:"
# for f in \
#   data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#   data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
#   data/ud_marathi/mr_ufal-ud-train.conllu \
#   data/ud_sanskrit/sa_ufal-ud-test.conllu \
#   data/ud_magahi/mag_mgtb-ud-test.conllu \
#   data/ud_urdu/ur_udtb-ud-test.conllu; do
#   if [ -f "$f" ]; then
#     count=$(grep -c "^# sent_id" "$f" 2>/dev/null || echo "?")
#     echo "  ✓ $f ($count sentences)"
#   else
#     echo "  ✗ MISSING: $f"
#   fi
# done

# echo ""
# echo "========================================================"
# echo "Running evaluations in parallel across 4 GPUs"
# echo "========================================================"

# # ── GPU 0: H-Net — Bhojpuri + Marathi ─────────────────────────────────────────
# echo "[1] GPU 0: H-Net POS — Bhojpuri + Marathi"
# CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_chunk_pos \
#     --hnet_ckpt      runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
#     --data           data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho       data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
#     --test_mr        data/ud_marathi/mr_ufal-ud-train.conllu \
#     --save_dir       runs_qwen/hnet_chunk_pos_bho_mr \
#     --epochs         5 \
#     --n_local_layers 1 \
#     > logs/pos_hnet_bho_mr.log 2>&1 &
# PID1=$!
# echo "  PID: $PID1 | log: logs/pos_hnet_bho_mr.log"

# # ── GPU 1: H-Net — Sanskrit + Magahi ──────────────────────────────────────────
# echo "[2] GPU 1: H-Net POS — Sanskrit + Magahi"
# CUDA_VISIBLE_DEVICES=1 python -m scripts.eval_chunk_pos \
#     --hnet_ckpt      runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
#     --data           data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho       data/ud_sanskrit/sa_ufal-ud-test.conllu \
#     --test_mr        data/ud_magahi/mag_mgtb-ud-test.conllu \
#     --save_dir       runs_qwen/hnet_chunk_pos_sa_mag \
#     --epochs         5 \
#     --n_local_layers 1 \
#     > logs/pos_hnet_sa_mag.log 2>&1 &
# PID2=$!
# echo "  PID: $PID2 | log: logs/pos_hnet_sa_mag.log"

# # ── GPU 2: H-Net — Urdu ───────────────────────────────────────────────────────
# echo "[3] GPU 2: H-Net POS — Urdu"
# CUDA_VISIBLE_DEVICES=2 python -m scripts.eval_chunk_pos \
#     --hnet_ckpt      runs_qwen/hnet_pretrain_pos_guided/step_7000.pt \
#     --data           data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho       data/ud_urdu/ur_udtb-ud-test.conllu \
#     --save_dir       runs_qwen/hnet_chunk_pos_urdu \
#     --epochs         5 \
#     --n_local_layers 1 \
#     > logs/pos_hnet_urdu.log 2>&1 &
# PID3=$!
# echo "  PID: $PID3 | log: logs/pos_hnet_urdu.log"

# # ── GPU 3: Qwen — all languages ───────────────────────────────────────────────
# echo "[4] GPU 3: Qwen subword POS — all languages"
# CUDA_VISIBLE_DEVICES=3 python -m scripts.qwen_subword_pos_baseline \
#     --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_hindi data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho   data/ud_sanskrit/sa_ufal-ud-test.conllu \
#     --test_mr    data/ud_magahi/mag_mgtb-ud-test.conllu \
#     --save_dir   runs_qwen/qwen_pos_sa_mag \
#     --epochs     5 \
#     > logs/pos_qwen_sa_mag.log 2>&1 &
# PID4=$!
# echo "  PID: $PID4 | log: logs/pos_qwen_sa_mag.log"

# echo ""
# echo "All jobs launched. PIDs: $PID1 $PID2 $PID3 $PID4"
# echo "Monitor with: tail -f logs/pos_hnet_bho_mr.log"
# echo ""

# # ── Wait for GPU 0-3 to finish ────────────────────────────────────────────────
# echo "Waiting for GPU 0-3 jobs to finish..."
# wait $PID1 && echo "✓ GPU 0 (H-Net Bhojpuri+Marathi) done" || echo "✗ GPU 0 failed"
# wait $PID2 && echo "✓ GPU 1 (H-Net Sanskrit+Magahi) done"  || echo "✗ GPU 1 failed"
# wait $PID3 && echo "✓ GPU 2 (H-Net Urdu) done"             || echo "✗ GPU 2 failed"
# wait $PID4 && echo "✓ GPU 3 (Qwen Sanskrit+Magahi) done"   || echo "✗ GPU 3 failed"

# # ── Qwen Bhojpuri+Marathi+Urdu — reuse GPU 0 ─────────────────────────────────
# echo ""
# echo "[5] GPU 0: Qwen subword POS — Bhojpuri + Marathi"
# CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen_subword_pos_baseline \
#     --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_hindi data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho   data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
#     --test_mr    data/ud_marathi/mr_ufal-ud-train.conllu \
#     --save_dir   runs_qwen/qwen_pos_bho_mr \
#     --epochs     5 \
#     > logs/pos_qwen_bho_mr.log 2>&1 &
# PID5=$!

# echo "[6] GPU 1: Qwen subword POS — Urdu"
# CUDA_VISIBLE_DEVICES=1 python -m scripts.qwen_subword_pos_baseline \
#     --train_data data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_hindi data/ud_hindi_treebank/hi_hdtb-ud-train.conllu \
#     --test_bho   data/ud_urdu/ur_udtb-ud-test.conllu \
#     --save_dir   runs_qwen/qwen_pos_urdu \
#     --epochs     5 \
#     > logs/pos_qwen_urdu.log 2>&1 &
# PID6=$!

# echo "  PIDs: $PID5 $PID6"
# wait $PID5 && echo "✓ GPU 0 (Qwen Bhojpuri+Marathi) done" || echo "✗ GPU 0 failed"
# wait $PID6 && echo "✓ GPU 1 (Qwen Urdu) done"             || echo "✗ GPU 1 failed"

# # ── Print final summary ───────────────────────────────────────────────────────
# echo ""
# echo "========================================================"
# echo "ALL JOBS COMPLETE: $(date)"
# echo "========================================================"
# echo ""
# echo "Results summary:"
# echo ""

# python -c "
# import torch, os

# results = {}

# # H-Net results
# hnet_runs = {
#     'Bhojpuri': 'runs_qwen/hnet_chunk_pos_bho_mr',
#     'Marathi':  'runs_qwen/hnet_chunk_pos_bho_mr',
#     'Sanskrit': 'runs_qwen/hnet_chunk_pos_sa_mag',
#     'Magahi':   'runs_qwen/hnet_chunk_pos_sa_mag',
#     'Urdu':     'runs_qwen/hnet_chunk_pos_urdu',
# }
# for lang, run_dir in hnet_runs.items():
#     try:
#         r = torch.load(f'{run_dir}/results.pt', map_location='cpu')
#         for k, v in r.items():
#             if k != 'primary':
#                 results[f'hnet_{k}'] = v
#     except: pass

# # Qwen results
# qwen_runs = {
#     'bho_mr':  'runs_qwen/qwen_pos_bho_mr',
#     'sa_mag':  'runs_qwen/qwen_pos_sa_mag',
#     'urdu':    'runs_qwen/qwen_pos_urdu',
# }
# for key, run_dir in qwen_runs.items():
#     try:
#         r = torch.load(f'{run_dir}/results.pt', map_location='cpu')
#         for lang, acc in r.items():
#             if lang not in ['Hindi']:
#                 results[f'qwen_{lang}'] = acc
#     except: pass

# # Print table
# langs = ['Bhojpuri', 'Marathi', 'Sanskrit', 'Magahi', 'Urdu']
# print(f'{\"Language\":12s} | {\"H-Net ZS\":10s} | {\"H-Net XL\":10s} | {\"Qwen ZS\":10s} | {\"Qwen XL\":10s}')
# print('-'*60)
# for lang in langs:
#     hzs = results.get(f'hnet_{lang}_ZS', 0) * 100
#     hxl = results.get(f'hnet_{lang}_XL', 0) * 100
#     qzs = results.get(f'qwen_{lang}_ZS', 0) * 100
#     qxl = results.get(f'qwen_{lang}_XL', 0) * 100
#     print(f'{lang:12s} | {hzs:6.1f}%    | {hxl:6.1f}%    | {qzs:6.1f}%    | {qxl:6.1f}%')
# "

# echo ""
# echo "Full logs in: logs/"
# echo "Results in:   runs_qwen/hnet_chunk_pos_*/results.pt"


# XL Bhojpuri — train on 80% Bhojpuri, test on 20% Bhojpuri
CUDA_VISIBLE_DEVICES=0 python -m scripts.qwen_subword_pos_baseline \
    --train_data data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
    --test_hindi data/ud_bhojpuri/bho_bhtb-ud-test.conllu \
    --save_dir   runs_qwen/gemma_pos_xl_bho \
    --epochs     10 > logs/gemma_pos_xl_bho.log 2>&1 &

# XL Marathi
CUDA_VISIBLE_DEVICES=1 python -m scripts.qwen_subword_pos_baseline \
    --train_data data/ud_marathi/mr_ufal-ud-train.conllu \
    --test_hindi data/ud_marathi/mr_ufal-ud-train.conllu \
    --save_dir   runs_qwen/gemma_pos_xl_mr \
    --epochs     10 > logs/gemma_pos_xl_mr.log 2>&1 &

# XL Magahi
CUDA_VISIBLE_DEVICES=2 python -m scripts.qwen_subword_pos_baseline \
    --train_data data/ud_magahi/mag_mgtb-ud-test.conllu \
    --test_hindi data/ud_magahi/mag_mgtb-ud-test.conllu \
    --save_dir   runs_qwen/gemma_pos_xl_mag \
    --epochs     10 > logs/gemma_pos_xl_mag.log 2>&1 &

# XL Sanskrit
CUDA_VISIBLE_DEVICES=3 python -m scripts.qwen_subword_pos_baseline \
    --train_data data/ud_sanskrit/sa_ufal-ud-test.conllu \
    --test_hindi data/ud_sanskrit/sa_ufal-ud-test.conllu \
    --save_dir   runs_qwen/gemma_pos_xl_sa \
    --epochs     10 > logs/gemma_pos_xl_sa.log 2>&1 &

# XL Sanskrit
CUDA_VISIBLE_DEVICES=3 python -m scripts.qwen_subword_pos_baseline \
    --train_data data/ud_urdu/ur_udtb-ud-test.conllu \
    --test_hindi data/ud_urdu/ur_udtb-ud-test.conllu \
    --save_dir   runs_qwen/gemma_pos_xl_ur \
    --epochs     10 > logs/gemma_pos_xl_ur.log 2>&1 &