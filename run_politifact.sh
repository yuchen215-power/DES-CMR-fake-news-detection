#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-./formatted_data/declare/}"

if "$PYTHON_BIN" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  CUDA_FLAG=1
else
  CUDA_FLAG=0
fi

COMMON_ARGS=(
  --dataset=PolitiFact
  --cuda="$CUDA_FLAG"
  --fixed_length_left=30
  --fixed_length_right=100
  --log=logs/getral
  --loss_type=cross_entropy
  --batch_size=32
  --num_folds=5
  --use_claim_source=0
  --use_article_source=1
  --path="$DATA_ROOT"
  --hidden_size=300
  --epochs=100
  --num_att_heads_for_words=5
  --num_att_heads_for_evds=2
  --gnn_window_size=3
  --lr=0.0001
  --gnn_dropout=0.2
  --seed=123656
  --alpha=0.5
  --gsl_rate=0.7
  --kernel_number=11
  --topk=15
  --mask_rate=0.2
  --consistency_lambda=0.0
)

MODELS=(
  # MasterFC/master_getral_origin.py
  # MasterFC/master_getral_prompt.py
  # MasterFC/master_getral_prompt_dy.py
  MasterFC/master_getral_prompt_dy_topk15_entropy.py
)

for model in "${MODELS[@]}"; do
  "$PYTHON_BIN" "$model" "${COMMON_ARGS[@]}"
done
