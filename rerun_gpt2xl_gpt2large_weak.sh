#!/bin/bash
#
# Rerun gpt2-xl experiments with gpt2-large weak labels
# Using the existing weak labels from the baseline run
#

set -e

# Path to existing gpt2-large weak labels
WEAK_LABELS_PATH="results/default/bs=32-dn=boolq-e=4-ee=100-lp=0-l=xent-l=1e-05-ls=cosi_anne-mc=1024-md=0.0-mxr=1.0-mxs=sample-ms=gpt2-large-nd=20000-ntd=10000-o=adam-s=0-twd=0/weak_labels"

# Configuration
DATASET="boolq"
STRONG_MODEL="gpt2-xl"
N_DOCS=20000
N_TEST_DOCS=10000
EVAL_EVERY=100
EPOCHS=4
BATCH_SIZE=32
LR=1e-05
RESULTS_FOLDER="./results_rerun"
SWEEP_SUBFOLDER="gpt2xl_gpt2large_clean"

echo "============================================================"
echo "Rerunning gpt2-xl with gpt2-large weak labels"
echo "============================================================"
echo "Strong model: $STRONG_MODEL"
echo "Weak labels from: gpt2-large"
echo "Dataset: $DATASET"
echo "Weak labels path: $WEAK_LABELS_PATH"
echo "Results folder: $RESULTS_FOLDER/$SWEEP_SUBFOLDER"
echo "============================================================"
echo ""

# Missing configs at mix_ratio=0.25:
# - gpt2-xl + gpt2-large @ 0.0
# - gpt2-xl + gpt2-large @ 0.25 (exists but seems wrong)
# - gpt2-xl + gpt2-large @ 0.75

# Run mix_ratio=0.0
echo "Running mix_ratio=0.0..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_simple.py \
    --model_size="$STRONG_MODEL" \
    --ds_name="$DATASET" \
    --weak_labels_path="$WEAK_LABELS_PATH" \
    --mix_ratio=0.0 \
    --mix_strategy=sample \
    --n_docs=$N_DOCS \
    --n_test_docs=$N_TEST_DOCS \
    --eval_every=$EVAL_EVERY \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --results_folder="$RESULTS_FOLDER" \
    --sweep_subfolder="$SWEEP_SUBFOLDER" \
    --force_retrain

echo ""
echo "Running mix_ratio=0.25..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_simple.py \
    --model_size="$STRONG_MODEL" \
    --ds_name="$DATASET" \
    --weak_labels_path="$WEAK_LABELS_PATH" \
    --mix_ratio=0.25 \
    --mix_strategy=sample \
    --n_docs=$N_DOCS \
    --n_test_docs=$N_TEST_DOCS \
    --eval_every=$EVAL_EVERY \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --results_folder="$RESULTS_FOLDER" \
    --sweep_subfolder="$SWEEP_SUBFOLDER" \
    --force_retrain

echo ""
echo "Running mix_ratio=0.5..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_simple.py \
    --model_size="$STRONG_MODEL" \
    --ds_name="$DATASET" \
    --weak_labels_path="$WEAK_LABELS_PATH" \
    --mix_ratio=0.5 \
    --mix_strategy=sample \
    --n_docs=$N_DOCS \
    --n_test_docs=$N_TEST_DOCS \
    --eval_every=$EVAL_EVERY \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --results_folder="$RESULTS_FOLDER" \
    --sweep_subfolder="$SWEEP_SUBFOLDER" \
    --force_retrain

echo ""
echo "Running mix_ratio=0.75..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_simple.py \
    --model_size="$STRONG_MODEL" \
    --ds_name="$DATASET" \
    --weak_labels_path="$WEAK_LABELS_PATH" \
    --mix_ratio=0.75 \
    --mix_strategy=sample \
    --n_docs=$N_DOCS \
    --n_test_docs=$N_TEST_DOCS \
    --eval_every=$EVAL_EVERY \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --lr=$LR \
    --results_folder="$RESULTS_FOLDER" \
    --sweep_subfolder="$SWEEP_SUBFOLDER" \
    --force_retrain

echo ""
echo "============================================================"
echo "All runs complete!"
echo "Results saved to: $RESULTS_FOLDER/$SWEEP_SUBFOLDER/"
echo "============================================================"
