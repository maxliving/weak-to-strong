#!/bin/bash
#
# Regenerate gpt2-large weak labels by retraining the baseline
# This will overwrite the corrupted weak labels
#

set -e

echo "============================================================"
echo "Regenerating gpt2-large weak labels (retraining baseline)"
echo "============================================================"
echo ""

CUDA_VISIBLE_DEVICES=4,5 python train_simple.py \
    --model_size=gpt2-large \
    --ds_name=boolq \
    --n_docs=20000 \
    --n_test_docs=10000 \
    --eval_every=100 \
    --epochs=4 \
    --batch_size=32 \
    --lr=1e-05 \
    --mix_ratio=1.0 \
    --force_retrain

echo ""
echo "============================================================"
echo "Done! Weak labels regenerated at:"
echo "results/default/bs=32-dn=boolq-e=4-ee=100-lp=0-l=xent-l=1e-05-ls=cosi_anne-mc=1024-md=0.0-mxr=1.0-mxs=sample-ms=gpt2-large-nd=20000-ntd=10000-o=adam-s=0-twd=0/weak_labels"
echo "============================================================"
