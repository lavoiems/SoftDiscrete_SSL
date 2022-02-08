#!/bin/bash
MODEL=vq-byol
DATASET=dsprites
DATASET_K=2
EXPNAME=$MODEL-$DATASET-$DATASET_K

python src/main.py \
  --exp_name=$EXPNAME \
  --reload \
  --log_every=10 \
  --eval_every=100 \
  $MODEL \
  --seed=100 \
  --max_steps=100 \
  --dataset=$DATASET \
  --dataset_K=$DATASET_K \
  --n_train=45000 \
  --n_sub_train=45000 \
  --train_batch_size=256 \
  --lr=1e-3 \
  --wd=1e-4 \
  --hidden_size=256 \
  --ema=0.99 \
  --projector_type=MLP \
  --last_bn=True \
  --embedder_type=Linear \
  --encode_method=VQ \
  --hard=False \
  --channel_tau=1. \
  --vq_dimz=8 \
  --vq_beta=0.25 \
  --vq_warmup_steps=0 \
  --vq_update_rule=ema \
  --vq_ema=0.85 \
  --predictor_type=Linear \
  --voc_size=32 \
  --message_size=256
