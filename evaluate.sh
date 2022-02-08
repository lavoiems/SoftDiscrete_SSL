#!/bin/bash
python -u src/evaluate.py \
  --root-path ./test-eval \
  --exp-name vq-byol-testing \
  vq-byol linear_probe \
  --data-path $DATA_PATH \
  --run-path $@
