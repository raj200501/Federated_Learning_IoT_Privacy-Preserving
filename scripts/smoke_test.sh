#!/usr/bin/env bash
set -euo pipefail

python scripts/run_federated_demo.py \
  --num-devices 5 \
  --num-samples 50 \
  --num-sensors 6 \
  --num-clients 2 \
  --num-rounds 1
