#!/usr/bin/env bash

source activate CONDA_ENV

python -B ../infer_s2s.py --config-dir ../conf/av-finetune --config-name large_clean_pt_clean_ft_433h_clean_dec.yaml

