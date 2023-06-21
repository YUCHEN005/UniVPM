#!/usr/bin/env bash

source activate CONDA_ENV

fairseq-hydra-train --config-dir ../conf/av-finetune --config-name large_clean_pt_clean_ft_433h.yaml

