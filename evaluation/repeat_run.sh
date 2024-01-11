#!/bin/bash

for i in {1..500}; do
  timeout 6m python evaluate.cpython-39.pyc configs/jason_lbg_eval_cfg.yaml
  sleep 1
done