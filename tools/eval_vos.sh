#!/usr/bin/env bash

accelerate launch --multi_gpu --num_processes=4  tools/evaluate_vos.py ${@:1}