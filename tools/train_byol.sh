#!/usr/bin/env bash

accelerate launch --multi_gpu --num_processes 4 tools/train_byol.py ${@:1}