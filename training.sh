#!/usr/bin/env bash

nohup python classification.py --use_normals --use_uniform_sample --model PointNet > nohup/PointNet.out &
