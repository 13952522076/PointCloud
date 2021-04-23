#!/usr/bin/env bash

printf "\n\n\n\n Training PointNet2SSG\n"
python classification.py --use_normals --use_uniform_sample --model PointNet2SSG

printf "\n\n\n\n Training PointTransformer\n"
python classification.py --use_normals --use_uniform_sample --model PointTransformer

printf "\n\n\n\n Training Simpler2Amax\n"
python classification.py --use_normals --use_uniform_sample --model Simpler2Amax

printf "\n\n\n\n Training Simpler2A\n"
python classification.py --use_normals --use_uniform_sample --model Simpler2A

printf "\n\n\n\n Training PCT\n"
python classification.py --use_uniform_sample --model PCT

printf "\n\n\n\n Training MLP_max\n"
python classification.py --use_normals --use_uniform_sample --model MLP_max

printf "\n\n\n\n Training MLP2_max\n"
python classification.py --use_normals --use_uniform_sample --model MLP2_max

printf "\n\n\n\n Training develop1Amax\n"
python classification.py --use_normals --use_uniform_sample --model develop1Amax

printf "\n\n\n\n Training develop2Amax\n"
python classification.py --use_normals --use_uniform_sample --model develop2Amax

printf "\n\n\n\n Training develop3Amax\n"
python classification.py --use_normals --use_uniform_sample --model develop3Amax

printf "\n\n\n\n Training develop4Amax\n"
python classification.py --use_normals --use_uniform_sample --model develop4Amax

printf "\n\n\n\n Training develop5Amax\n"
python classification.py --use_normals --use_uniform_sample --model develop5Amax





#nohup python classification.py --use_normals --use_uniform_sample --model PointNet > nohup/PointNet.out &
