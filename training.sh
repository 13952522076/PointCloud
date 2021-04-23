#!/usr/bin/env bash

nohup python classification.py --use_normals --use_uniform_sample --model PointNet2SSG > nohup/PointNet2SSG.out &

nohup python classification.py --use_normals --use_uniform_sample --model PointTransformer > nohup/PointTransformer.out &

nohup python classification.py --use_normals --use_uniform_sample --model Simpler2Amax > nohup/Simpler2Amax.out &

nohup python classification.py --use_normals --use_uniform_sample --model Simpler2A > nohup/Simpler2A.out &

nohup python classification.py --use_uniform_sample --model PCT > nohup/PCT.out &

nohup python classification.py --use_normals --use_uniform_sample --model MLP_max > nohup/MLP_max.out &

nohup python classification.py --use_normals --use_uniform_sample --model MLP2_max > nohup/MLP2_max.out &

nohup python classification.py --use_normals --use_uniform_sample --model develop1Amax > nohup/develop1Amax.out &

nohup python classification.py --use_normals --use_uniform_sample --model develop2Amax > nohup/develop2Amax.out &

nohup python classification.py --use_normals --use_uniform_sample --model develop3Amax > nohup/develop3Amax.out &

nohup python classification.py --use_normals --use_uniform_sample --model develop4Amax > nohup/develop4Amax.out &

nohup python classification.py --use_normals --use_uniform_sample --model develop5Amax > nohup/develop5Amax.out &





#nohup python classification.py --use_normals --use_uniform_sample --model PointNet > nohup/PointNet.out &
