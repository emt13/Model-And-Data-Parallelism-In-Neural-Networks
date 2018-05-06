#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH -J bl-8-1x8

srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 32 > results/both/large/n8-1x8-32.out
srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 64 > results/both/large/n8-1x8-64.out
srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 128 > results/both/large/n8-1x8-128.out
srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 256 > results/both/large/n8-1x8-256.out
srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 512 > results/both/large/n8-1x8-512.out
srun -n 8 -N 8 python neuralnet.py 1 8 both large 10 100 0.000001 1024 > results/both/large/n8-1x8-1024.out
