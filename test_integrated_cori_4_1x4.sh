#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J b-4-1x4

srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 32 > results/both/medium/n4-1x4-32.out
srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 64 > results/both/medium/n4-1x4-64.out
srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 128 > results/both/medium/n4-1x4-128.out
srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 256 > results/both/medium/n4-1x4-256.out
srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 512 > results/both/medium/n4-1x4-512.out
srun -n 4 -N 4 python neuralnet.py 1 4 both medium 10 100 0.000001 1024 > results/both/medium/n4-1x4-1024.out
