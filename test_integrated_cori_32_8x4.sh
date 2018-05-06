#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 00:30:00
#SBATCH -J b-8x4

srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 32 > results/both/medium/n32-8x4-32.out
srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 64 > results/both/medium/n32-8x4-64.out
srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 128 > results/both/medium/n32-8x4-128.out
srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 256 > results/both/medium/n32-8x4-256.out
srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 512 > results/both/medium/n32-8x4-512.out
srun -n 32 -N 32 python neuralnet.py 8 4 both medium 10 100 0.000001 1024 > results/both/medium/n32-8x4-1024.out
