#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 00:30:00
#SBATCH -J b-4x8

srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 32 > results/both/medium/n32-4x8-32.out
srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 64 > results/both/medium/n32-4x8-64.out
srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 128 > results/both/medium/n32-4x8-128.out
srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 256 > results/both/medium/n32-4x8-256.out
srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 512 > results/both/medium/n32-4x8-512.out
srun -n 32 -N 32 python neuralnet.py 4 8 both medium 10 100 0.000001 1024 > results/both/medium/n32-4x8-1024.out
