#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 00:30:00
#SBATCH -J b-16x2

srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 32 > results/both/medium/n32-16x2-32.out
srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 64 > results/both/medium/n32-16x2-64.out
srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 128 > results/both/medium/n32-16x2-128.out
srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 256 > results/both/medium/n32-16x2-256.out
srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 512 > results/both/medium/n32-16x2-512.out
srun -n 32 -N 32 python neuralnet.py 16 2 both medium 10 100 0.000001 1024 > results/both/medium/n32-16x2-1024.out
