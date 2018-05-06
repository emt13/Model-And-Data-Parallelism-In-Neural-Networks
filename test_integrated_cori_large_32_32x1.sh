#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 00:30:00
#SBATCH -J bl-32x1

srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 32 > results/both/large/n32-32x1-32.out
srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 64 > results/both/large/n32-32x1-64.out
srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 128 > results/both/large/n32-32x1-128.out
srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 256 > results/both/large/n32-32x1-256.out
srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 512 > results/both/large/n32-32x1-512.out
srun -n 32 -N 32 python neuralnet.py 32 1 both large 10 100 0.000001 1024 > results/both/large/n32-32x1-1024.out
