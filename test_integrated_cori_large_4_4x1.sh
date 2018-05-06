#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J bl-4-4x1

srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 32 > results/both/large/n4-4x1-32.out
srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 64 > results/both/large/n4-4x1-64.out
srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 128 > results/both/large/n4-4x1-128.out
srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 256 > results/both/large/n4-4x1-256.out
srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 512 > results/both/large/n4-4x1-512.out
srun -n 4 -N 4 python neuralnet.py 4 1 both large 10 100 0.000001 1024 > results/both/large/n4-4x1-1024.out
