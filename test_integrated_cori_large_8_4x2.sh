#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH -J bl-8-4x2

srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 32 > results/both/large/n8-4x2-32.out
srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 64 > results/both/large/n8-4x2-64.out
srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 128 > results/both/large/n8-4x2-128.out
srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 256 > results/both/large/n8-4x2-256.out
srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 512 > results/both/large/n8-4x2-512.out
srun -n 8 -N 8 python neuralnet.py 4 2 both large 10 100 0.000001 1024 > results/both/large/n8-4x2-1024.out
