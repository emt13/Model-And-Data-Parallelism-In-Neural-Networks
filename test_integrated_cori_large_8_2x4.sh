#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 8
#SBATCH -t 00:30:00
#SBATCH -J bl-8-2x4

srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 32 > results/both/large/n8-2x4-32.out
srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 64 > results/both/large/n8-2x4-64.out
srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 128 > results/both/large/n8-2x4-128.out
srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 256 > results/both/large/n8-2x4-256.out
srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 512 > results/both/large/n8-2x4-512.out
srun -n 8 -N 8 python neuralnet.py 2 4 both large 10 100 0.000001 1024 > results/both/large/n8-2x4-1024.out
