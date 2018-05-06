#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 16
#SBATCH -t 00:30:00
#SBATCH -J bl-4x4

srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 32 > results/both/large/n16-4x4-32.out
srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 64 > results/both/large/n16-4x4-64.out
srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 128 > results/both/large/n16-4x4-128.out
srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 256 > results/both/large/n16-4x4-256.out
srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 512 > results/both/large/n16-4x4-512.out
srun -n 16 -N 16 python neuralnet.py 4 4 both large 10 10000 0.000001 1024 > results/both/large/n16-4x4-1024.out
