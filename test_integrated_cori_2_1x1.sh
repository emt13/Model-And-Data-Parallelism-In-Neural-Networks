#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J b-2-1x1

srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 32 > results/both/medium/n2-1x1-32.out
srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 64 > results/both/medium/n2-1x1-64.out
srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 128 > results/both/medium/n2-1x1-128.out
srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 256 > results/both/medium/n2-1x1-256.out
srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 512 > results/both/medium/n2-1x1-512.out
srun -n 2 -N 2 python neuralnet.py 1 1 both medium 10 100 0.000001 1024 > results/both/medium/n2-1x1-1024.out
