#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 16
#SBATCH -t 00:30:00
#SBATCH -J b-16-2x8

srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 32 > results/both/medium/n16-2x8-32.out
srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 64 > results/both/medium/n16-2x8-64.out
srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 128 > results/both/medium/n16-2x8-128.out
srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 256 > results/both/medium/n16-2x8-256.out
srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 512 > results/both/medium/n16-2x8-512.out
srun -n 16 -N 16 python neuralnet.py 2 8 both medium 10 100 0.000001 1024 > results/both/medium/n16-2x8-1024.out
