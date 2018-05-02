#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 01:00:00
#SBATCH -J auto-nn 

srun -n 1 -N 1 python neuralnet.py 1 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n1-medium.out
srun -n 2 -N 2 python neuralnet.py 2 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n2-medium.out
srun -n 4 -N 4  python neuralnet.py 4 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n4-medium.out
srun -n 8 -N 8  python neuralnet.py 8 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n8-medium.out
srun -n 16 -N 16  python neuralnet.py 16 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n16-medium.out
srun -n 32 -N 32  python neuralnet.py 32 1 batch medium 10 10000 0.00000001 32,32 > results/batch/medium/n32-medium.out

