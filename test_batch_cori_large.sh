#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -N 32
#SBATCH -t 01:00:00
#SBATCH -J auto-nn 

srun -n 1 -N 1 python neuralnet.py 1 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n1-large.out
srun -n 2 -N 2 python neuralnet.py 2 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n2-large.out
srun -n 4 -N 4  python neuralnet.py 4 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n4-large.out
srun -n 8 -N 8  python neuralnet.py 8 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n8-large.out
srun -n 16 -N 16  python neuralnet.py 16 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n16-large.out
srun -n 32 -N 32  python neuralnet.py 32 1 batch large 10 10000 0.00000001 32,32 > results/batch/large/n32-large.out

