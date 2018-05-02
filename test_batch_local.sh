# executes all of the tests for batch

# <number of nodes for batch> 
# <number of nodes for model> 
# <dataset> 
# <num epochs> 
# <mini batch size> 
# <eta> 
# <csv style neurons for each layer>

mpiexec -n 1 python neuralnet.py 1 1 batch toy 10 10 0.0001 8,8 > n1-toy-batch.out 
mpiexec -n 2 python neuralnet.py 2 1 batch toy 10 10 0.0001 8,8 > n2-toy-batch.out 

mpiexec -n 1 python neuralnet.py 1 1 batch medium 10 100 0.0001 8,8 > n1-medium-batch.out 
mpiexec -n 2 python neuralnet.py 2 1 batch medium 10 100 0.0001 8,8 > n2-medium-batch.out 

mpiexec -n 1 python neuralnet.py 1 1 batch large 10 10000 0.0001 8,8 > n1-large-batch.out 
mpiexec -n 2 python neuralnet.py 2 1 batch large 10 10000 0.0001 8,8 > n2-large-batch.out 

