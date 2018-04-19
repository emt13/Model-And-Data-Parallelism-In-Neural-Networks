import numpy as np
import layers
from mpi4py import MPI
import test
from batch_helper import scatter_data

class NeuralNetwork:

        def __init__(self, nodes_model = 1, nodes_batch = 1):
                """
                Initialize the NeuralNetwork

                :param nodes_model: int, number of nodes to use for model parallelism 
                :param nodes_batch: int, number of nodes to use for batch parallelism
                """
                self.layers = []
                self.nodes_model = nodes_model
                self.nodes_batch = nodes_batch
                pass

        def add_layer(self, layer_type, size_input=0, size_output=0):
                """
                TODO: Add a layer to the NeuralNetwork. 
                """
                self.layers.append((layer_type, size_input, size_output))

        def train_model_parallelism(self, x, y):
                """
                TODO: Training procedure for model parallelism
                """
                pass

            
        def train_batch_parallelism(self, x, y, epochs, mini_batch_size, eta, test_data=None):
                """
                TODO: Training procedure for batch parallelism
                """
                if test_data: n_test = len(test_data)
                #To do
                training_data = TODO
                n = len(training_data)
                for j in range(epochs):
                    np.random.shuffle(training_data)
                    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    for mini_batch in mini_batches:
                        comm = MPI.COMM_WORLD
                        rank = comm.Get_rank()
                        size = comm.Get_size()
                        #To do
                        mini_batch_x_rank = scatter_data(mini_batch_x, (5,2), comm, rank, size)
                        #To do
                        mini_batch_y_rank = scatter_data(mini_batch_y, (5,1), comm, rank, size)
                     
                        mini_batch_out_rank = [mini_batch_x_rank]
                        
                        for layer in self.layers:
                            mini_batch_x_rank = layer.forward(mini_batch_x_rank)
                            mini_batch_out_rank.append(mini_batch_x_rank)
                                
                        (loss, mini_batch_dout_rank) = l2_loss(mini_batch_out_rank[-1] , mini_batch_y_rank )

                        for layer in reversed(self.layers):
                            # To Do
                            (mini_batch_dx_rank, mini_batch_dw_rank, mini_batch_db_rank) = layer.backward(mini_batch_dout_rank)
                            
                            mini_batch_dout_rank = mini_batch_dx_rank
                            
                            mini_batch_dw = AllReduce(mini_batch_dw_rank)
                            mini_batch_db = AllReduce(mini_batch_db_rank)
                            
                            layer.apply_gradient(mini_batch_dw, mini_batch_db)
                    
                    # ToDo
                    if test_data:
                        print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
                    else:
                        print ("Epoch {0} complete".format(j))
                            
                                
                pass

        def test(self, x):
                """
                TODO: Test procedure
                """
                pass

        def train(self, x, y):
                """
                TODO: Combined training procedure for model and batch parallelism
                """
                pass



def _test():
    nn = NeuralNetwork(nodes_model=8, nodes_batch=1)
    
    nn.add_layer("fc", 100, 90)
    nn.add_layer("l12")
    nn.add_layer("fc", 160, 20)
    nn.add_layer("l12")
    nn.add_layer("fc", 80, 50)
    nn.add_layer("softmax")
    print(nn.layers)  
    
    print("leo's work")
    x = np.random.rand(5, 2, 2)
    y = np.random.rand(5)
    a = test.scatter_data(x,y)
    print(a)


if __name__=="__main__":
    _test()    
