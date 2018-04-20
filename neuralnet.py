import numpy as np
from mpi4py import MPI
from batch_helper import scatter_data, all_reduce_data
from layers import l2_loss

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
                
                # mpi init
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                # create the layers themselves
                layers = self.layers
                
                # for 1 ... epoch:
                for j in range(epochs):
                #   if rank is 0, shuffle
                    if(rank==0):
                        training_data = list(zip(x, y))
                        n = len(training_data)
                        np.random.shuffle(training_data)
                #       if rank is 0, break into minibatches
                        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                        


            #       for i in range(num_minibatches):
                    for mini_batch in mini_batches:
                        all_x = mini_batch[:,0]
                        all_y = mini_batch[:,1]
            #           x = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        x = scatter_data(all_x, (len(all_x), len(all_x[0])) , comm, rank, size)
            #           y = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        y = scatter_data(all_y, (len(all_y), len(all_y[0])) , comm, rank, size)
            
            #           all_zs = [(x)]
                        all_zs = [x]
            #           in = x
            #           for layer in layers:
                        for layer in layers:
            #               out = layer.forward(in)
                            z = layer.append(x)
            #               all_zs.append(out)
                            all_zs.append(z)
                            x = z
            #                       
            #           loss, dy = l2_loss(all_zs[-1], y) 
                        loss, dy = l2_loss(all_zs[-1], y) 
            #           dws, dbs = [], []   
                        dws, dbs = [], []
            #           for layer in reversed(layers):
                        for layer in reversed(layers):
            #               dx, dw, db = layer.backwards(dy)
                            dx, dw, db = layer.backwards(dy)
            #               dy = dx
                            dy = dx
            #               dws.append(dw)
                            dws.append(dw)
            #               dbs.append(db)
                            dws.append(dw)
            #           allReduce(dws, size)
                        reduced_dws = all_reduce_data(dws)
            #           allReduce(dbs, size)
                        reduced_dbs = all_reduce_data(dbs)
                        
                        
                        for i in range(len(layers)):
                            layer = layers[i]
                            layer.apply_gradient(reduced_dws[i],reduced_dbs[i])
                            
                    print ("Epoch {0} complete".format(j))
                


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

if __name__=="__main__":
    _test()    
