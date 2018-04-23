import numpy as np
from mpi4py import MPI
from batch_helper import scatter_data, all_reduce_data
from layers import l2_loss, fully_connected_layer

class NeuralNetwork:

        def __init__(self, nodes_model = 1, nodes_batch = 1):
                """
                Initialize the NeuralNetwork

                :param nodes_model: int, number of nodes to use for model parallelism 
                :param nodes_batch: int, number of nodes to use for batch parallelism
                """
                self.layers = []
                self.loss = None
                self.nodes_model = nodes_model
                self.nodes_batch = nodes_batch
                pass

        def add_layer(self, layer_type, size_input=0, size_output=0):
                """
                TODO: Add a layer to the NeuralNetwork. 
                """
                self.layers.append((layer_type, size_input, size_output))
        
        def add_loss(self, loss_function):
            self.loss = loss_function

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
                layers = []
                for layer in self.layers:
                    if layer[0] == "fc":
                        layers.append(fully_connected_layer(layer[1],layer[2]))
                    else:
                        return ("sorry one layer type is not valid")
                
                if self.loss == "l2":
                    loss = l2_loss()

                else : 
                    return ("sorry this loss is not valid")
                
                # for 1 ... epoch:
                for j in range(epochs):
                #   if rank is 0, shuffle
                    if(rank==0):
                        training_data = zip(list(x), list(y))
                        n = len(training_data)
                        np.random.shuffle(training_data)
                #       if rank is 0, break into minibatches
                        mini_batches =[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                        
                    else:
                        training_data = zip(list(x), list(y))
                        n = len(training_data)
                        #todo
                        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                        
                    
            #       for i in range(num_minibatches):
                    count_batch = 0
                    for mini_batch in mini_batches:
                        
                        all_x = np.array([i[0] for i in mini_batch])
                        #if rank == 0:
                            #print("mini batch: ", mini_batch)
                        all_y = np.array([i[1] for i in mini_batch])
            #           x = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        
                        x_rank = scatter_data(all_x, (len(all_x), len(all_x[0])) , comm, rank, size)
                        """
                        if rank == 0:
                            print("x_rank: ", x_rank)
                        """
            #           y = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        y_rank = scatter_data(all_y, (len(all_y), len(all_y[0])) , comm, rank, size)
            
            #           all_zs = [(x)]
                        all_zs = [x_rank]
            #           in = x
            #           for layer in layers:
                        count_layer = 0
                        for layer in layers:
            #               out = layer.forward(in)
                            z = layer.forward(x_rank)
            #               all_zs.append(out)
                            all_zs.append(z)
                            x_rank = z
                            
                            """
                            if rank== 0: 
                                print("rank: ", rank, "count_batch",count_batch, "count_layer: ", count_layer, "w: ",layer.w)
                            """
                            
                            count_layer +=1
                                  
            #           loss, dy = l2_loss(all_zs[-1], y) 
                        loss_value, dy = loss.loss(all_zs[-1], y_rank)
                        
                        """
                        if rank== 0: 
                            print("dy", dy)
                            print("loss", loss_value)
                        """
                        
            #           dws, dbs = [], []   
                        dws, dbs = [], []
            #           for layer in reversed(layers):
                        for layer in reversed(layers):
            #               dx, dw, db = layer.backwards(dy)
                            dx, dw, db = layer.backward(dy)
                            
                            """
                            #if rank== 0: 
                            #    print("dw: ", dw)
                            #    print("dx: ", dx)
                            """
                            
            #               dy = dx
                            dy = dx
            #               dws.append(dw)
                            dws.append(dw)
            #               dbs.append(db)
                            dbs.append(db)
                        
                        """
                        #count_batch+=1
                        """
                        
            #           allReduce(dws, size)
                        reduced_dws = all_reduce_data(dws, comm, rank, size)
                        
                        """
                        #if rank == 0:
                        #    print("reduced_dws: ", reduced_dws)
                        """
                        
            #           allReduce(dbs, size)
                        reduced_dbs = all_reduce_data(dbs, comm, rank, size)
                        
                        L = len(layers)
                        for i in range(L):
                            layer = layers[L-1-i]
                            layer.apply_gradient(reduced_dws[i],reduced_dbs[i], 0.1, mini_batch_size)
                                
                        
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


def _test_batch():
    full_batch_size = 10
    input_shape = 1
    output_shape = 1
    
    epochs = 3
    mini_batch_size = 2
    eta = 0.5
    
    x = np.random.randn(full_batch_size,input_shape)
    y = np.sin(x)
    
    nn = NeuralNetwork(nodes_model=1, nodes_batch=2)
    
    nn.add_layer("fc", input_shape, 7)
    nn.add_layer("fc",7, output_shape)
    nn.add_loss("l2")
    
    nn.train_batch_parallelism(x, y, epochs, mini_batch_size,eta)
    
    
    
if __name__=="__main__":
    _test_batch()    
