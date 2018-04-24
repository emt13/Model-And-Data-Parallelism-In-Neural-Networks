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
                Add a layer to the NeuralNetwork. 
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
                Training procedure for batch parallelism
                """
                # mpi init
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                # Create the layers themselves
                layers = []
                seed = 0
                for layer in self.layers:
                    if layer[0] == "fc":
                        # Important note here: every layer is initialized on each process.
                        # The initialization is random so: either we broadcast the weights and biases
                        # Or we add a seed in layer. For now, we chose the latest.
                        layers.append(fully_connected_layer(layer[1],layer[2],seed))
                    else:
                        return ("sorry at least, one layer type is not valid")
                    seed += 1
                
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
                    
                    # The following mini_batches are dummies. There are needed to go trhough the following for loop on each process
                    # There might be a better way to do it though.
                    else:
                        training_data = zip(list(x), list(y))
                        n = len(training_data)
                        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    
                    """    
                    if rank == 0 :    
                        print("mini_batches on rank 0", mini_batches)
                    """
                    
            #       for i in range(num_minibatches):
                    # count_batch = 0 (for debugging purpose)
                    for mini_batch in mini_batches:
                        
                        all_x = np.array([i[0] for i in mini_batch])
                        all_y = np.array([i[1] for i in mini_batch])
                        
                        """
                        if rank == 1:
                            print("all_x dims: ", all_x.shape[0])
                        """
                        
            #           x = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        x_rank = scatter_data(all_x, (all_x.shape[0], all_x.shape[1]) , comm, rank, size)
                
                #       y = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        y_rank = scatter_data(all_y, (all_y.shape[0], all_y.shape[1]) , comm, rank, size)
                        
                        # The following if statement solves the problem when there is one single data to scatter on 2 processes. The second process will receive an empty data...
                        if x_rank.size != 0:                                              
                #           all_zs = [(x)]
                            all_zs = [x_rank]
                #           in = x
                #           for layer in layers:
                            # count_layer = 0 (for dubugging purpose)
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
                                
                                # count_layer +=1
                                  
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
                            
                            
                            #count_batch+=1
                        #End if x_rank.size != 0:
                        else:    
                            dws, dbs = [], []
                            for layer in reversed(layers):
                                dw = np.zeros(layer.w.shape)
                                db = np.zeros(layer.b.shape)
                                dws.append(dw)
                                dbs.append(db)
                        
            #           allReduce(dws, size)
                        reduced_dws = all_reduce_data(dws, comm, rank, size)
                        
            #           allReduce(dbs, size)
                        reduced_dbs = all_reduce_data(dbs, comm, rank, size)
                        
                        """
                        #if rank == 0:
                        #    print("reduced_dws: ", reduced_dws)
                        """
                        
                        L = len(layers)

                        for i in range(L):
                            layer = layers[L-1-i]
                            layer.apply_gradient(reduced_dws[i],reduced_dbs[i], eta, len(mini_batch))
                    
                    """
                    if rank == 0:
                        print("weights first layer", layers[0].w)
                    """
                    if rank == 0:
                        if test_data:
                            print ("Epoch {0}/{1} complete - loss: {2}".format(j+1, epochs, self.evaluate(test_data, layers, loss)))
                        else:
                            print ("Epoch {0}/{1} complete".format(j, epochs))
                
        def evaluate(self, test_data, layers, loss):
            test_results = [(self.feedforward(np.array([x_test]), layers), y_test)
                            for (x_test, y_test) in test_data]
            n_test = len(test_data)
            return (1.0/(1.0*n_test) * sum(loss.loss(y_predicted, y_truth)[0] for (y_predicted, y_truth) in test_results))
        
        def feedforward(self, a, layers):
            """Return the output of the network if ``a`` is input."""
            for layer in layers:
                a = layer.forward(a)
            return a


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
    full_batch_size_train = 20
    full_batch_size_test = 8
    input_shape = 2
    output_shape = 1
    epochs = 10
    mini_batch_size = 2
    eta = 0.01
    
    # This seed is necessary to initialize x so that the optimization converges
    # Otherwise the network will diverge to inf
    np.random.seed(3753934041)
    x_train = np.random.randn(full_batch_size_train,input_shape)
    y_train = np.transpose([np.sin(x_train[:,0])])
    
    x_test = np.random.randn(full_batch_size_test,input_shape)
    y_test = np.transpose([np.sin(x_test[:,0])])
    
    test_data = zip(list(x_test), list(y_test))
    
    # We don't really care about nodes_model and nodes_batch for now 
    nn = NeuralNetwork(nodes_model=1, nodes_batch=2)
    
    nn.add_layer("fc", input_shape, 7)
    nn.add_layer("fc",7, output_shape)
    nn.add_loss("l2")
    
    nn.train_batch_parallelism(x_train, y_train, epochs, mini_batch_size,eta, test_data = test_data)
    
    
    
if __name__=="__main__":
    _test_batch()    
