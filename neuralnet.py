import numpy as np
from mpi4py import MPI
from batch_helper import scatter_data, all_reduce_data
from layers import l2_loss, fully_connected_layer, softmax_loss

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

        def train_model_parallelism(self, x, y, epochs, mini_batch_size, eta, test_data=None):
                """
                TODO: Training procedure for model parallelism
                """
                x_shape = x.shape
                y_shape = y.shape

                mini_batch_shapes = [len(x[k:k + mini_batch_size]) for k in range(0, len(x), mini_batch_size)]

                # mpi init
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                #TODO
                # Create the layers themselves
                layers, loss = self._init_layers()
                
                for e in range(epochs):
                    training_data = list(zip(list(x), list(y)))
                    n = len(training_data)
                    np.random.shuffle(training_data)
                    mini_batches =[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    
                    for i in range(len(mini_batch_shapes)):
                        # Naming convention
                        # variable_all means that variable is common to all processes
                        # all_variable referes to a lsit of variables from each layer
                        x_all = np.array([j[0] for j in mini_batches[i]])
                        y_all = np.array([j[1] for j in mini_batches[i]])
                        all_zs_reduced = [x_all]
                        
                        for layer in layers:
                            z_rank = layer.forward(x_all)
                            z_reduced = np.vstack(all_reduce_data(z_rank, comm, rank, size))
                            all_zs_reduced.append(z_reduced)
                            x_all = z_reduced
                        
                        loss_value, dy = loss.loss(all_zs_reduced[-1], y_all)
                        
                        
                        for layer in reversed(layers):
                            #TODO
                            dx_rank, dw_rank, db_rank = layer.backward(dy)
                            dx_reduced = np.vstack(all_reduce_data(dx_rank, comm, rank, size))
                            dy = dx_reduced
                            
                            #TODO 
                            layer.apply_gradient(dw_rank, db_rank, eta, mini_batch_shapes[i])
                            
                    if test_data:
                        print ("Epoch {0}/{1} complete - loss: {2}".format(e+1, epochs, self.evaluate(test_data, layers, loss)))
                    else:
                        print ("Epoch {0}/{1} complete".format(e+1, epochs))
                                
                            

        def train_batch_parallelism(self, x, y, epochs, mini_batch_size, eta, test_data=None):
                """
                Training procedure for batch parallelism
                """
                x_shape = x.shape
                y_shape = y.shape
                #print("x:",x_shape)
                #print("y:",y_shape)

                mini_batch_shapes = [len(x[k:k + mini_batch_size]) for k in range(0, len(x), mini_batch_size)]
                #print("mbs:", mini_batch_shapes, len(mini_batch_shapes))
                 

                # mpi init
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()
                
                # Create the layers themselves
                layers, loss = self._init_layers()
                #print(layers)
                #print("*", rank, "*", mini_batch_size, len(x))
                
                   
                               
                # for 1 ... epoch:
                for e in range(epochs):
                #   if rank is 0, shuffle
                    if(rank==0):
                        #print("\n\n ---- EPOCH", e)
                        training_data = list(zip(list(x), list(y)))
                        n = len(training_data)
                        np.random.shuffle(training_data)
                #       if rank is 0, break into minibatches
                        mini_batches =[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    
                    # The following mini_batches are dummies. There are needed to go trhough the following for loop on each process
                    # There might be a better way to do it though.
                    #else:
                    #    training_data = list(zip(list(x), list(y)))
                    #    n = len(training_data)
                    #    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
                    
                    #print("*", rank, "*", mini_batches)
                   
            #       for i in range(num_minibatches):
                    # count_batch = 0 (for debugging purpose)
                    #for mini_batch in mini_batches:
                    for i in range(len(mini_batch_shapes)):
                        #print("     - mini:", i)
                        
                        all_x = None
                        all_y = None
                        if rank == 0:
                            all_x = np.array([j[0] for j in mini_batches[i]])
                            all_y = np.array([j[1] for j in mini_batches[i]])
                            #print(" -- all_x:", all_x)
                            #print(" -- all_y:", all_y)
                        
                        """
                        if rank == 1:
                            print("all_x dims: ", all_x.shape[0])
                        """
                        
            #           x = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        x_rank = scatter_data(all_x, (mini_batch_shapes[i], x_shape[1]) , comm, rank, size)
                        #print("*", rank, "* x", x_rank, mini_batch_shapes[i], x_shape[1])
                
                #       y = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                        y_rank = scatter_data(all_y, (mini_batch_shapes[i], y_shape[1]) , comm, rank, size)
                        #print("*", rank, "* y", y_rank, mini_batch_shapes[i], x_shape[1])

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
                                
                            #if rank== 0: 
                            #    print("rank: ", rank, "count_batch", count_batch, "count_layer: ", count_layer, "w: ",layer.w)
                                
                                # count_layer +=1
                                  
                #           loss, dy = l2_loss(all_zs[-1], y) 
                            loss_value, dy = loss.loss(all_zs[-1], y_rank)
                            #print("mb:", i, "loss:", loss_value, all_zs[-1]) 
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
                        
                        #print(rank, "rdws", reduced_dws)
                        
            #           allReduce(dbs, size)
                        reduced_dbs = all_reduce_data(dbs, comm, rank, size)
                        
                        """
                        #if rank == 0:
                        #    print("reduced_dws: ", reduced_dws)
                        """
                        
                        L = len(layers)

                        for j in range(L):
                            layer = layers[L-1-j]
                            layer.apply_gradient(reduced_dws[j],reduced_dbs[j], eta, mini_batch_shapes[i])
                    
                    """
                    if rank == 0:
                        print("weights first layer", layers[0].w)
                    """
                    if rank == 0:
                        '''
                        count = 1
                        print("number of layers:", len(layers))
                        for l in layers:
                            print(" -- ", count, " --")
                            print(rank, l.w)
                            print("++")
                            print()
                            count += 1
                        '''
                        if test_data:
                            print ("Epoch {0}/{1} complete - loss: {2}".format(e+1, epochs, self.evaluate(test_data, layers, loss)))
                        else:
                            print ("Epoch {0}/{1} complete".format(e+1, epochs))
                
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

        def train_serial(self, x, y):
            layers, loss = self.init_layers() 
            
                

        def _init_layers(self, seed=0):
            layers = []
            loss = None 
            seed = 0
            for layer in self.layers:
                if layer[0] == "fc":
                    # Important note here: every layer is initialized on each process.
                    # The initialization is random so: either we broadcast the weights and biases
                    # Or we add a seed in layer. For now, we chose the latter.
                    layers.append(fully_connected_layer(layer[1],layer[2],seed))
                else:
                    print(layer[0], "is not valid")
                    return []
                seed += 1
            if self.loss == "l2":
                loss = l2_loss()
            elif self.loss == "softmax":
                loss = softmax_loss()
            else : 
                print("invalid loss layer")
                return []
            return layers, loss
             


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
    
    # UCI airfoil test
    data = np.loadtxt(open("Data/airfoil_self_noise.dat", "rb"), delimiter="\t")    
    x = data[:,1:5]
    y = data[:,5].reshape(len(x), 1)

    x_train = x[:1200]
    y_train = y[:1200]
    x_test = x[1200:]
    y_test = y[1200:]
    print(x.shape)
    print(y.reshape(len(y), 1).shape)

    test_data = list(zip(list(x_test), list(y_test)))
    
    #full_batch_size_train = 20
    #full_batch_size_test = 8
    input_shape = x.shape[1]
    output_shape = y.shape[1]
    epochs = 100
    mini_batch_size = 128
    eta = 0.00000000011
    

    
    '''
    # sinoid dataset 
    full_batch_size_train = 20
    full_batch_size_test = 8
    input_shape = 2
    output_shape = 1
    epochs = 1000
    mini_batch_size = 2
    eta = 0.001
    
 
    # This seed is necessary to initialize x so that the optimization converges
    # Otherwise the network will diverge to inf
    #np.random.seed(3753934041)
    #np.random.seed(3)
    x_train = np.random.randn(full_batch_size_train,input_shape)
    y_train = np.transpose([np.sin(x_train[:,0])])
    
    x_test = np.random.randn(full_batch_size_test,input_shape)
    y_test = np.transpose([np.sin(x_test[:,0])])
    
    test_data = list(zip(list(x_test), list(y_test)))
    '''
    ''' 
    # toy dataset
    mini_batch_size = 2
    epochs = 10
    eta = 0.001
    
    x_train = np.array([[1,2],[3,4]])
    y_train = np.array([[3],[7]])
    
    x_test = np.array([[1,3]])
    y_test = np.array([[4]]) 
    
    test_data = list(zip(list(x_test), list(y_test)))
    '''
    # We don't really care about nodes_model and nodes_batch for now 
    nn = NeuralNetwork(nodes_model=1, nodes_batch=2)
    
    nn.add_layer("fc", 4, 7)
    nn.add_layer("fc", 7, 8)
    nn.add_layer("fc", 8, 1)
    nn.add_loss("l2")
    #nn.add_loss("softmax")

    #nn.add_layer("fc", input_shape, 7)
    #nn.add_layer("fc",7, output_shape)
    #nn.add_loss("l2")
    
    nn.train_batch_parallelism(x_train, y_train, epochs, mini_batch_size,eta, test_data = test_data)
    
def _test_model(): 
    # UCI airfoil test
    data = np.loadtxt(open("Data/airfoil_self_noise.dat", "rb"), delimiter="\t")    
    x = data[:,1:5]
    y = data[:,5].reshape(len(x), 1)

    x_train = x[:1200]
    y_train = y[:1200]
    x_test = x[1200:]
    y_test = y[1200:]
    print(x.shape)
    print(y.reshape(len(y), 1).shape)

    test_data = list(zip(list(x_test), list(y_test)))
    
    input_shape = x.shape[1]
    output_shape = y.shape[1]
    epochs = 100
    mini_batch_size = 2
    eta = 0.00000000011
    
    # We don't really care about nodes_model and nodes_batch for now 
    nn = NeuralNetwork(nodes_model=1, nodes_batch=2)
    
    nn.add_layer("fc", 4, 7)
    nn.add_layer("fc", 7, 8)
    nn.add_layer("fc", 8, 1)
    nn.add_loss("l2")
    #nn.add_loss("softmax")

    #nn.add_layer("fc", input_shape, 7)
    #nn.add_layer("fc",7, output_shape)
    #nn.add_loss("l2")
    
    nn.train_model_parallelism(x_train, y_train, epochs, mini_batch_size,eta, test_data = test_data)
    
if __name__=="__main__":
    _test_model()    
