import layers

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

        def train_batch_parallelism(self, x, y):
                """
                TODO: Training procedure for batch parallelism
                """
                
                # mpi init
                
                # create the layers themselves

                # for 1 ... epoch:
                #       if rank is 0, shuffle

                #       if rank is 0, break into minibatches
                
                #       for i in range(num_minibatches):
                #               x = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                #               y = scatter_helper(minibatches[i], mb_dimension, comm, rank, size)
                
                #               all_zs = [(x)]
                #               in = x
                #               for layer in layers:
                #                       out = layer.forward(in)
                #                       all_zs.append(out)
                #                       
                #               loss, dy = l2_loss(all_zs[-1], y) 
                #               
                #               dws, dbs = [], []       
                #               for layer in reversed(layers):
                #                       dx, dw, db = layer.backwards(dy)
                #                       dy = dx
                #                       dws.append(dw)
                #                       dbs.append(db)
                #               allReduce(dws, size)
                #               allReduce(dbs, size)
                


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
