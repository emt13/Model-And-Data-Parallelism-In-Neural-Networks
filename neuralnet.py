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

        def add_layer(self, layer_type, size_input, size_output):
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
    nn.add_layer("fc", 160, 20)
    nn.add_layer("fc", 80, 50)

    print(nn.layers)    

if __name__=="__main__":
    _test()    
