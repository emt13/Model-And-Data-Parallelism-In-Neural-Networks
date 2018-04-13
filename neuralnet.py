import layers

class NeuralNetwork:

	def __init__(self, nodes_model = 1, nodes_batch = 1):
		"""
		Initialize the NeuralNetwork

		:param nodes_model: int, number of nodes to use for model parallelism 
		:param nodes_batch: int, number of nodes to use for batch parallelism
		"""
		layers = []
		nodes_model = nodes_model
		nodes_batch = nodes_batch
		pass

	def add_layer(self, layer_type, size_input, size_output)::
		"""
		TODO: Add a layer to the NeuralNetwork. 
		"""
		pass

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
