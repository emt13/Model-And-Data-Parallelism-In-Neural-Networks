import numpy as np

class fully_connected_layer:
	def __init__(self, size_input, size_output):
		self.w = np.random.random((size_input, size_output))
		self.b = np.random.random(size_output)
		self.cache = []

	def forward(self, x):
		"""
		Applies the forward pass, storing variables needed for the backward 
		pass in the cache

		:param x: ndarray, input data
		:return out: nparay, output of affine layer
		"""
		N = x.shape[0]
		D = np.prod(x.shape[1:])
		x_flattened = np.reshape(x,(N,D))
		out = np.dot(x, self.w) + self.b
		
		self.cache = x
		return out

	def backward(self, dout):
		"""
		Applies the backward pass

		:param dout: ndarray, downstream derivative
		:return dx: ndarray, derivative for next layer upstream
		:return dw: ndarray, weight gradients
		:return db: ndarray, bias gradients
		"""
		assert len(self.cache) != 0

		x = self.cache
		N = x.shape[0]
		D = np.prod(x.shape[1:])
		x_flattened = np.reshape(x, (N,D))

		dx_flattened = np.dot(dout, self.w.T)
		dw = np.dot(dx_flattened.T, dout)
		db = np.dot(dout.T, np.ones(N))
		dx = np.reshape(dx_flattened, x.shape)

		return dx, dw, db

	def apply_gradient(self, dw, db):
		"""
		Changes the weights based on the gradients provided

		Note, expects gradients to be scaled by step size or any relevant
		momentum, making it blind to the descent method

		:param dw: ndarray, weight gradients
		:param db: ndarray, bias gradients
		"""
		assert dw.size == self.w.size
		assert db.size == self.b.size
		self.w += dw
		self.b += db


if __name__ == "__main__":
	"""
	Test layers here
	"""
	x = np.reshape(np.array([1,2,3,4,5,6,7,8,9]), [3,3]).astype(float)
	weights = np.reshape(np.array([5,4,3,2,1,0]), [3,2]).astype(float)
	biases = np.ones(2)

	fc = fully_connected_layer(3,2)
	fc.w = weights
	fc.b = biases

	out = fc.forward(x)

	assert out.all() == (np.dot(x, weights) + biases).all()

	#Not sure what to test for backward pass. Just checking whether everything
	#from cache is working as expected

	grad = np.ones((3,2))
	assert out.shape == grad.shape
	dx,dw,db = fc.backward(grad)

	print("dx: {} \ndw: {}\n db: {}".format(dx, dw, db))

	fc.apply_gradient(dw, db)

	pass