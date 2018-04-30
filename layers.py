import numpy as np

class fully_connected_layer:
        def __init__(self, size_input, size_output, seed, mask=None):
                np.random.seed(seed)
                #mask needed for model parellelism to zero out values held by other processes
                #self.w = np.random.randn(size_input, size_output)
                #self.b = np.random.randn(size_output)
                self.w = np.ones((size_input, size_output))
                self.b = np.ones(size_output)
                self.mask_w = np.ones((size_input, size_output)) if mask is None else mask
                self.mask_b = np.ones(size_output) if mask is None else mask[0]
                assert self.mask_w.shape == self.w.shape
                assert self.mask_b.shape == self.b.shape
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
                out = np.dot(x, np.multiply(self.w, self.mask_w)) + self.b
                
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


                dx_flattened = np.dot(dout, np.multiply(self.w.T, self.mask_w.T))
                dw = np.dot(dx_flattened.T, dout)
                db = np.multiply(np.dot(dout.T, np.ones(N)), self.mask_b)
                dx = np.reshape(dx_flattened, x.shape)

                return dx, dw, db

        def apply_gradient(self, dw, db, eta, B):
                """
                Changes the weights based on the gradients provided

                Note, expects gradients to be scaled by step size or any relevant
                momentum, making it blind to the descent method

                :param dw: ndarray, weight gradients
                :param db: ndarray, bias gradients
                """
                assert dw.size == self.w.size
                assert db.size == self.b.size
                self.w = self.w - (1.0*eta)/(1.0*B)*dw
                self.b = self.b - (1.0*eta)/(1.0*B)*db

class l2_loss:
        def loss(self, x, y):
                """
                Returns L2 loss

                :param x: ndarray, prediction
                :param y: ndarray, correct value
                :return loss: float, L2 loss
                :return dx: gradient
                """
                N = x.shape[0]
                D = np.prod(x.shape[1:])

                _y = np.reshape(y, (N,D))
                _x = np.reshape(x, (N,D))
                _dx = 2*_x - 2*_y

                # loss = np.dot(_x.T, _x) - 2 * np.dot(_x.T, _y) + np.dot(_y.T, _y)
                #print(x - y)
                loss = np.linalg.norm(x-y, 1)
                dx = np.reshape(_dx, x.shape)

                return loss, dx


class softmax_loss:
        def loss(self, x, y):
                """
                Returns softmax loss

                :param x: ndarray, prediction
                :param y: ndarray, correct value
                :return loss: float, softmax loss
                :return dx: gradient
                """

                #scores = np.exp(x) / np.sum(np.exp(x), axis=0)  
                
                
                pass


"""
TESTS ---
"""
def test_fc():
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

def test_l2():
        x = np.ones((2,2))
        y = np.reshape(np.array([1,2,3,4]), [2,2]).astype(float)

        l2 = l2_loss()
        print(l2.loss(x,y))

def test_softmax():
        x = np.ones((2,2))
        y = np.reshape(np.array([1,2,3,4]), [2,2]).astype(int)

        sm = softmax_loss()
        print(sm.loss(x,y))

if __name__ == "__main__":
        test_l2()
        



        pass
