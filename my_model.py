import numpy as np
from util import *

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, x):
        pass

    def backward(self, dy):
        pass

class LinearLayer(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        rng = np.random.default_rng(25)
        self.params["W"] = 0.01 * rng.standard_normal(size=(in_dim, out_dim))
        self.params["b"] = np.zeros((1, out_dim))

    def forward(self, x):
        if (x.shape[1] != self.params["W"].shape[0]):
            x = x.reshape(x.shape[0], -1) # reshape x as (N, pts)
        self.cache["x"] = x # for backward
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dy):
        x = self.cache["x"] # assigned value in node

        self.grads["W"] = x.T @ dy
        self.grads["b"] = dy.sum(axis=0, keepdims=True)

        dx = dy @ self.params["W"].T # for next layer
        return dx

class Conv2D(Layer):
    """
    입력:  (N, Cin, H, W)
    파라미터: W (Cout, Cin, KH, KW), b (1, Cout, 1, 1)
    출력:  (N, Cout, OH, OW)  where OH,OW = get_out_shape(H,W,KH,KW,stride,pad)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, he_init=True):
        super().__init__()
        KH, KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.KH, self.KW = KH, KW
        self.stride = stride
        self.pad = padding

        fan_in = in_channels * KH * KW
        scale = np.sqrt(2.0/fan_in) if he_init else 0.01
        rng = np.random.default_rng(25)
        self.params["W"] = (rng.standard_normal((out_channels, in_channels, KH, KW)) * scale).astype(np.float32)
        self.params["b"] = np.zeros((1, out_channels, 1, 1), dtype=np.float32)

    def forward(self, x):
        # x: (N,C,H,W)
        self.cache["x_shape"] = x.shape
        W = self.params["W"]              # (Cout,Cin,KH,KW)
        b = self.params["b"]              # (1,Cout,1,1)
        N, C, H, Wimg = x.shape
        Cin = W.shape[1]
        assert C == Cin, f"Cin mismatch: {C} vs {Cin}"

        Xcol, OH, OW = im2col(x, self.KH, self.KW, self.stride, self.pad)    # (N*OH*OW, Cin*KH*KW)
        Wcol = W.reshape(W.shape[0], -1).T                                   # (Cin*KH*KW, Cout)
        out = Xcol @ Wcol + b.reshape(1, -1)                                 # (N*OH*OW, Cout)
        out = out.reshape(N, OH, OW, W.shape[0]).transpose(0, 3, 1, 2)       # (N,Cout,OH,OW)

        # cache
        self.cache["Xcol"] = Xcol
        self.cache["Wcol"] = Wcol
        self.cache["OH"] = OH
        self.cache["OW"] = OW
        return out

    def backward(self, dY):
        # dY: (N,Cout,OH,OW)
        X_shape = self.cache["x_shape"]
        Xcol = self.cache["Xcol"]                      # (N*OH*OW, Cin*KH*KW)
        Wcol = self.cache["Wcol"]                      # (Cin*KH*KW, Cout)
        N, C, H, Wimg = X_shape
        Cout = dY.shape[1]
        OH, OW = self.cache["OH"], self.cache["OW"]

        dYrs = dY.transpose(0, 2, 3, 1).reshape(N*OH*OW, Cout)  # (N*OH*OW, Cout)

        # dW
        dWcol = Xcol.T @ dYrs                                   # (Cin*KH*KW, Cout)
        dW = dWcol.T.reshape(self.params["W"].shape)            # (Cout,Cin,KH,KW)

        # db
        db = dY.sum(axis=(0, 2, 3), keepdims=True)              # (1,Cout,1,1)

        # dX
        dXcol = dYrs @ Wcol.T                                   # (N*OH*OW, Cin*KH*KW)
        dX = col2im(dXcol, X_shape, self.KH, self.KW, self.stride, self.pad)  # (N,Cin,H,W)

        self.grads["W"] = dW
        self.grads["b"] = db
        return dX

class MaxPool2D(Layer):
    """
    윈도우 내 최대값만 통과. 파라미터 없음.
    입력:  (N,C,H,W) -> 출력: (N,C,OH,OW)
    """
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.KH, self.KW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pad = padding

    def forward(self, x):
        self.cache["x_shape"] = x.shape
        Xcol, OH, OW = im2col(x, self.KH, self.KW, self.stride, self.pad)  # (N*OH*OW, C*KH*KW)
        N, C, H, W = x.shape
        Xcol = Xcol.reshape(N*OH*OW, C, self.KH*self.KW)                   # (N*OH*OW, C, K)
        max_idx = Xcol.argmax(axis=2)                                      # (N*OH*OW, C)
        out = Xcol[np.arange(Xcol.shape[0])[:,None], np.arange(C), max_idx]# (N*OH*OW, C)
        out = out.reshape(N, OH, OW, C).transpose(0,3,1,2)                 # (N,C,OH,OW)

        self.cache["Xcol_shape"] = (N, C, OH, OW)
        self.cache["max_idx"] = max_idx
        return out

    def backward(self, dY):
        N, C, OH, OW = self.cache["Xcol_shape"]
        K = self.KH * self.KW
        dYrs = dY.transpose(0,2,3,1).reshape(N*OH*OW, C)        # (N*OH*OW, C)

        dXcol = np.zeros((N*OH*OW, C, K), dtype=dY.dtype)       # (N*OH*OW, C, K)
        max_idx = self.cache["max_idx"]
        # put dY into argmax positions
        dXcol[np.arange(dXcol.shape[0])[:,None], np.arange(C), max_idx] = dYrs
        dXcol = dXcol.reshape(N*OH*OW, C*K)

        # col2im으로 원복
        X_shape = self.cache["x_shape"]
        dX = col2im(dXcol, X_shape, self.KH, self.KW, self.stride, self.pad)
        return dX

class Flatten(Layer):
    '''transition b/w mat <-> array'''
    def forward(self, x):
        self.cache["shape"] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.cache["shape"])


class ActiveFunc(Layer):
    def __init__(self):
        super().__init__()

class ReLU(ActiveFunc):
    def forward(self, x):
        mask = x > 0
        self.cache["mask"] = mask # for backward
        return x * mask # False entry as 0

    def backward(self, dy):
        return dy * self.cache["mask"]

class SoftmaxCrossEntropy: # Combine Softmax and CE => Easy calculation
    def __init__(self):
        self.cache = {}

    def forward(self, y_hat, y):
        z = y_hat - y_hat.max(axis=1, keepdims=True) # prevent overflow
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-12, 1.0)
        self.cache["probs"] = probs
        self.cache["y"] = y

        loss = -np.mean(np.sum(y * np.log(probs), axis=1))
        return loss

    def backward(self):
        probs = self.cache["probs"]
        y = self.cache["y"]
        N = probs.shape[0]

        return (probs - y) / N

class Optimizer:
    def __init__(self, lr = 0.01):
        self.learning_rate = lr

    def step(self):
        pass

    def zero_grad(self, layers):
        for _, grads in layers:
            grads[...] = 0.0

class SGD(Optimizer):
    def __init__(self, lr=0.01, weight_decay=0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def step(self, layers):
        for params, grads in layers:
            if self.weight_decay > 0:
                grads = grads + self.weight_decay * params
            params -= self.learning_rate * grads

class Model:
    def __init__(self, layers, evaluate, optimizer):
        self.layers = layers
        self.evaluate = evaluate
        self.parameters = []
        self.optimizer = optimizer

    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        return self.evaluate.forward(x, y)

    def backward(self):
        grad = self.evaluate.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        self.vectorize_parameters()

        self.optimizer.step(self.parameters)
        self.optimizer.zero_grad(self.parameters)

    def vectorize_parameters(self): # vectorize parameters and gradients
        self.parameters.clear()
        for layer in self.layers:
            for k in layer.params.keys():
                self.parameters.append((layer.params[k], layer.grads[k]))
