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
        self.params["W"] = np.sqrt(2.0 / in_dim) * rng.standard_normal(size=(in_dim, out_dim))
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
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(25)
        fan_in = in_channels * self.kernel_size * self.kernel_size
        self.params["W"] = np.sqrt(2.0 / fan_in) * rng.standard_normal((out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.params["b"] = np.zeros((1, out_channels, 1, 1))

    def forward(self, x):
        self.cache["x_shape"] = x.shape
        weight = self.params["W"]
        batch_size = x.shape[0]
        kernel_size = weight.shape[2]

        x_col, out_width, out_height = im2col(x=x, kernel_width=kernel_size, kernel_height=kernel_size, stride=self.stride, padding=self.padding
        )
        w_col = weight.reshape(weight.shape[0], -1).T

        out = x_col @ w_col + self.params["b"].reshape(1, -1)
        out = out.reshape(batch_size, out_width, out_height, weight.shape[0]).transpose(0, 3, 1, 2)

        self.cache["x_col"] = x_col
        self.cache["w_col"] = w_col
        self.cache["out_width"] = out_width
        self.cache["out_height"] = out_height
        return out

    def backward(self, dy):
        x_shape = self.cache["x_shape"]
        batch_size = x_shape[0]
        out_channels = dy.shape[1]

        dy_reshaped = dy.transpose(0, 2, 3, 1).reshape(batch_size * self.cache["out_width"] * self.cache["out_height"], out_channels)

        dw_col = self.cache["x_col"].T @ dy_reshaped
        dW = dw_col.T.reshape(self.params["W"].shape)
        db = dy.sum(axis=(0, 2, 3), keepdims=True)

        dx_col = dy_reshaped @ self.cache["w_col"].T
        dx = col2im(columns=dx_col, x_shape=x_shape, kernel_width=self.kernel_size, kernel_height=self.kernel_size, stride=self.stride, padding=self.padding)

        self.grads["W"] = dW
        self.grads["b"] = db
        return dx


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        self.cache["x_shape"] = x.shape
        x_col, out_width, out_height = im2col(x=x, kernel_width=self.kernel_size, kernel_height=self.kernel_size, stride=self.stride, padding=self.padding)
        batch_size, channels, _, _ = x.shape
        x_col = x_col.reshape(batch_size * out_width * out_height, channels, self.kernel_size * self.kernel_size)

        max_index = x_col.argmax(axis=2)
        out = x_col[np.arange(x_col.shape[0])[:, None], np.arange(channels), max_index]
        out = out.reshape(batch_size, out_width, out_height, channels).transpose(0, 3, 1, 2)

        self.cache["out_shape"] = (batch_size, channels, out_width, out_height)
        self.cache["max_index"] = max_index
        return out

    def backward(self, dy):
        batch_size, channels, out_width, out_height = self.cache["out_shape"]
        kernel_area = self.kernel_size * self.kernel_size
        dy_reshaped = dy.transpose(0, 2, 3, 1).reshape(batch_size * out_width * out_height, channels)

        dx_col = np.zeros((batch_size * out_width * out_height, channels, kernel_area))
        dx_col[np.arange(dx_col.shape[0])[:, None], np.arange(channels), self.cache["max_index"]] = dy_reshaped
        dx_col = dx_col.reshape(batch_size * out_width * out_height, channels * kernel_area)

        x_shape = self.cache["x_shape"]
        dx = col2im(columns=dx_col, x_shape=x_shape,kernel_width=self.kernel_size, kernel_height=self.kernel_size, stride=self.stride, padding=self.padding)
        return dx


class Flatten(Layer):
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
