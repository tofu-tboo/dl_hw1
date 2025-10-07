import torch
import torch.nn.functional as functional

class TorchModel:
    def __init__(self):
        self.layers = None
        self.evaluate = None
        self.optimizer = None

        self.cache = {}
        self.last_loss = None
    
    def forward(self, x, y_onehot, train=True):
        pass

    def backward(self):
        self.optimizer.zero_grad()
        self.last_loss.backward()
        self.optimizer.step()
        self.last_loss = None

class TorchNNModel(TorchModel):
    def __init__(self, layers, evaluate, optimizer):
        self.layers = layers

        self.evaluate = evaluate
        self.optimizer = optimizer

        self.cache = {}
        self.last_loss = None

    @torch.no_grad()
    def probs_from_y_hat(self, y_hat):
        return functional.softmax(y_hat, dim=1).cpu().numpy()

    def forward(self, x, y_onehot, train=True):
        if train:
            self.layers.train()
        else:
            self.layers.eval()

        x = torch.from_numpy(x).float()
        y_idx = torch.from_numpy(y_onehot.argmax(axis=1)).long()

        y_hat = self.layers(x)
        loss = self.evaluate(y_hat, y_idx)

        self.cache["probs"] = self.probs_from_y_hat(y_hat)

        if train:
            self.last_loss = loss
        else:
            self.last_loss = None

        return float(loss.detach().cpu())

class TorchCNNModel(TorchModel):
    def __init__(self, layers, evaluate, optimizer):
        self.layers = layers
        self.evaluate = evaluate
        self.optimizer = optimizer

        self.cache = {}
        self.last_loss = None

    @torch.no_grad()
    def probs_from_y_hat(self, y_hat):
        return functional.softmax(y_hat, dim=1).cpu().numpy()

    def forward(self, x, y_onehot, train=True):
        if train:
            self.layers.train()
        else:
            self.layers.eval()

        x = torch.from_numpy(x).float()
        y_idx = torch.from_numpy(y_onehot.argmax(axis=1)).long()

        y_hat = self.layers(x)
        loss = self.evaluate(y_hat, y_idx)

        self.cache["probs"] = self.probs_from_y_hat(y_hat)

        if train:
            self.last_loss = loss
        else:
            self.last_loss = None

        return float(loss.detach().cpu())