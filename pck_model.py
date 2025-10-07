import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchNNModel:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.net = nn.Sequential(
            nn.Flatten(),          # (N,1,28,28) -> (N,784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.cache = {}
        self._last_loss = None

    @torch.no_grad()
    def _probs_from_logits(self, logits):
        return F.softmax(logits, dim=1).cpu().numpy()

    def forward(self, X_np, Y_onehot_np, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()

        x = torch.from_numpy(X_np).float()
        y_idx = torch.from_numpy(Y_onehot_np.argmax(axis=1)).long()

        logits = self.net(x)
        loss = self.criterion(logits, y_idx)

        self.cache["probs"] = self._probs_from_logits(logits)

        if train:
            self._last_loss = loss
        else:
            self._last_loss = None

        return float(loss.detach().cpu())

    def backward(self):
        self.optimizer.zero_grad()
        self._last_loss.backward()
        self.optimizer.step()
        self._last_loss = None
