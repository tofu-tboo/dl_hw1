import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class LossGraph:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
    
    def show(self):
        plt.plot(self.train_losses, label="train")
        plt.plot(self.test_losses, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss graph")
        plt.legend()
        plt.show()

class ConfusionMatrix:
    def __init__(self):
        self.mat = np.zeros((10, 10))

    def show(self):
        plt.figure(figsize=(6,6))
        sns.heatmap(self.mat / self.mat.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues")
        plt.ylabel("Prediction")
        plt.show()

class Top3Images:
    def __init__(self):
        self.all_imgs = []
        self.all_probs = []
        self.all_preds = []

    def show(self):
        imgs  = np.concatenate(self.all_imgs,  axis=0)
        probs = np.concatenate(self.all_probs, axis=0)
        preds = np.concatenate(self.all_preds, axis=0)
        confs = probs.max(axis=1)

        top_idx = {c: [] for c in range(10)}
        for c in range(10):
            idx = np.where(preds == c)[0]
            if idx.size == 0:
                continue
            order = np.argsort(-confs[idx])[:3]
            top_idx[c] = idx[order]

        _, axes = plt.subplots(10, 3, figsize=(4, 12))
        for c in range(10):
            for j in range(3):
                ax = axes[c, j]
                if len(top_idx[c]) > j:
                    i = top_idx[c][j]
                    ax.imshow(imgs[i,0], cmap="gray")
                    ax.set_title(f"{probs[i, preds[i]]*100:.1f}%", fontsize=8)
                ax.axis("off")
        plt.tight_layout()
        plt.show()