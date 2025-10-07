from dataloader import Dataloader
from my_model import *
from pck_model import *
from graph import *

train_loader = Dataloader(".", is_train=True, shuffle=True, batch_size=64)
test_loader  = Dataloader(".", is_train=False, shuffle=False, batch_size=256)

layers = [
    Conv2D(in_channels=1, out_channels=8,  kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2D(kernel_size=2),
    Flatten(),
    LinearLayer(16*7*7, 64),
    ReLU(),
    LinearLayer(64, 10),
]
my_model = Model(layers, evaluate=SoftmaxCrossEntropy(), optimizer=SGD(lr=0.01, weight_decay=1e-4))
pck_model = TorchNNModel(lr=0.01, weight_decay=0.0)

loss_graph_my = LossGraph()
conf_mat_my = ConfusionMatrix()
top3_images_my = Top3Images()
loss_graph_pck = LossGraph()
conf_mat_pck = ConfusionMatrix()
top3_images_pck = Top3Images()

for _ in range(20):
    total_loss_my, total_samples_my = 0.0, 0
    total_loss_pck, total_samples_pck = 0.0, 0
    for Xb, Yb in train_loader:
        loss_my = my_model.forward(Xb, Yb)
        my_model.backward()

        loss_pck = pck_model.forward(Xb, Yb, train=True)
        pck_model.backward()

        total_loss_my += loss_my * Xb.shape[0]
        total_samples_my += Xb.shape[0]
        total_loss_pck += loss_pck * Xb.shape[0]
        total_samples_pck += Xb.shape[0]

    loss_graph_pck.train_losses.append(total_loss_pck / total_samples_pck)
    loss_graph_my.train_losses.append(total_loss_my / total_samples_my)
    
    total_loss_my, total_samples_my = 0.0, 0
    total_loss_pck, total_samples_pck = 0.0, 0
    for Xb, Yb in test_loader:
        loss_my = my_model.forward(Xb, Yb)
        loss_pck = pck_model.forward(Xb, Yb, train=False)

        total_loss_my += loss_my * Xb.shape[0]
        total_samples_my += Xb.shape[0]
        total_loss_pck += loss_pck * Xb.shape[0]
        total_samples_pck += Xb.shape[0]

    loss_graph_pck.test_losses.append(total_loss_pck / total_samples_pck)
    loss_graph_my.test_losses.append(total_loss_my / total_samples_my)

Xb, Yb = next(iter(test_loader))
for Xb, Yb in test_loader:
    my_model.forward(Xb, Yb)
    probs_my = my_model.evaluate.cache["probs"]
    preds_my = probs_my.argmax(axis=1)

    pck_model.forward(Xb, Yb, train=False)
    probs_pck = pck_model.cache["probs"]
    preds_pck = probs_pck.argmax(axis=1)


    labels = Yb.argmax(axis=1)
    for t, p in zip(labels, preds_my):
        conf_mat_my.mat[t, p] += 1
    for t, p in zip(labels, preds_pck):
        conf_mat_pck.mat[t, p] += 1

    top3_images_my.all_imgs.append(Xb)
    top3_images_my.all_probs.append(probs_my)
    top3_images_my.all_preds.append(probs_my.argmax(1))

    top3_images_pck.all_imgs.append(Xb)
    top3_images_pck.all_probs.append(probs_pck)
    top3_images_pck.all_preds.append(probs_pck.argmax(1))

loss_graph_my.show()
conf_mat_my.show()
top3_images_my.show()

loss_graph_pck.show()
conf_mat_pck.show()
top3_images_pck.show()