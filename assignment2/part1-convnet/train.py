"""
Plotter.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import matplotlib.pyplot as plt

from modules import ConvNet
from optimizer import SGD
from trainer import ClassifierTrainer
from data import get_CIFAR10_data

root = 'data/cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(root)

model_list = [dict(type='Conv2D', in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
              dict(type='ReLU'),
              dict(type='MaxPooling', kernel_size=2, stride=2),
              dict(type='Linear', in_dim=8192, out_dim=10)]
criterion = dict(type='SoftmaxCrossEntropy')
model = ConvNet(model_list, criterion)
optimizer = SGD(model, learning_rate=0.0001, reg=0.001, momentum=0.9)

trainer = ClassifierTrainer()

loss_history, train_acc_history = trainer.train(
    X_train[:50], y_train[:50], model, batch_size=10, num_epochs=10,
    verbose=True, optimizer=optimizer)

plt.plot(train_acc_history)
plt.legend(['train'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('train.png')
