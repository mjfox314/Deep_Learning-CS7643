"""
Softmax Cross Entropy Module.  (c) 2021 Georgia Tech

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
import numpy as np


class SoftmaxCrossEntropy:
    """
    Compute softmax cross-entropy loss given the raw scores from the network.
    """

    def __init__(self):
        self.dx = None
        self.cache = None

    def forward(self, x, y):
        """
        Compute Softmax Cross Entropy Loss
        :param x: raw output of the network: (N, num_classes)
        :param y: labels of samples: (N, )
        :return: computed CE loss of the batch
        """
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N, _ = x.shape
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        self.cache = (probs, y, N)
        return probs, loss

    def backward(self):
        """
        Compute backward pass of the loss function
        :return:
        """
        probs, y, N = self.cache
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        self.dx = dx
