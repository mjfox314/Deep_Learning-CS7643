"""
SGD Optimizer.  (c) 2021 Georgia Tech

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

from ._base_optimizer import _BaseOptimizer
from collections import defaultdict
import numpy as np


class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum
        self.velocities = defaultdict(dict)

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                if 'weight' not in self.velocities[m].keys():
                    self.velocities[m]['weight'] = np.zeros(m.weight.shape)

                self.velocities[m]['weight'] = self.momentum * self.velocities[m]['weight'] - self.learning_rate * m.dw
                m.dw = self.velocities[m]['weight']
                m.weight += m.dw


                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                if 'bias' not in self.velocities[m].keys():
                    self.velocities[m]['bias'] = np.zeros(m.bias.shape)

                self.velocities[m]['bias'] = self.momentum * self.velocities[m]['bias'] - self.learning_rate * m.db
                m.db = self.velocities[m]['bias']
                m.bias += m.db
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
