"""
Vanilla RNN Model.  (c) 2021 Georgia Tech

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
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """ An implementation of vanilla RNN using Pytorch Linear layers and activations.
        You will need to complete the class init function, forward function and hidden layer initialization.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """ Init function for VanillaRNN class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
                output_size (int): the size of the output layer

            Returns: 
                None
        """
        super(VanillaRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #############################################################################
        # TODO:                                                                     #
        #    Initialize parameters and layers. You should                           #
        #    include a hidden unit, an output unit, a tanh function for the hidden  #
        #    unit, and a log softmax for the output unit.                           #
        #    hidden unit needs to be initialized before the output unit to pass GS  #
        #    You MUST NOT use Pytorch RNN layers(nn.RNN, nn.LSTM, etc).             #
        #############################################################################

        # initialize block that creates hidden

        # initialize block that creates output

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward function of the Vanilla RNN
            Args:
                input (tensor): a batch of data of shape (batch_size, input_size) at one time step
                hidden (tensor): the hidden value of previous time step of shape (batch_size, hidden_size)

            Returns:
                output (tensor): the output tensor of shape (batch_size, output_size)
                hidden (tensor): the hidden value of current time step of shape (batch_size, hidden_size)
        """


        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass for the Vanilla RNN. Note that we are only   #
        #   going over one time step. Please refer to the structure in the notebook.#                                              #
        #############################################################################

        output, hidden = None, None  #remove this line when you start implementing your code
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
