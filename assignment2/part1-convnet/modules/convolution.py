"""
2d Convolution Module.  (c) 2021 Georgia Tech

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


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
       
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        kernels, channels, height, width = x.shape

        # Define output dimensions
        output_Height = int((height + 2 * self.padding - self.kernel_size)/self.stride) + 1
        output_Width = int((width + 2 * self.padding - self.kernel_size)/self.stride) + 1

        self.cache = (x, output_Height, output_Width)

        if self.padding > 0:
            x = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))

        out = np.zeros((kernels, self.out_channels, output_Height, output_Width))

        for kernel in range(kernels):

            for channel in range(self.out_channels):

                for h in range(output_Height):
                    height_start = h * self.stride
                    height_end = height_start + self.kernel_size


                    for w in range(output_Width):
                        width_start = w * self.stride
                        width_end = width_start + self.kernel_size

                        temp = np.multiply(x[kernel, :, height_start:height_end, width_start:width_end], self.weight[channel, :, :, :])

                        out[kernel, channel, h, w] = np.sum(temp) + self.bias[channel]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x, output_H, output_W = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        kernels, channels, height, width = x.shape
        dx = np.zeros(x.shape)

        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        
        dw = np.zeros((self.out_channels, channels, self.kernel_size, self.kernel_size))

        p0 = np.sum(dout, axis=0)
        p1 = np.sum(p0, axis=1)
        db = np.sum(p1, axis=1)

        dx_padding = np.zeros(x.shape)

        for kernel in range(kernels):

            for channel in range(self.out_channels):

                for h in range(output_H):
                    height_start = h * self.stride
                    height_end = height_start + self.kernel_size

                    for w in range(output_W):
                        width_start = w * self.stride
                        width_end = width_start + self.kernel_size

                        dw[channel, :, :, :] += x[kernel, :, height_start:height_end, width_start:width_end] * dout[kernel, channel, h, w]
                        dx_padding[kernel, :, height_start:height_end, width_start:width_end] += self.weight[channel, :, :, :] * dout[kernel, channel, h, w]

        
        dx = dx_padding[:, :, self.padding:(self.padding + height), self.padding:(self.padding + width)]

        self.dx = dx
        self.dw = dw
        self.db = db
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
