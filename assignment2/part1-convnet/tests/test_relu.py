"""
ReLU Tests.  (c) 2021 Georgia Tech

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

import unittest
import numpy as np
from modules import ReLU
from .utils import *


class TestReLU(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def _relu_forward(self, x):
        relu = ReLU()
        return relu.forward(x)

    def test_forward(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        relu = ReLU()
        out = relu.forward(x)
        correct_out = np.array([[0., 0., 0., 0., ],
                                [0., 0., 0.04545455, 0.13636364, ],
                                [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
        diff = rel_error(out, correct_out)
        self.assertAlmostEquals(diff, 0, places=7)

    def test_backward(self):
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)

        dx_num = eval_numerical_gradient_array(lambda x: self._relu_forward(x), x, dout)

        relu = ReLU()
        out = relu.forward(x)
        relu.backward(dout)
        dx = relu.dx

        self.assertAlmostEquals(rel_error(dx_num, dx), 0, places=7)
