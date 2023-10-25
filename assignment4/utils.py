"""
Helper functions.  (c) 2021 Georgia Tech

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

import math
import time
import random

# Pytorch packages
import torch
import torch.optim as optim
import torch.nn as nn

# Numpy
import numpy as np

# Tqdm progress bar
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

RANDOM_SEED = 0


def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)


def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        if model.__class__.__name__ == 'FullTransformerTranslator':
            translation = model(source, target)
        else:
            translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].transpose(1, 0).to(device)
            target = data[1].transpose(1, 0).to(device)

            if model.__class__.__name__ == 'FullTransformerTranslator':
                translation = model(source, target)
            else:
                translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def plot_curves(train_perplexity_history, valid_perplexity_history, filename):
    '''
    Plot learning curves with matplotlib. Make sure training perplexity and validation perplexity are plot in the same figure
    :param train_perplexity_history: training perplexity history of epochs
    :param valid_perplexity_history: validation perplexity history of epochs
    :param filename: filename for saving the plot
    :return: None, save plot in the current directory
    '''
    #############################################################################
    # TODO:                                                                     #
    #    1) Plot learning curves of training and validation loss                #
    #    2) Plot learning curves of training and validation accuracy            #
    #############################################################################
    epochs = range(len(train_perplexity_history))
    plt.plot(epochs, train_perplexity_history, label='train')
    plt.plot(epochs, valid_perplexity_history, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Perplexity Curve - '+filename)
    plt.savefig(filename+'.png')
    plt.show()


def unit_test_values(testcase):
    if testcase == 'rnn':
        return torch.FloatTensor([[-0.9080, -0.5639, -3.5862],
                                  [-1.2683, -0.4294, -2.6910],
                                  [-1.7300, -0.3964, -1.8972],
                                  [-2.3217, -0.4933, -1.2334]]), torch.FloatTensor([[0.9629,  0.9805, -0.5052,  0.8956],
                                                                                    [0.7796,  0.9508, -
                                                                                        0.2961,  0.6516],
                                                                                    [0.1039,  0.8786, -
                                                                                        0.0543,  0.1066],
                                                                                    [-0.6836,  0.7156,  0.1941, -0.5110]])

    if testcase == 'lstm':
        ht = torch.FloatTensor([[-0.0452,  0.7843, -0.0061,  0.0965],
                                [-0.0206,  0.5646, -0.0246,  0.7761],
                                [-0.0116,  0.3177, -0.0452,  0.9305],
                                [-0.0077,  0.1003,  0.2622,  0.9760]])
        ct = torch.FloatTensor([[-0.2033,  1.2566, -0.0807,  0.1649],
                                [-0.1563,  0.8707, -0.1521,  1.7421],
                                [-0.1158,  0.5195, -0.1344,  2.6109],
                                [-0.0922,  0.1944,  0.4836,  2.8909]])
        return ht, ct

    if testcase == 'encoder':
        expected_out = torch.FloatTensor([[[-0.7773, -0.2031],
                                         [-0.6186, -0.2321]],

                                        [[ 0.0599, -0.0151],
                                         [-0.9237,  0.2675]],

                                        [[ 0.6161,  0.5412],
                                         [ 0.7036,  0.1150]],

                                        [[ 0.6161,  0.5412],
                                         [-0.5587,  0.7384]],

                                        [[-0.9062,  0.2514],
                                         [-0.8684,  0.7312]]])
        expected_hidden = torch.FloatTensor([[[ 0.4912, -0.6078],
                                         [ 0.4932, -0.6244],
                                         [ 0.5109, -0.7493],
                                         [ 0.5116, -0.7534],
                                         [ 0.5072, -0.7265]]])
        return expected_out, expected_hidden

    if testcase == 'decoder':
        expected_out = torch.FloatTensor(
        [[-2.1507, -1.6473, -3.1772, -3.2119, -2.6847, -2.1598, -1.9192, -1.8130,
         -2.6142, -3.1621],
        [-1.9727, -2.1730, -3.3104, -3.1552, -2.4158, -1.7287, -2.1686, -1.7175,
         -2.6946, -3.2259],
        [-2.1952, -1.7092, -3.1261, -2.9943, -2.5070, -2.1580, -1.9062, -1.9384,
         -2.4951, -3.1813],
        [-2.1961, -1.7070, -3.1257, -2.9950, -2.5085, -2.1600, -1.9053, -1.9388,
         -2.4950, -3.1811],
        [-2.7090, -1.1256, -3.0272, -2.9924, -2.8914, -3.0171, -1.6696, -2.4206,
         -2.3964, -3.2794]])
        expected_hidden = torch.FloatTensor([[
                                            [-0.1854,  0.5561],
                                            [-0.6016,  0.0276],
                                            [ 0.0255,  0.3106],
                                            [ 0.0270,  0.3131],
                                            [ 0.9470,  0.8482]]])
        return expected_out, expected_hidden

    if testcase == 'seq2seq':
        expected_out = torch.FloatTensor(
        [[[-2.4136, -2.2861, -1.7145, -2.5612, -1.9864, -2.0557, -1.7461,
          -2.1898],
         [-2.0869, -2.9425, -2.0188, -1.6864, -2.5141, -2.3069, -1.4921,
          -2.3045]],

        [[-1.8506, -2.3783, -2.1297, -1.9083, -2.5922, -2.3552, -1.5708,
          -2.2505],
         [-2.0939, -2.1570, -2.0352, -2.2691, -2.1251, -1.8906, -1.8156,
          -2.3654]]]
        )
        return expected_out

    if testcase == 'attention':

        hidden = torch.FloatTensor(
            [[[-0.7232, -0.6048],
              [0.9299, 0.7423],
              [-0.4391, -0.7967],
              [-0.0012, -0.2803],
              [-0.3248, -0.3771]]]
        )

        enc_out = torch.FloatTensor(
            [[[-0.7773, -0.2031],
              [-0.6186, -0.2321]],

             [[0.0599, -0.0151],
              [-0.9237, 0.2675]],

             [[0.6161, 0.5412],
              [0.7036, 0.1150]],

             [[0.6161, 0.5412],
              [-0.5587, 0.7384]],

             [[-0.9062, 0.2514],
              [-0.8684, 0.7312]]]
        )

        expected_attention = torch.FloatTensor(
            [[[0.4902, 0.5098]],

             [[0.7654, 0.2346]],

             [[0.4199, 0.5801]],

             [[0.5329, 0.4671]],

             [[0.6023, 0.3977]]]
        )
        return hidden, enc_out, expected_attention

    if testcase == 'seq2seq_attention':
        expected_out = torch.FloatTensor(
            [[[-2.8071, -2.4324, -1.7512, -2.7194, -1.7530, -2.1202, -1.6578,
               -2.0519],
              [-2.2137, -2.4308, -2.0972, -2.1079, -1.9882, -2.0411, -1.6965,
               -2.2229]],

             [[-1.9549, -2.4265, -2.1293, -1.9744, -2.2882, -2.4210, -1.4311,
               -2.4892],
              [-2.1284, -2.2369, -2.1940, -1.9027, -2.1065, -2.2274, -1.7391,
               -2.2220]]]
        )
        return expected_out

    if testcase == 'full_trans_fwd':
        expected_out = torch.FloatTensor(
        [[[ 3.6680e-01,  2.1252e-01],
         [-1.2687e-01, -1.6845e-02],
         [ 8.4793e-01, -1.6569e-03],
         [ 1.8504e-01, -4.2573e-01],
         [ 7.8903e-01, -2.0726e-01],
         [ 1.0729e-01, -1.2556e-01],
         [ 5.2014e-01, -2.0366e-01],
         [ 4.6855e-01,  3.2419e-01],
         [ 9.6934e-01, -2.4157e-01],
         [ 8.0141e-01, -6.5982e-01],
         [ 1.2299e+00, -3.5231e-01],
         [ 6.9334e-01, -4.2950e-01],
         [ 2.7598e-01, -3.1835e-01],
         [ 9.7785e-02, -3.2622e-01],
         [-6.0962e-02,  3.9045e-01],
         [ 6.9368e-01,  1.7958e-01],
         [ 7.4034e-01, -2.7244e-01],
         [ 9.2248e-01,  6.3926e-01],
         [ 6.1330e-01, -4.1219e-02],
         [ 7.0952e-01, -4.9079e-01],
         [ 5.7191e-01, -3.0470e-01],
         [ 4.6507e-01,  8.3403e-01],
         [ 2.1201e-01, -3.8308e-01],
         [ 4.3040e-01, -7.8571e-03],
         [ 1.2411e+00, -1.4215e-01],
         [ 2.3801e-02, -4.4040e-01],
         [ 7.0947e-01,  1.2884e-01],
         [ 1.0503e+00, -2.9420e-01],
         [ 9.6175e-02, -1.7561e-02],
         [ 1.4140e+00, -4.5639e-01],
         [ 5.1277e-01, -2.4841e-01],
         [ 4.9476e-01,  4.9553e-02],
         [ 3.5567e-01,  2.6022e-01],
         [ 6.0528e-01,  1.3823e-01],
         [ 1.1330e+00, -9.1921e-04],
         [ 9.0739e-01, -5.6595e-01],
         [ 9.0783e-01,  3.5331e-01],
         [ 6.0216e-01, -3.0372e-02],
         [ 6.2611e-01, -5.0479e-01],
         [ 6.8441e-01, -2.6078e-01],
         [ 6.0329e-01,  6.3386e-01],
         [ 5.4371e-01, -3.6148e-01],
         [ 2.3584e-01, -1.2726e-01]],

        [[ 3.8140e-01,  5.7107e-03],
         [ 1.6377e-01, -6.9416e-01],
         [ 3.6861e-01,  5.3238e-01],
         [ 3.0739e-02, -4.6234e-02],
         [ 4.9156e-01,  3.6072e-01],
         [ 1.1309e+00, -3.9870e-01],
         [ 2.4588e-01, -5.6991e-01],
         [ 2.3677e-01,  1.2573e-02],
         [ 8.9364e-01, -1.8963e-01],
         [ 4.4903e-01,  3.7950e-02],
         [ 7.1135e-01, -2.3661e-01],
         [ 6.7263e-01,  7.3550e-02],
         [ 7.3834e-01, -2.5687e-01],
         [ 2.6783e-01, -9.2334e-02],
         [ 7.3444e-01, -3.0316e-01],
         [ 5.4604e-01, -2.9387e-01],
         [ 9.1274e-01, -6.3237e-01],
         [ 8.9689e-01,  6.5350e-01],
         [ 3.4517e-01,  5.2061e-02],
         [ 6.5110e-01, -5.3729e-01],
         [ 2.1209e-01, -1.2686e-01],
         [ 2.0774e-01,  4.6330e-01],
         [ 6.6190e-01, -6.9394e-01],
         [ 3.4096e-01,  5.1402e-01],
         [ 1.4163e+00, -3.4571e-01],
         [ 2.9734e-01, -1.3164e-01],
         [ 1.3582e-01,  6.3813e-01],
         [-3.1242e-01,  3.0542e-02],
         [-9.0688e-02,  1.7369e-01],
         [ 8.5798e-01, -7.7958e-01],
         [-6.9660e-02,  9.2180e-02],
         [ 5.1660e-01,  1.7840e-01],
         [ 1.7422e-01,  4.3534e-01],
         [ 7.1363e-01, -4.4415e-01],
         [ 1.1238e+00, -1.5653e-01],
         [ 8.9552e-01,  3.3986e-01],
         [ 5.5951e-01,  3.5498e-01],
         [ 7.8363e-01,  5.5853e-01],
         [ 4.8786e-01, -2.9512e-01],
         [ 7.3725e-01, -1.6509e-01],
         [ 1.1258e-01,  1.9746e-01],
         [ 7.8411e-01, -5.6759e-02],
         [-3.5624e-02, -3.5026e-01]]]
         )
        return expected_out

    if testcase == 'full_trans_translate':
        expected_out = torch.FloatTensor(
        [[[3.3754e-01, 3.4120e-01],
          [4.7097e-02, -8.8609e-01],
          [-1.5206e-01, 9.0409e-01],
          [-2.8484e-01, 7.7650e-01],
          [-2.1604e-01, -1.7033e-01],
          [-8.3685e-02, -4.6718e-02],
          [-7.0285e-01, 5.0233e-01],
          [-5.4760e-01, 8.4246e-01],
          [-8.2701e-05, 3.5315e-01],
          [-2.2406e-01, 4.7082e-01],
          [-1.2476e-02, 1.3287e-01],
          [-1.2327e+00, 9.9374e-02],
          [-3.1840e-01, -1.3268e-01],
          [-6.0202e-01, 5.3006e-01],
          [-2.8903e-01, 3.3590e-02],
          [-3.0085e-01, 6.6473e-01],
          [-2.3081e-01, 1.9403e-01],
          [1.3384e-01, 1.4132e+00],
          [-4.0764e-01, -2.0022e-01],
          [-2.5855e-01, -1.9414e-01],
          [2.1617e-01, 6.7191e-02],
          [-7.9027e-02, 9.3593e-01],
          [2.7318e-01, -2.8950e-01],
          [-7.2171e-01, 8.2865e-01],
          [1.3157e+00, -3.8048e-01],
          [-1.7557e-01, 3.9178e-01],
          [6.0256e-01, -2.3496e-01],
          [-1.0316e+00, 4.1051e-01],
          [1.6212e-01, -6.9505e-02],
          [-5.9072e-01, -7.0511e-01],
          [2.9040e-01, 1.4021e-01],
          [-1.0198e-01, 7.0227e-01],
          [-4.2921e-01, 1.8300e-01],
          [-4.0161e-01, 8.1099e-02],
          [1.9886e-01, -5.3163e-01],
          [-1.7350e-01, -4.3070e-01],
          [7.4640e-01, 8.7889e-01],
          [3.2246e-01, 1.4947e-01],
          [-5.1718e-01, 1.1028e-02],
          [1.6620e-01, 4.7638e-01],
          [1.2984e-01, 1.1708e-01],
          [4.6783e-03, 5.5419e-01],
          [1.3422e-01, 1.4745e-01]],

         [[4.0842e-01, 1.0604e-02],
          [-7.3882e-01, 3.8131e-01],
          [2.9895e-01, 5.9621e-01],
          [-1.3731e-01, 5.8457e-01],
          [-2.4069e-01, 7.7224e-01],
          [2.7787e-01, 3.9152e-01],
          [-4.7583e-01, 7.3804e-03],
          [-3.6493e-01, 3.0972e-01],
          [9.5165e-02, 1.7377e-01],
          [4.6629e-01, 2.2459e-01],
          [2.0510e-01, 4.6955e-01],
          [-2.0433e-01, -2.0210e-02],
          [-8.2917e-02, -3.8209e-01],
          [-7.3977e-01, 1.6179e-01],
          [7.7701e-02, -1.3880e-02],
          [2.4427e-01, 4.1570e-01],
          [4.3639e-01, -6.6143e-02],
          [6.6535e-02, 5.3236e-01],
          [1.1496e+00, -5.0002e-01],
          [3.9622e-01, 2.3867e-01],
          [1.5201e+00, -6.9545e-02],
          [7.6039e-01, 7.2519e-01],
          [2.1579e-02, -5.3039e-01],
          [6.7461e-01, 4.6904e-01],
          [5.4411e-01, 3.9383e-01],
          [4.7705e-01, -5.2423e-01],
          [2.3734e-01, 7.4307e-03],
          [7.2856e-01, -6.8935e-01],
          [-6.5800e-02, 8.2293e-02],
          [9.6814e-01, -4.1541e-01],
          [-8.2306e-01, 5.7651e-02],
          [2.0749e-01, 3.0622e-01],
          [-2.8527e-01, 3.9096e-01],
          [-4.2774e-02, 2.4232e-01],
          [2.1262e-02, -6.9833e-01],
          [5.7973e-01, -2.1132e-01],
          [8.1885e-01, -2.4871e-01],
          [6.9119e-01, 3.3242e-02],
          [4.4813e-01, -5.5311e-01],
          [2.2172e-01, -2.7462e-01],
          [5.8928e-01, 2.3333e-01],
          [7.3170e-01, 2.3331e-03],
          [7.5364e-02, -7.4780e-01]]]
        )
        return expected_out



