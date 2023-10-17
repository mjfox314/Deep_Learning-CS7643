import torch
import torch.nn as nn
from torch.autograd import Variable

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Use torch tensor math function or else you will run into issues later      #
        # where the computational graph is broken and appropriate gradients cannot   #
        # be computed.                                                               #
        ##############################################################################

        loss = Variable(torch.tensor(0.0))
        difference_h = img[:,:,0:-1,:] - img[:,:,1:,:]
        difference_v = img[:,:,:,0:-1] - img[:,:,:,1:]
        # loss = tv_weight * (torch.sum(difference_h**2) + torch.sum(difference_v**2))
        loss.add_(torch.sum(torch.pow(difference_h, 2)))
        loss.add_(torch.sum(torch.pow(difference_v, 2)))
        loss *= tv_weight

        # difference_h = torch.sum(torch.pow((img[:,:,:,:-1] - img[:,:,:,1:]), 2))
        # difference_v = torch.sum(torch.pow((img[:,:,:1,:] - img[:,:,1:,:]), 2))

        # loss += tv_weight * (difference_h + difference_v)

        return loss

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################