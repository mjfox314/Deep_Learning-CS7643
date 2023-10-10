import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt

from image_utils import preprocess
class SaliencyMap:
    def compute_saliency_maps(self, X, y, model):
        """
        Compute a class saliency map using the model for images X and labels y.

        Input:
        - X: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the saliency map.

        Returns:
        - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
        images.
        """
        # Make sure the model is in "test" mode
        model.eval()

        # Wrap the input tensors in Variables
        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y, requires_grad=False)
        saliency = None

        ##############################################################################
        # TODO: Implement this function. Perform a forward and backward pass through #
        # the model to compute the gradient of the correct class score with respect  #
        # to each input image.                                                       #    
        #                                                                            #
        # There are two approaches to performing backprop using the PyTorch command  #
        # tensor.backward() when computing the gradient of a non-scalar  tensor. One #
        # approach is listed in PyTorch docs. Alternatively, one can take the sum of #
        # all the elements of the tensor and do a single backprop with the resulting #
        # scalar. This second approach is simpler and preferable as it lends itself  #  
        # vectorization, and this is the one you should implement.                   #
        ##############################################################################
        # Note: Only a single back-propagation pass is required
        pass

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return saliency

    def show_saliency_maps(self, X, y, class_names, model):
        # Convert X and y from numpy arrays to Torch Tensors
        X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
        y_tensor = torch.LongTensor(y)

        # Compute saliency maps for images in X
        saliency = self.compute_saliency_maps(X_tensor, y_tensor, model)
        # Convert the saliency map from Torch Tensor to numpy array and show images
        # and saliency maps together.
        saliency = saliency.numpy()

        N = X.shape[0]
        for i in range(N):
            plt.subplot(2, N, i + 1)
            plt.imshow(X[i])
            plt.axis('off')
            plt.title(class_names[y[i]])
            plt.subplot(2, N, N + i + 1)
            plt.imshow(saliency[i], cmap=plt.cm.gray)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 5)
        plt.savefig('visualization/saliency_map.png', bbox_inches = 'tight')
        #plt.show()
