"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        # 1) embedding layer 
        self.embedding_layer = nn.Embedding(self.output_size, self.emb_size)

        # 2) recurrent layer 
        if model_type=="RNN":
            self.recurrent_layer = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True)
        
        if model_type=="LSTM":
            self.recurrent_layer = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True)

        # 3) single linear layer with (log)softmax layer for output
        self.linear_layer = nn.Linear(self.decoder_hidden_size, self.output_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        # 4) dropout layer
        self.dropout = nn.Dropout(dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        q = hidden
        K = encoder_outputs

        q = torch.transpose(q, 0, 1)
        # K = torch.transpose(K, 1, 2)

        product = q @ K.mT

        q_norm = torch.linalg.norm(q, dim=-1, keepdim=True).squeeze(-1)
        K_norm = torch.linalg.norm(K, dim=-1, keepdim=True).squeeze()
        norm = q_norm.multiply(K_norm)

        cosine_similarity = torch.divide(product.squeeze(), norm).unsqueeze(1)

        probabilities = torch.nn.Softmax(dim=-1)
        attention = probabilities(cosine_similarity)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       If attention is true, compute the attention probabilities and use   #
        #       them to do a weighted average on the encoder_outputs to determine   #
        #       the hidden (and cell if LSTM) states that will be consumed by the   #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################

        embedding = self.dropout(self.embedding_layer(input))
        
        if self.model_type == 'LSTM':
            (hidden, cell_state) = hidden
            
        if attention:
            hidden_attention = self.compute_attention(hidden, encoder_outputs)
            hidden = hidden_attention @ encoder_outputs
            hidden = hidden.transpose(0, 1)

            if self.model_type=="LSTM":
                cell_attention = self.compute_attention(cell_state, encoder_outputs)
                cell_state = cell_attention @ encoder_outputs
                cell_state = cell_state.transpose(0, 1)

        if self.model_type == "RNN":
            output, hidden = self.recurrent_layer(embedding, hidden)
        elif self.model_type=="LSTM":
            output, hidden = self.recurrent_layer(embedding, (hidden, cell_state))
        
        # pass the output -> linear layer -> LogSoftmax
        output = self.linear_layer(output.squeeze())
        output = torch.nn.functional.log_softmax(output, dim=-1)  
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
