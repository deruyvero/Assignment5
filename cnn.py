#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch.nn as nn
from pdb import set_trace as bp
import torch

class CNN(nn.Module):
    # input = tensor => e_char X max_length_word X batch_sizeC

    def __init__(self,number_of_filters,kernel_size,embed_size):
        super(CNN,self).__init__()
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        self.conv =nn.Conv1d(self.number_of_filters ,self.embed_size,self.kernel_size)
        self.relu = nn.ReLU()

    def forward(self, batch_input):
        x_conv = self.conv(batch_input)
        relu_output = self.relu(x_conv)
        x_conv_out = torch.max(relu_output,2)[0]
        return x_conv_out






### END YOUR CODE

