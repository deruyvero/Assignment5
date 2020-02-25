#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch.nn as nn
from pdb import set_trace as bp
import torch


class Highway(nn.Module):


    def __init__(self, embed_size_word):
        super(Highway, self).__init__()
        self.embed_size_word = embed_size_word
        self.linear_layer = nn.Linear(self.embed_size_word, self.embed_size_word, bias=True)
        self.linear_layer_2 = nn.Linear(self.embed_size_word, self.embed_size_word,bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_batch):

        y = self.linear_layer(input_batch)
        x_proj = self.relu(y)
        y_2 = self.linear_layer_2(input_batch)
        x_gate = self.sigmoid(y_2)
        new_x_gate = torch.add(torch.ones(x_gate.size()),-x_gate)
        x_first = torch.mul(x_gate ,x_proj)
        x_second = torch.mul(new_x_gate,input_batch)
        x_highway = torch.add(x_first,x_second)
        return x_highway









### END YOUR CODE 

