#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pdb import set_trace as bp

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size,hidden_size)
        self.char_output_projection = nn.Linear(hidden_size,len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id) , char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embeddings = self.decoderCharEmb(input)
        #dec_hidden =dec_hidden.permute(1,0,2)
        output,hn = self.charDecoder(char_embeddings,dec_hidden)
        y = self.char_output_projection(output)
        return y, hn
        
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        input_sequence = char_sequence[0:-1]
        output,hn = self.forward(input_sequence,dec_hidden)
        (length, batch, classes) = output.shape
        target_sequence = char_sequence[1:]
        output = output.reshape(-1,classes)
        target = target_sequence.reshape(-1)
        cross_entropy_losses = self.loss(output,target)
        return cross_entropy_losses
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_words = []
        (one, batch, hidden_size) = initialStates[0].shape
        current_char = torch.tensor([self.target_vocab.char2id["{"]]*batch,device = device)
        initial_states = initialStates
        for x in range(0,max_length):
            current_char_emb = self.decoderCharEmb(current_char)
            current_char_emb= current_char_emb.unsqueeze(dim=0)
            yt,new_initial_states=self.charDecoder(current_char_emb,initial_states)
            st = self.char_output_projection(yt)
            initial_states = new_initial_states
            pt = nn.functional.softmax(st,dim=2)
            current_char = torch.argmax(pt,dim=2)
            current_char = current_char.squeeze(dim=0)
            current_char_list = current_char.tolist()
            output_words.append(current_char_list)
        list_words = []
        for i in range(0,batch):
            word = ""
            x = 0
            while x <= max_length:
                if output_words[x][i] != self.target_vocab.end_of_word:
                    word = word + self.target_vocab.id2char[output_words[x][i]]
                x += 1
            list_words.append(word)
        return list_words


        
        ### END YOUR CODE

