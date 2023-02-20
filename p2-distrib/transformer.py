# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, n_heads):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.em = nn.Embedding(vocab_size, d_model) # d_model = 512...
        # self.num_positions = num_positions
        self.d_model = d_model
        self.transformers = nn.ModuleList([TransformerLayer(d_model, d_internal, n_heads=n_heads) for _ in range(num_layers)])
        # self.transformer = TransformerLayer(d_model, d_internal, n_heads=1)
        self.output = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.p_encoder = PositionalEncoding(d_model, num_positions, batched=False)


    def forward(self, indices):
        """
        :param indices: list of input indices (longs)
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        #1. positional encoding
        vectors = self.em(indices) # from d = 20 -> d = 512
        positions = torch.zeros((20, self.d_model))
        vectors = torch.add(self.p_encoder.forward(positions), vectors)
        #2. attention layers
        attn = torch.clone(vectors)
        attn_maps = []
        for transformer in self.transformers:
            _map, attn = transformer.forward(attn)
            attn_maps.append(_map)
        # attn_map_1, attn = self.transformers[0].forward(vectors)
        # attn_map_2, attn = self.transformers[0].forward(attn)

        
        # attn_map_1, attn = self.transformer.forward(vectors) # from 512 -> 512
        # attn_map_2, attn = self.transformer.forward(attn)
        # attn_maps = [attn_map_1, attn_map_2]
        #3. lim, log probs
        log_probs = self.log_softmax(self.output(attn)) # from 512 -> 20
        # print(S.size())
        return (log_probs, attn_maps)
        

        


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# 1. self-attention. single headed 
# 2. residual connection
# 3. linear layer, nonlinear, linear layer
# 4. final residual connection
# vector-sequence model
# positional encoder
# attention unit
# feedforward unit
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, n_heads):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()



        self.d_model = d_model
        self.d_internal = d_internal
        self.d_ff = 100
        self.softmax = nn.Softmax(dim=1)

        self.Q = nn.ModuleList([nn.Linear(d_model, d_internal) for _ in range(n_heads)])
        self.K = nn.ModuleList([nn.Linear(d_model, d_internal) for _ in range(n_heads)])
        self.V = nn.ModuleList([nn.Linear(d_model, d_internal) for _ in range(n_heads)])

        # self. = nn.ModuleList([nn.Linear(d_internal, d_model)])
        self.W_0 = nn.Linear(d_internal*n_heads, d_model)

        self.g = nn.ReLU()
        self.lin = nn.Linear(d_model, self.d_ff)

        self.W = nn.Linear(self.d_ff, d_model)
        self.log_softmax = nn.LogSoftmax(dim=1)
        # Initialize weights according to a formula due to Xavier Glorot.
        # for q,k,v in zip(self.Q,self.K,self.V):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, input_vecs):

        Z_cat = torch.Tensor()
        #1 multi-head self attention
        for trio in zip(self.Q, self.K, self.V):
            q,k,v = trio
            S = torch.matmul(q(input_vecs), torch.t(k(input_vecs)))
            S = self.softmax(S/np.sqrt(self.d_internal)) #20x20
            Z = torch.matmul(S, v(input_vecs)) #20x20 x 20xd_internal = 20 x d_internal
            Z_cat = torch.cat((Z_cat, Z), dim=1)

        Z_cat = self.W_0(Z_cat) # 20 x d_internal*n_heads -> 20 x d_model
        #2. residual connection
        Attention = torch.add(input_vecs, Z_cat)
        #3. linear layer, nonlinear, linear layer FFN
        FeedForward = self.W(self.g(self.lin(Attention)))
        # 4. final residual connection
        return (S, torch.add(FeedForward, Attention))

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(27, 20, 512, 64,3,num_layers=1, n_heads=8)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 2
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        # ex_idxs = ex_idxs[0:int(len(ex_idxs)/2)]
        # random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        print(len(ex_idxs))
        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            # print(ex.size())
            # print("hello")
            loss = loss_fcn(model.forward(ex.input_tensor)[0], ex.output_tensor) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
            # print(ex_idx)
        print("hello " + str(loss_this_epoch))
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
