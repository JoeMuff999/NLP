# models.py

import numpy as np
import torch.nn as nn
import torch
from transformer import PositionalEncoding
from utils import Indexer
import random
import math


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        print(len(context))
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_indexer : Indexer):
        self.model = model
        self.vocab_index = vocab_indexer

    def get_next_char_log_probs(self, context):
        # print(context)
        # context = [context]
        # for i in range(20 - len(context)):
        #     context.insert(0, ' ')
        context = ' ' + context
        context = torch.LongTensor(np.array([self.vocab_index.index_of(ci) for ci in context]))
        # print(context.size())
        probs = self.model.forward(context)
        # print(probs.size())
        probs = probs[len(probs)-1].detach().numpy()
        # print(probs[len(probs)-1])
        # probs = np.log(probs)
        return probs

import time
def train_lm(args, train_text, dev_text, vocab_index : Indexer):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    seq_len = 20
    vocab_size = 27
    train_text = train_text[0:int(len(train_text))]
    start_time = time.time()
    training_chunks = []
    # construct chunks of seq_len
    for i in range(len(train_text) - seq_len):
        training_chunks.append(' ' + str(train_text[i:i+seq_len]))
    
    model = TransformerLM(vocab_size, seq_len, 200, 2, 200, 27, n_layers=4)
    model.zero_grad()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 1
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(len(training_chunks))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            ex = training_chunks[ex_idx] 
            label = torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in ex[1:]]))
            ex = ex[:len(ex)-1] # chop
            ex = np.array([vocab_index.index_of(ci) for ci in ex])
            ex = torch.LongTensor(ex)
            # print(ex.size())
            
            x = model.forward(ex)
            # print(x.sum())
            # print(label.size())
            loss = loss_fcn(x,label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print("time elapsed: " + str(time.time() - start_time))
        print("hello " + str(loss_this_epoch))
    
    model.eval()
    return NeuralLanguageModel(model, vocab_index)


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model, n_head, d_ff, d_output=27, n_layers=1):
        super().__init__()
        self.em = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.positional_encoder = PositionalEncoding(d_model, seq_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_layers)
        self.decoder_layer = nn.Linear(d_model, d_output)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, context):
        embed = self.em(context) * math.sqrt(self.d_model)
        mask = torch.triu(torch.ones(len(context), len(context))*float('-inf'), diagonal=1)
        encoded_context = torch.add(self.positional_encoder.forward(embed), embed)
        output = self.transformer_encoder(encoded_context, mask=mask)
        output = self.decoder_layer(output)
        return self.log_softmax(output)
