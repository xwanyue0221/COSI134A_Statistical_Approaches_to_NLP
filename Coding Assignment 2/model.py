#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 9/22/20 9:09 PM
"""Logistic Regression Model

Feel free to change/restructure the the code below
"""
import torch
import torch.nn as nn
from torchtext.vocab import GloVe
from torch.nn import functional as F

class LR(nn.Module):
  def __init__(self, labels, num_length):
    super(LR, self).__init__()
    self.layer1 = nn.Linear(num_length*3, len(labels), bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    print("Reset parameters for LR Module")
    nn.init.normal_(self.layer1.weight, std=0.01)
    nn.init.constant_(self.layer1.bias, 0)

  def forward(self, arg1, conn, arg2):
    x = torch.cat([arg1.float(), conn.float(), arg2.float()],1)
    o = self.layer1(x)
    return o

class LR_Dense(nn.Module):
  def __init__(self, labels, vocab_size: int, num_embedding:int = 300, num_hidden:int = 180):
    super(LR_Dense, self).__init__()
    self.embedding = nn.Embedding(vocab_size, num_embedding)
    self.layer1 = nn.Linear(num_embedding*3, num_hidden, bias=True)
    self.layer2 = nn.Linear(num_hidden, num_hidden, bias=True)
    self.layer3 = nn.Linear(num_hidden, len(labels), bias=True)
    self.act = nn.ReLU()
    self.reset_parameters()

  def reset_parameters(self):
    print("Reset parameters for LR Module with Dense Embedding")
    nn.init.normal_(self.layer1.weight, std=0.01)
    nn.init.constant_(self.layer1.bias, 0)
    nn.init.normal_(self.layer2.weight, std=0.01)
    nn.init.constant_(self.layer2.bias, 0)
    nn.init.normal_(self.layer3.weight, std=0.01)
    nn.init.constant_(self.layer3.bias, 0)

  def forward(self, arg1, conn, arg2):
    arg1 = self.embedding(arg1.long())
    arg1 = torch.mean(arg1, 1)
    conn = self.embedding(conn.long())
    conn = torch.mean(conn, 1)
    arg2 = self.embedding(arg2.long())
    arg2 = torch.mean(arg2, 1)

    x = torch.cat([arg1, conn, arg2],1)
    s1 = self.act(self.layer1(x))
    s2 = self.act(self.layer2(s1))
    o = self.layer3(s2)
    return o

class LR_Glove(nn.Module):
  def __init__(self, labels, num_features, GloVe_vectors, num_hidden:int = 180):
    super(LR_Glove, self).__init__()
    self.embed = nn.Embedding.from_pretrained(GloVe_vectors)
    self.layer1 = nn.Linear(num_features, num_hidden, bias=True)
    self.layer2 = nn.Linear(num_hidden, num_hidden, bias=True)
    self.layer3 = nn.Linear(num_hidden, len(labels), bias=True)
    self.act = nn.ReLU()
    self.reset_parameters()

  def reset_parameters(self):
    print("Reset parameters for LR Module with GloVe Embedding")
    nn.init.normal_(self.layer1.weight, std=0.01)
    nn.init.constant_(self.layer1.bias, 0)
    nn.init.normal_(self.layer2.weight, std=0.01)
    nn.init.constant_(self.layer2.bias, 0)
    nn.init.normal_(self.layer3.weight, std=0.01)
    nn.init.constant_(self.layer3.bias, 0)

  def forward(self, arg1, conn, arg2):
    arg1 = self.embed(arg1.long())
    arg1 = torch.sum(arg1, 1)
    conn = self.embed(conn.long())
    conn = torch.sum(conn, 1)
    arg2 = self.embed(arg2.long())
    arg2 = torch.sum(arg2, 1)

    x = torch.cat([arg1, conn, arg2],1)
    s1 = self.act(self.layer1(x))
    s2 = self.act(self.layer2(s1))
    o = self.layer3(s2)
    return o

class LR_CNN(nn.Module):
  def __init__(self, labels, num_features, GloVe_vectors, kernel_size:int = 3, hidden_size:int = 180):
    super(LR_CNN, self).__init__()
    self.embedding = self.embed = nn.Embedding.from_pretrained(GloVe_vectors)
    self.conv = nn.Conv1d(num_features, hidden_size, kernel_size)
    self.linear = nn.Linear(hidden_size, len(labels), bias=True)
    self.reset_parameters()

  def reset_parameters(self):
    print("Reset parameters")
    nn.init.normal_(self.linear.weight, std=0.01)
    nn.init.constant_(self.linear.bias, 0)

  def forward(self, arg1, conn, arg2):
    arg1 = self.embed(arg1.long())
    arg1 = torch.transpose(arg1, 1, 2)
    conn = self.embed(conn.long())
    conn = torch.transpose(conn, 1, 2)
    arg2 = self.embed(arg2.long())
    arg2 = torch.transpose(arg2, 1, 2)
    x = torch.cat([arg1, conn, arg2],1)

    s1 = self.conv(x)
    conv = F.relu(s1)
    conv = F.max_pool1d(conv, conv.shape[-1])
    conv = torch.squeeze(conv, -1)

    logits = self.linear(conv)
    return logits