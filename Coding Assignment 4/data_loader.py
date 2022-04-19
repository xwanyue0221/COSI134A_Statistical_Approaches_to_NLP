#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/15/20 4:24 AM
"""PTB dataset and batch dataloader

updated for Fall 2021"""
import os
import random
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import consts as C
from utils import load_txt


def convert_seq(seq, vocab, return_str=False):
  """converts an incoming sequence into its mapping based on vocab

  Args:
    seq: str or list of strs or list of ints
    vocab: stoi or itos
    return_str: whether to return the output as a string

  Returns:
    if return_str is True: a string delimited by white-space
    otherwise: a list of either str or int
  """
  if type(seq) is str:
    seq = seq.split()

  def is_in(tok, vocab):
    if type(vocab) is dict:
      return tok in vocab
    else: # type(vocab) is list:
      return 0 <= tok < len(vocab) # never False but just in case

  out_seq = []
  for tok in seq:
    if is_in(tok, vocab):
      out_seq.append(vocab[tok])
    else:
      out_seq.append(vocab[C.UNK])

  if return_str:
    out_seq = " ".join(out_seq)

  return out_seq

##################################### PTB ######################################
def load_ptb_dataset(data_dir, dataset_name):
  """loads a raw ptb dataset from data_dir"""
  # sentinels
  sent_path = os.path.join(data_dir, C.SENT_TEMPLATE.format(dataset_name))
  assert os.path.exists(sent_path)
  tree_path = os.path.join(data_dir, C.TREE_GOLD_TEMPLATE.format(dataset_name))
  assert os.path.exists(tree_path)

  sents = load_txt(sent_path, delimiter='\n')
  trees = load_txt(tree_path, delimiter='\n')

  return sents, trees

class PTB(Dataset):
  """Penn TreeBank Corpus"""
  def __init__(self, name, data, sent_stoi, tree_stoi):
    print(f"\n{name.capitalize()} PTB init")

    self.data = []
    for sent, tree in zip(*data):
      # EOS attached to the end of source sentences
      sent_idx = convert_seq(sent.split() + [C.EOS], sent_stoi)
      # BOS and EOS attached to the front and end of linearized target parse trees
      tree_idx = convert_seq([C.BOS] + tree.split() + [C.EOS], tree_stoi)
      self.data.append([sent_idx, tree_idx])

    # sample vector
    sample_idx = random.randint(0, len(self.data)-1)
    print("Sample vector from", name.capitalize())
    print(f"  Sent: {data[0][sample_idx]} {C.EOS}")
    print("  Sent Vector:", self.data[sample_idx][0])
    print(f"  Tree: {C.BOS} {data[1][sample_idx]} {C.EOS}")
    print("  Tree Vector:", self.data[sample_idx][1])

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    sent, tree = self.data[idx]

    # into Tensor
    sent = torch.LongTensor(sent)
    tree = torch.LongTensor(tree)

    # also collect valid sentence lengths, i.e. number of non-padding tokens
    sent_len = len(sent)

    return sent, tree, sent_len

  def __len__(self):
    return len(self.data)

################################# DATA LOADER ##################################
def collate_fn(batch, sent_pad_idx=0., tree_pad_idx=0.):
  """collator function to construct a single padded mini-batch

  In addition to padded sentence and parse tree tensors, also returns a list
  of valid lengths for sentence data
  """
  sents, trees, sent_lens = zip(*batch)

  sents = pad_sequence(sents, batch_first=True, padding_value=sent_pad_idx)
  trees = pad_sequence(trees, batch_first=True, padding_value=tree_pad_idx)

  return sents, trees, list(sent_lens)

def init_data_loader(data, sent_stoi, tree_stoi, batch_size):
  """initializes data loader (no bucketing)"""
  # partial function handle with padding index kwargs specified
  collate_fn_wrapper = partial(
    collate_fn, sent_pad_idx=sent_stoi[C.PAD], tree_pad_idx=tree_stoi[C.PAD])

  dataloader = DataLoader(
    data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_wrapper)

  return dataloader
