#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/15/20 2:05 AM
"""constants

updated for Fall 2021"""
# official PTB dataset split names
DEV = 'dev'
TRAIN = 'train'
TEST = 'test'
PTB_SPLITS = ['dev', 'train', 'test']

# default filenames
ARGS_FNAME = 'args.json'
SENT_VOCAB_FNAME = 'sent_vocab.json'
TREE_VOCAB_FNAME = 'tree_vocab.json'

# filename templates
SENT_TEMPLATE = '{}_sent.txt'
TREE_GOLD_TEMPLATE = '{}_tree.txt'
TREE_PRED_TEMPLATE = '{}_pred{}.txt'
MODEL_PT_TEMPLATE = 'epoch_{}.pt'

# vocab special symbols
PAD = '<pad>' # padding token
UNK = '<unk>' # unknown token
BOS = '<bos>' # beginning of sequence
EOS = '<eos>' # end of sequence
SPECIAL_SYMBOLS = [PAD, UNK, BOS, EOS]

# GloVe names
GLOVE_6B = '6B'
GLOVE_42B = '42B'
GLOVE_840B = '840B'
GLOVE_TWITTER_27B = 'twitter.27B'
GLOVE_NAMES = [GLOVE_6B, GLOVE_42B, GLOVE_840B, GLOVE_TWITTER_27B]

# accepted GloVe dims
GLOVE_DIMS = {
  GLOVE_6B : [50, 100, 200, 300],
  GLOVE_42B : [300],
  GLOVE_840B : [300],
  GLOVE_TWITTER_27B : [25, 50, 100, 200]
}

# vocab handling strategy when using GloVe
KEEP_GLOVE = 'glove' # keep GloVe's vocabs (and discard sentence vocabs)
KEEP_OVERLAP = 'overlap' # only keep vocabs that appear in both GloVe and sentence vocabs
KEEP_SENT = 'sent' # keep sentence vocabs; for those not in GloVe, sample from normal distribution
GLOVE_STRATEGIES = [KEEP_GLOVE, KEEP_OVERLAP, KEEP_SENT]

# Seq2Seq choices
VANILLA = 'vanilla' # vanilla seq2seq: decoder with no attention
BAHDANAU = 'bahdanau' # decoder with Bahdanau (additive) attention
LUONG_DOT = 'luong_dot' # decoder with Luong Dot attention
LUONG_GENERAL = 'luong_general' # decoder with Luong General attention
SEQ2SEQ_TYPES = [VANILLA, BAHDANAU, LUONG_DOT, LUONG_GENERAL]

# RNN choices
RNN = 'rnn'
GRU = 'gru'
LSTM = 'lstm'
RNN_TYPES = [RNN, GRU, LSTM]
