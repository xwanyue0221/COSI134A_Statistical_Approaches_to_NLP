#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/14/20 8:47 PM
"""Seq2Seq and its componenets

updated for Fall 2021"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import consts as C

################################### ENCODER ####################################
class Encoder(nn.Module):
  """Recurrent Encoder"""
  def __init__(self, sent_stoi, embed_dim, hidden_dim, num_layers, dropout, rnn_type):
    """configs and layers for Encoder

    Args:
      sent_stoi: sent str-to-int vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: number of RNN layers
      dropout: dropout probability
      rnn_type: RNN, GRU or LSTM
    """
    super().__init__()
    ### configs
    self.rnn_type = rnn_type

    ### layers
    self.embedding = nn.Embedding(len(sent_stoi), embed_dim, padding_idx=sent_stoi[C.PAD])
    self.dropout = nn.Dropout(dropout)
    if rnn_type == C.GRU:
      self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
    elif rnn_type == C.LSTM:
      self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    else:
      self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)

    self.reset_parameters()

  def reset_parameters(self):
    for name, param in self.rnn.named_parameters():
      if 'weight' in name:
        nn.init.xavier_uniform_(param)
      elif 'bias' in name:
        nn.init.constant_(param, 0.)

  def init_pretrained_embedding(self, weights, finetune=False):
    """initializes nn.Embedding layer with pre-trained embedding weights

    Args:
      weights: GloVe embedding vector
      finetune: whether to finetune the embedding matrix during training
    """
    self.embedding = nn.Embedding.from_pretrained(weights, freeze=not finetune)

  def forward(self, x, lengths):
    """encodes source sentences

    Args:
      x: source tensor, (batch_size, sent_seq_len)
      lengths: valid source length list, (batch_size)

    Returns:
      outputs: RNN hidden states for all time-steps, (batch_size, sent_seq_len, hidden_dim)
      state: last RNN hidden state
        if RNN or GRU:  (num_layers, batch_size, hidden_dim)
        if LSTM: Tuple(
          (num_layers, batch_size, hidden_dim),
          (num_layers, batch_size, hidden_dim)
        )
    """
    # (batch_size, src_seq_len, embed_size)
    x = self.dropout(self.embedding(x))
    x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    outputs, state = self.rnn(x)
    # output: (batch_size, src_seq_len, hidden_dim)
    outputs, _ = pad_packed_sequence(outputs, batch_first=True)
    return outputs, state

################################### DECODER ####################################
class Decoder(nn.Module):
  """Vanilla Recurrent Decoder"""
  def __init__(self, tree_stoi, embed_dim, hidden_dim, num_layers, dropout, rnn_type):
    """configs and layers for Vanilla Decoder

    Args:
      tree_stoi: tree str-to-int vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: number of RNN layers
      dropout: dropout probability
      rnn_type: RNN, GRU or LSTM
    """
    super().__init__()
    ### configs
    vocab_size = len(tree_stoi)

    ### layers
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.dropout = nn.Dropout(dropout)
    if rnn_type == C.GRU:
      self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
    elif rnn_type == C.LSTM:
      self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    else:
      self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True)
    self.dense = nn.Linear(hidden_dim, vocab_size)

    self.reset_parameters()

  def reset_parameters(self):
    for name, param in self.rnn.named_parameters():
      if 'weight' in name:
        nn.init.xavier_uniform_(param)
      elif 'bias' in name:
        nn.init.constant_(param, 0.)
    nn.init.xavier_uniform_(self.dense.weight)
    nn.init.constant_(self.dense.bias, 0.)

  def forward(self, x, state, *args):
    """decodes for `tgt_seq_len` number of steps

    when `tgt_seq_len` == 1, we decode for one time-step. This is useful
      during inference, when not using teacher forcing

    when `tgt_seq_len` > 1, we decode for more than one-step. This can happen
      only when using teacher forcing, i.e. during training

    Args:
      x: target tensor, (batch_size, tgt_seq_len)
      state: previous hidden state
        if RNN or GRU:  (num_layers, batch_size, hidden_dim)
        if LSTM: Tuple(
          (num_layers, batch_size, hidden_dim),
          (num_layers, batch_size, hidden_dim)
        )
      *args: captures other unused arguments

    Returns:
      output: token-level logits, (batch_size, tgt_seq_len, vocab_size)
      state: decoder's last RNN hidden state. Similar shape as `state`
    """
    # (batch_size, src_seq_len, embed_size)
    x = self.dropout(self.embedding(x))

    # output: (batch_size, tgt_seq_len, hidden_dim)
    output, state = self.rnn(x, state)

    # (batch_size, tgt_seq_len, vocab_size)
    output = self.dense(output)
    return output, state

################################### BAHDANAU ###################################
class BahdanauAttentionDecoder(nn.Module):
  """Bahdanau (Additive) Attentional Decoder

  score = v^T \cdot \tanh(W_h \cdot H_h + W_e \cdot H_e)

  where H_e: encoder outputs, and H_h: previous decoder hidden state
  """
  def __init__(self, tree_stoi, embed_dim, hidden_dim, num_layers, dropout, rnn_type):
    """configs and layers for Decoder with Bahdanau Attention

    Args:
      tree_stoi: tree str-to-int vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: number of RNN layers
      dropout: dropout probability
      rnn_type: RNN, GRU or LSTM
    """
    super().__init__()
    raise NotImplementedError()

  def reset_parameters(self):
    pass

  def forward(self, x, state, encoder_outputs, mask=None):
    """single decoder step with Bahdanau attention. Additive attention takes place
    when the outptus of `fc_hidden` and `fc_encoder` are summed. There are other
    interpretations that prefer concat over addition.

    `x` may be a gold decoder input (with teacher forcing) or a previous decoder's
    prediction (without teacher forcing).

    Args:
      x: decoder input, (batch_size, 1)
      state: decoder's previous RNN hidden state
        if RNN or GRU:  (num_layers, batch_size, hidden_dim)
        if LSTM: Tuple(
          (num_layers, batch_size, hidden_dim),
          (num_layers, batch_size, hidden_dim)
        )
      encoder_outputs: RNN hidden states for all time-steps, (batch_size, src_seq_len, hidden_dim)
      mask: boolean tensor, (batch_size, 1)

    Returns:
      output: logits, (batch_size, 1, vocab_size)
      state: decoder's last RNN hidden state. Similar shape as `state`
    """
    raise NotImplementedError()

#################################### LUONG #####################################
class LuongAttentionDecoder(nn.Module):
  """Luong (Multiplicative) Attention

  Dot:
      score = H_e \cdot H_h
  General:
      score = H_e \cdot W \cdot H_h

  where H_e: encoder outputs, and H_h: previous decoder hidden state

  There also exists Luong Concat, but you are not asked to implement it.
  """
  def __init__(self, tree_stoi, embed_dim, hidden_dim, num_layers, dropout, rnn_type, mode):
    """configs and layers for Decoder with Luong Attention

    Args:
      tree_stoi: tree str-to-int vocab
      embed_dim: embedding feature dimension
      hidden_dim: RNN hidden dimension
      num_layers: number of RNN layers
      dropout: dropout probability
      rnn_type: RNN, GRU or LSTM
      mode: `dot` or `general`
    """
    super().__init__()
    raise NotImplementedError()

  def reset_parameters(self):
    pass

  def forward(self, x, state, encoder_outputs, mask=None):
    """single decoder step with Luong attention

    `x` may be a gold decoder input (with teacher forcing) or a previous decoder's
    prediction (without teacher forcing).

    Args:
      x: decoder input, (batch_size, 1)
      state: decoder's previous RNN hidden state
        if RNN or GRU:  (num_layers, batch_size, hidden_dim)
        if LSTM: Tuple(
          (num_layers, batch_size, hidden_dim),
          (num_layers, batch_size, hidden_dim)
        )
      encoder_outputs: RNN hidden states for all time-steps, (batch_size, src_seq_len, hidden_dim)
      mask: boolean tensor, (batch_size, 1)

    Returns:
      output: logits, (batch_size, 1, vocab_size)
      state: decoder's last RNN hidden state. Similar shape as `state`
    """
    raise NotImplementedError()
################################### SEQ2SEQ ####################################
class Seq2Seq(nn.Module):
  """Seq2Seq"""
  def __init__(self, *, model, sent_stoi, tree_stoi, embed_dim, hidden_dim,
               num_layers, dropout, rnn_type, glove, finetune_glove, device):
    super().__init__()
    print(f"\n{model.capitalize()} Seq2Seq init")

    ### configs
    self.model = model

    self.tree_vocab_size = len(tree_stoi)
    self.sent_pad_idx = sent_stoi[C.PAD]
    self.tree_bos_idx = tree_stoi[C.BOS]
    self.tree_eos_idx = tree_stoi[C.EOS]

    self.dropout = dropout
    self.rnn_type = rnn_type

    ### modules
    self.encoder = Encoder(sent_stoi=sent_stoi, embed_dim=embed_dim,
                           hidden_dim=hidden_dim, num_layers=num_layers,
                           dropout=dropout, rnn_type=rnn_type)
    if glove is not None:
      self.encoder.init_pretrained_embedding(glove, finetune=finetune_glove)

    decoder_kwargs = {'tree_stoi': tree_stoi, 'embed_dim': embed_dim,
                      'hidden_dim': hidden_dim, 'num_layers': num_layers,
                      'dropout': dropout, 'rnn_type': rnn_type}
    if model == C.BAHDANAU:
      self.decoder = BahdanauAttentionDecoder(**decoder_kwargs)
    elif model in [C.LUONG_DOT, C.LUONG_GENERAL]:
      mode = model.split('_')[-1]
      self.decoder = LuongAttentionDecoder(**decoder_kwargs, mode=mode)
    else:
      self.decoder = Decoder(**decoder_kwargs)

    self.device = device

  def parse(self, x, x_lens):
    """forward computation for inference

    during inference we keep it simple by assuming `batch_size` == 1

    Args:
      x: (1, src_seq_len)
      x_lens: (1)

    Returns:
      predictions as list of ints
    """
    # encode step
    encoder_outputs, state = self.encoder(x, x_lens)

    # setup for decoding loop
    max_decode_step = x.size(1) * 3

    # padding mask necessary for attentional decoders; not used by vanilla decoder
    # True if valid token, False otherwise (i.e. padding)
    padding_mask = x==self.sent_pad_idx

    # (1, 1)
    yt = torch.tensor([[self.tree_bos_idx]], dtype=torch.long, device=self.device)

    # decoding loop
    preds = []
    for i in range(max_decode_step):
      # decode step
      output, state = self.decoder(yt, state, encoder_outputs, padding_mask)

      # current time-step prediction
      yt = output.argmax(-1)
      yt_int = yt.item()

      # when parsing, no need to keep EOS in our predictions but simply terminate
      if yt_int == self.tree_eos_idx:
        break
      preds.append(yt_int)

    return preds

  def forward(self, x, y, x_lens, teacher_forcing_ratio=0.0):
    """forward computation for training

    `x` and `y` are padded tensors where:
    `x`: each row has valid tokens + EOS + possibly PADs
    `y`: except for row with longest valid length, has BOS + valid tokens + EOS + possibly PADs
    row with longest valid length has BOS + valid tokens, because `y` == `trees[:,:-1]`
    from the training loop in `train.py`

    See Recitation Week 12 & 13 slides for details

    Args:
      x: (batch_size, src_seq_len)
      y: (batch_size, tgt_seq_len)
      x_lens: (batch_size)
      teacher_forcing_ratio: float to determine whether to use teacher forcing
        at each decoding step

    Returns:
      token-level logits, (batch_size, tgt_seq_len, tree_vocab_size)
    """
    # encode step
    encoder_outputs, state = self.encoder(x, x_lens)

    if teacher_forcing_ratio == 1.:
      assert self.model == C.VANILLA, \
        f'full teacher forcing only supports Vanilla Seq2Seq, but your model is {self.model}'

      # decode step with teacher forcing; let PyTorch iterate through `tgt_seq_len` dim internally
      outputs, _ = self.decoder(y, state)
    else:
      batch_size, tgt_seq_len = y.shape

      # padding mask necessary for attentional decoders; not used by vanilla decoder
      # True if valid token, False otherwise (i.e. padding)
      padding_mask = x==self.sent_pad_idx

      # decoding initial inputs as BOS, (batch_size, 1)
      yt = y[:, 0].unsqueeze(-1)

      # the first two dimensions are swapped to make storing easier
      outputs = torch.zeros([tgt_seq_len, batch_size, self.tree_vocab_size], device=self.device)

      # manual iteration through `tgt_seq_len` dimension
      for i in range(tgt_seq_len):
        # output: (batch_size, 1, tree_vocab_size)
        output, state = self.decoder(yt, state, encoder_outputs, padding_mask)

        # save the model output: (1, batch_size, tree_vocab_size)
        outputs[i] = output.transpose(0, 1)

        # decoding input: (batch_size, 1)
        if random.random() < teacher_forcing_ratio:
          # without teacher forcing, use current prediction
          yt = output.argmax(-1)
        else:
          # with teacher forcing, fetch the next step's gold input
          try:
            yt = y[:,i+1].reshape([batch_size, 1])
          except IndexError:
            pass # last step, will terminate

      # (batch_size, tgt_seq_len, tree_vocab_size)
      outputs = outputs.transpose(0, 1)

    return outputs
