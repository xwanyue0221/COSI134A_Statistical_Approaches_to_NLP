#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/15/20 3:16 AM
"""vocabs

a vocab consits of itos (int-to-str) and stoi (str-to-int)

also handles GloVe

updated for Fall 2021"""
import os
import re
from collections import Counter

import numpy as np
import torch
import tqdm

import consts as C
from utils import export_json, load_json


def display_vocabs(sent_itos, tree_itos=None, prefix=''):
  """displays vocab info"""
  print(f"\n{prefix}Vocab Info:")
  print(f"  Sent ({len(sent_itos)}) => {', '.join(sent_itos[:100])}, ...")
  if tree_itos is not None:
    print(f"  Tree ({len(tree_itos)}) => {', '.join(tree_itos)}")

def export_vocabs(vocabs, out_dir):
  """exports vocab counters separately within out_dir"""
  sent_counter, tree_counter = vocabs

  sent_vocab_path = os.path.join(out_dir, C.SENT_VOCAB_FNAME)
  tree_vocab_path = os.path.join(out_dir, C.TREE_VOCAB_FNAME)

  export_json(sent_counter, sent_vocab_path)
  export_json(tree_counter, tree_vocab_path)

def to_stoi(itos):
  """from int-to-str to str-to-int"""
  return {word: i for i, word in enumerate(itos)}

def compile_vocab(data):
  """compiles vocabs for sentences and linearized phrase structure trees separately"""
  print("\nCompiling Vocab..")
  sent_counter, tree_counter = Counter(), Counter()

  for sent, tree in zip(*data):
    sent_counter.update(sent.split())
    tree_counter.update(tree.split())

  def sort_counter(counter):
    # sort by the number of occurrences
    return {x:v for x,v in sorted(counter.items(), reverse=True, key=lambda x: x[1])}

  sent_counter = sort_counter(sent_counter)
  tree_counter = sort_counter(tree_counter)

  display_vocabs(list(sent_counter.keys()), list(tree_counter.keys()))

  return sent_counter, tree_counter

def init_vocab(counter, threshold=-1, special_symbols=None):
  """init a single vocab object (tuple of itos and stoi) from counter"""
  if not special_symbols:
    special_symbols = []

  itos = special_symbols + [x for x, y in counter.items() if y >= threshold]
  stoi = to_stoi(itos)

  return itos, stoi

def load_vocabs(data_dir, sent_threshold=-1, tree_threshold=-1):
  """loads and inits sent and tree vocabs

  Args:
    data_dir: dir which contains vocab counter jsons
    sent_threshold: min. number of occurrences to count as sent vocab
    tree_threshold: min. number of occurrences to count as tree vocab

  Returns:
    sent_itos, sent_stoi, tree_itos, tree_stoi
  """
  sent_fpath = os.path.join(data_dir, C.SENT_VOCAB_FNAME)
  tree_fpath = os.path.join(data_dir, C.TREE_VOCAB_FNAME)

  sent_counter = load_json(sent_fpath)
  tree_counter = load_json(tree_fpath)

  sent_itos, sent_stoi = init_vocab(sent_counter, sent_threshold,
                                    special_symbols=[C.PAD, C.UNK, C.EOS])
  tree_itos, tree_stoi = init_vocab(tree_counter, tree_threshold,
                                    special_symbols=[C.PAD, C.UNK, C.BOS, C.EOS])

  validate_vocab(sent_stoi)
  validate_vocab(tree_stoi, is_target=True)

  display_vocabs(sent_itos, tree_itos)

  return sent_itos, sent_stoi, tree_itos, tree_stoi

def validate_vocab(vocab, is_target=False):
  """validates the integrity of ptb vocab

  Args:
    vocab: stoi dict
    is_target: whether `vocab` is that of target
  """
  # padding always at index 0
  assert vocab[C.PAD] == 0

  # eos required for both sent and tree vocab
  assert C.EOS in vocab

  # unk required for both sent and tree vocab
  assert C.UNK in vocab

  if is_target:
    # bos only required for tree vocab
    assert C.BOS in vocab

  # disallow empty string as vocab
  assert "" not in vocab

#################################### GLOVE #####################################
def infer_glove_name(glove_dir):
  """try to infer the GloVe name from files inside `glove_dir`

  the filenames must have not been modified for this to work"""
  candidates = []
  for fname in os.listdir(glove_dir):
    candidate = re.findall(r'glove.(.*)B.*', fname)
    if candidate:
      candidates.append(candidate[0])

  candidates = set(candidates)
  if len(candidates) != 1:
    raise Exception('Cannot infer GloVe name; specify `--glove_name` flag')

  name = f'{candidates.pop()}B'
  return name

def load_glove(glove_dir, name, embed_dim, with_torchtext=False):
  """load GloVe either manually or using torchtext

  may raise ValueError when:
  1. `name` is None and cannot be inferred automatically from files within `glove_dir`
  2. `embed_dim` not compatible with the given GloVe configuration"""
  print("\nLoading GloVe")

  if not name:
    name = infer_glove_name(glove_dir)

  if embed_dim not in C.GLOVE_DIMS[name]:
    raise ValueError(
      f'specified {embed_dim} embed_dim not compatible with glove.{name}')

  if with_torchtext:
    # for those who'd rather use torchtext to load GloVe
    # so long as this is False, no need to have `torchtext` installed
    from torchtext.vocab import GloVe
    glove = GloVe(name, dim=embed_dim, cache=glove_dir)
    return glove.vectors, glove.itos
  else:
    itos = []

    def get_num_lines(f):
      """take a peek through file handle `f` for the total number of lines"""
      num_lines = sum(1 for _ in f)
      f.seek(0)
      return num_lines

    glove_fname = os.path.join(glove_dir, f'glove.{name}.{embed_dim}d.txt')
    with open(glove_fname, 'rb') as f:
      num_lines = get_num_lines(f)

      vectors = torch.zeros(num_lines, embed_dim, dtype=torch.float32)
      for i, l in enumerate(tqdm.tqdm(f, total=num_lines)):
        l = l.split(b' ') # using bytes here is tedious but avoids unicode error
        word, vector = l[0], l[1:]

        try:
          word = word.decode('utf-8')
        except UnicodeDecodeError:
          continue

        itos.append(word)
        vectors[i] = torch.tensor([float(x) for x in vector])

    return vectors, itos

def init_glove(glove_dir, name, embed_dim, sent_itos, strategy, with_torchtext=False):
  """initialize GloVe pre-trained embedding

  mainly supports either torchtext or files from https://nlp.stanford.edu/projects/glove/
  But, if using huggingface urls, must deal with additional <unk> at the end of
  the files. For the sake of consistency we drop them and use our mean vector for <unk>

  some GloVes known to contain duplicate vocabs, but these are rare and shouldn't
  be a problem in PA4; hence no action is taken to avoid unnecessary slowdown

  supports 3 vocab handling strategies:
  1. `glove`: discard `sent_itos` and keep all of GloVe's vocabs and embedding without
    modification; will have large memory footprint
  2. `overlap`: only keep the vocabs found in both GloVe's vocabs and `sent_itos`
  3. `sent`: keep all of `sent_itos.` For those in `sent_itos` that appear in
    Glove's vocabs, fetch their corresponding GloVe vectors; for the rest, sample
    from normal distribution parameterized by GloVe embedding's summary statistics

  see `load_glove` for details when loading GloVe

  Args:
    glove_dir: path to dir containing unzipped GloVe files or cache dir when using torchtext
    name: GloVe names. See `consts.py` for a full list
    embed_dim: embedding dimension
    sent_itos: sentence vocab
    strategy: vocab handling strategy
    with_torchtext: whether to use torchtext when loading GloVe

  Returns:
    Tuple of embedding as torch Tensor, itos and stoi
  """
  glove_vectors, glove_itos = load_glove(glove_dir, name, embed_dim, with_torchtext)

  # when using huggingface urls, the downloaded files include <unk> at the end,
  # possibly more than one. We ignore these <unk>s and insert mean vector later
  if C.UNK in glove_itos:
    unk_idx = glove_itos.index(C.UNK)
    glove_vectors = glove_vectors[:unk_idx]
    glove_itos = glove_itos[:unk_idx]

  for symbol in C.SPECIAL_SYMBOLS: # sentinels
    assert symbol not in glove_itos

  unk_vector = torch.mean(glove_vectors, dim=0, keepdim=True)
  pad_vector = torch.zeros_like(unk_vector)
  glove_mean, glove_std = torch.mean(glove_vectors), torch.std(glove_vectors)
  eos_vector = torch.normal(glove_mean, glove_std, size=[1, embed_dim])

  msg_prefix = 'Glove Vocab handling strategy:'
  if strategy == C.KEEP_GLOVE:
    print(f"{msg_prefix} Keeping GloVe vocabs")
    embedding = glove_vectors
    itos = glove_itos
  elif strategy == C.KEEP_OVERLAP:
    print(f"{msg_prefix} Keeping overlap between GloVe and Sentence vocabs")
    overlap_vocabs, overlap_glove_idx, _ = np.intersect1d(
      glove_itos, sent_itos, assume_unique=True, return_indices=True)
    embedding = glove_vectors[overlap_glove_idx]
    itos = overlap_vocabs.tolist()
  else:
    print(f"{msg_prefix} Keeping Sentence vocabs")
    itos = sent_itos[3:] # temporarily exclude pad, unk and bos
    embedding = torch.zeros([len(itos), embed_dim], dtype=torch.float32)
    for i in tqdm.trange(len(itos)):
      sent_vocab = sent_itos[i+3] # offset by 3 to skip pad, unk and bos
      if sent_vocab in glove_itos: # exists in GloVe and Sentence vocabs
        embedding[i] = glove_vectors[glove_itos.index(sent_vocab)].unsqueeze(0)
      else: # only exists in Sentence vocabs
        embedding[i] = torch.normal(glove_mean, glove_std, size=[1, embed_dim])

  # insert special symbols
  itos.insert(0, C.PAD)
  itos.insert(1, C.UNK)
  itos.insert(2, C.EOS)

  stoi = to_stoi(itos)
  validate_vocab(stoi)

  # form a continguous embedding matrix
  embedding = torch.cat([pad_vector, unk_vector, eos_vector, embedding], dim=0)
  shape = embedding.shape
  print(f"  GloVe Embedding shape: ({shape[0]} x {shape[1]})")

  if strategy != C.KEEP_SENT:
    display_vocabs(itos, prefix='Updated GloVe ')

  return embedding, itos, stoi




