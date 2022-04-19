#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/15/20 4:30 AM
"""load and preprocess raw PTB dataset

updated for Fall 2021"""
import argparse
import os
import time

import nltk
import tqdm

import consts as C
import utils
from vocabs import compile_vocab, export_vocabs

argparser = argparse.ArgumentParser("PA4 Data Preprocessing Argparser")

# path flags
argparser.add_argument('--ptb_dir', default='./ptb',
                       help='path to ptb directory')
argparser.add_argument('--out_dir', default='./outputs/ptb',
                       help='path to ptb outputs')
# ptb preprocessing flags
argparser.add_argument('--lower', action='store_true',
                       help='whether to lower all sentence strings')
argparser.add_argument('--reverse', action='store_true',
                       help='whether to reverse the source sentences')
argparser.add_argument('--prune', action='store_true',
                       help='whether to remove parenthesis for leaf POS tags')
argparser.add_argument('--XX_norm', action='store_true',
                       help='whether to normalize all POS tags to XX')
argparser.add_argument('--closing_tag', action='store_true',
                       help='whether to attach closing POS tags')
argparser.add_argument('--keep_index', action='store_true',
                       help='whether to keep trace indices')

############################### SENT PROCESSING ################################
def normalize_tok(tok: str, lower=False, keep_index=False):
  """map certain raw tokens to normalized form

  modified from https://github.com/sloria/textblob-aptagger/blob/master/textblob_aptagger/taggers.py

  Args:
    tok: candidate token to be normalized
    lower: whether to lower-case
    keep_index: whether to keep index for trace tokens

  Returns:
    normalized token
  """
  if tok.isdigit() and len(tok) == 4:
    return '!YEAR'
  elif tok[0].isdigit():
    return '!DIGITS'
  elif tok[0] == '*':
    if '-' in tok and not keep_index:
      tail = tok.split('-')[-1]
      try:
        int(tail) # may fail with ValueError
        tok = tok[:tok.rindex('-')]  # drop trace index
      except ValueError:
        pass # safety
    return tok
  elif lower:
    return tok.lower()
  else:
    return tok

def process_sent(sent, lower=False, reverse=False, keep_index=False):
  """processes a single sentence by applying token-level normalization

  Note that each sentence is already tokenized.

  Args:
    sent: sentence to be processed
    lower: whether to lower-case
    reverse: whether to reverse sentence
    keep_index: whether to keep index for trace tokens

  Returns:
    processed sentence
  """
  if reverse:
    sent = reversed(sent)

  norm_sent = []
  for tok in sent:
    norm_tok = normalize_tok(tok, lower=lower, keep_index=keep_index)
    if norm_tok is not None:  # implicitly drops empty tokens
      norm_sent.append(norm_tok)

  return " ".join(norm_sent)

############################### TREE PROCESSING ################################
def normalize_pos(tag: str, XX_norm=False, keep_index=False):
  """normalization for POS tags in phrase structure trees

  Certain linguistic phenomena like wh-movement leads to PTB annotation of certain
  POS tags with an index at its tail, e.g. `*T*-1`. This index can be retained
  by setting `keep_index` to True.

  Args:
    tag: candidate POS tag to be considered
    XX_norm: whether to normalize tag value to `XX`
    keep_index: whether to keep trace index

  Returns:
    normalized POS tag
  """
  if XX_norm:
    return "XX"

  if not keep_index:
    if '-' in tag:
      if tag == '-NONE-':
        return tag
      tail = tag.split('-')[-1]
      try:
        # drop indices, e.g. `NP-SBJ-1` -> `NP-SBJ`, `*-1` -> `*`
        int(tail) # may fail with ValueError
        tag = tag[:tag.rindex('-')]  # drop trace index
      except ValueError:
        # retain function tags, e.g. `NP-SBJ`
        pass

    if '=' in tag:
      # drop indices, e.g. `NP-SBJ=1` -> `NP-SBJ`
      tag = tag[:tag.index("=")]

  return tag

def linearize_parse_tree(tree: nltk.tree.Tree, prune_leaf_brackets=False,
                         XX_norm=False, closing_tag=False, keep_index=False):
  """linearizes a phrase structure parse tree through a DFS while applying
  normalization to POS tags

  Args:
    tree: nltk tree object to be linearized
    prune_leaf_brackets: whether to reduce `(TAG1 (TAG2 ) )` to `(TAG1 TAG2 )`
    XX_norm: whether to normalize all POS tags to XX, i.e. `(TAG1 (TAG2 ..` to `(XX (XX ..`
    closing_tag: whether to append the corresponding POS tag after closing parenthesis,
      i.e. `(TAG1 TAG2 )` to `(TAG1 TAG2 )TAG1`
    keep_index: whether to keep trace index. See `normalize_pos` for details

  Returns:
    linearized parse tree
  """
  tree = str(tree).strip().split()

  stack, out = [], []
  for i, tok in enumerate(tree):
    if '(' in tok:
      idx = tok.index("(")
      tag = tok[idx+1:]
      tag = normalize_pos(tag, XX_norm=XX_norm, keep_index=keep_index)
      stack.append(tag)

      new_tag = '(' + tag
      out.append(new_tag)
    else:
      idx = tok.index(")")
      for _ in range(idx, len(tok)):
        # for each closing parenthesis, from inside out if there is more than one
        new_tag = ')'
        tag = stack.pop()
        if closing_tag:
          new_tag += tag
        out.append(new_tag)

        if prune_leaf_brackets:
          try:
            prev_tag = out[-2] # may fail with IndexError
            if prev_tag.startswith('('): # if False, then non-terminal
              prev_tok_tag = prev_tag[1:]
              if prev_tok_tag == tag:
                out = out[:-2] + [tag]
          except IndexError:
            pass # safety

  out = list(filter(None, out)) # drop empty tokens
  return " ".join(out)

##################################### MAIN #####################################\
def export_ptb_dataset(dataset, out_dir, name: str):
  """exports raw ptb dataset within out_dir"""
  assert name in C.PTB_SPLITS # sentinel
  sents, trees = dataset

  sent_file = os.path.join(out_dir, C.SENT_TEMPLATE.format(name))
  utils.export_txt(sents, sent_file)

  tree_file = os.path.join(out_dir, C.TREE_GOLD_TEMPLATE.format(name))
  utils.export_txt(trees, tree_file)

def load_process_dataset(data_dir: str, dataset_name: str, lower=False,
                         reverse_sent=False, pruen_leaf_brackets=False,
                         XX_norm=False, closing_tag=False, keep_index=False):
  """loads a single PTB dataset (dev, test or train) and apply preprocessing to
  source sentence and target parse tree"""
  assert dataset_name in C.PTB_SPLITS # sentinel

  print(f"\nLoading {dataset_name}..")
  dataset_dir = os.path.join(data_dir, dataset_name)

  # 1. PTB-style bracketed corpus reader from NLTK for loading raw data
  # From this reader, you can extract sentences by calling `reader.sents()`
  # and parse trees by calling `reader.parsed_sents()`. For a large corpus like
  # PTB's training, these calls can be time-consuming
  reader = nltk.corpus.BracketParseCorpusReader(dataset_dir, r'.*/wsj_.*\.mrg')

  # each sent already tokenized
  raw_sents = reader.sents()
  # linearized phrase structure tree (implicitly reflects DFS traversal order)
  raw_trees = reader.parsed_sents()

  # 2. preprocessed sentence data
  sents = []
  for sent in tqdm.tqdm(raw_sents, desc='[Processing Sents]'):
    sent = process_sent(sent, lower, reverse_sent, keep_index=keep_index)
    sents.append(sent)

  # 3. linearized and preprocessed parse tree data
  trees = []
  for tree in tqdm.tqdm(raw_trees, desc='[Processing Trees]'):
    tree = linearize_parse_tree(tree, prune_leaf_brackets=pruen_leaf_brackets, XX_norm=XX_norm,
                                closing_tag=closing_tag, keep_index=keep_index)
    trees.append(tree)

  print("Sample ptb from", dataset_name)
  print("  Sent:", sents[0])
  print("  Tree:", trees[0])

  return sents, trees

def prepare_data(ptb_dir: str, out_dir: str, lower=False, reverse_sent=False,
                 prune_leaf_brackets=False, XX_norm=False, closing_tag=False,
                 keep_index=False):
  """main PTB data preprocessing function"""
  # setup
  assert os.path.exists(ptb_dir) # sentinel
  os.makedirs(out_dir, exist_ok=True)

  print("\nBegin loading and processing PTB..")
  datasets, vocabs = [], None
  for dataset_name in C.PTB_SPLITS:
    # load raw ptb dataset and apply preprocessing
    ptb_dataset = load_process_dataset(ptb_dir, dataset_name, lower=lower,
                                       reverse_sent=reverse_sent,
                                       pruen_leaf_brackets=prune_leaf_brackets,
                                       XX_norm=XX_norm, closing_tag=closing_tag,
                                       keep_index=keep_index)
    datasets.append(ptb_dataset)

    # export processed dataset
    export_ptb_dataset(ptb_dataset, out_dir, dataset_name)

    # if training, compile and export vocab
    if dataset_name == C.TRAIN:
      vocabs = compile_vocab(ptb_dataset)
      export_vocabs(vocabs, out_dir)

  return datasets, vocabs

def main():
  """prepare data script entry point"""
  begin = time.time()

  args = argparser.parse_args()
  utils.display_args(args)

  # export args as json for reference
  os.makedirs(args.out_dir, exist_ok=True)
  args_path = os.path.join(args.out_dir, C.ARGS_FNAME)
  utils.export_json(vars(args), args_path)

  prepare_data(args.ptb_dir, args.out_dir, lower=args.lower,
               reverse_sent=args.reverse, prune_leaf_brackets=args.prune,
               XX_norm=args.XX_norm, closing_tag=args.closing_tag,
               keep_index=args.keep_index)

  utils.display_exec_time(begin, "PA4 Data Preprocessing")

if __name__ == '__main__':
  main()
