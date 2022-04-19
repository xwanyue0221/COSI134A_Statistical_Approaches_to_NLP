#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/14/20 8:59 PM
"""inference

updated for Fall 2021"""
import argparse
import os
import time

import torch
import tqdm

import consts as C
import utils
import vocabs
from data_loader import PTB, convert_seq, load_ptb_dataset
from seq2seq import Seq2Seq

argparser = argparse.ArgumentParser("PA4 Inference Argparser")
argparser.add_argument(
  '--checkpoint', required=True, help='checkpoint to load model from')

################################### PREDICT ####################################
def predict(model, dataset, tree_itos, device):
  """make model prediction on `dataset`

  batch size is assumed to be 1 during prediction. While this allows us to not
  worry about padding, this is slower than if batch size were bigger than 1

  this means each data instance consists of:
  1. `sent`: [BOS, tok_1, tok_2, ...]
  2. `tree`: [BOS, tok_1, tok_2, ..., EOS]
  3. `sent_length`: single int

  Args:
    model: Seq2Seq model
    dataset: PTB dataset
    tree_itos: tree int-to-str vocab
    device: device

  Returns:
    Tuple of int predictions and accuracy
  """
  preds = []
  num_correct = num_tokens = 0
  with torch.no_grad():
    for i in tqdm.trange(len(dataset)):
      # `sent` and `tree`: 1D tensors with no batch_size
      sent, tree, sent_length = utils.to_device(dataset[i], device)

      # unsqueeze of `sent` and list wrapping of `sent_length` account for the
      # missing batch_size dimension
      pred = model.parse(sent.unsqueeze(0), [sent_length])
      pred_str = convert_seq(pred, tree_itos, return_str=True)
      preds.append(pred_str)

      # gold target as a list; ignore BOS at the beginning and EOS at the end
      gold = tree[1:-1].tolist()

      # token-level accuracy
      pred_len, gold_len = len(pred), len(gold)
      num_correct += sum([pred[i]==gold[i] for i in range(min(pred_len, gold_len))])
      num_tokens += gold_len

  acc = num_correct / num_tokens

  return preds, acc

def run_inference(args):
  """main inference function"""
  # sentinel
  assert os.path.exists(args.model_dir), f"`model_dir` at {args.model_dir} doesn't exist"

  ### 0. setup
  utils.set_seed(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("\nUsing device:", device)

  ### 1. load vocab
  sent_itos, sent_stoi, tree_itos, tree_stoi  = vocabs.load_vocabs(
    args.data_dir, sent_threshold=args.sent_threshold, tree_threshold=args.tree_threshold)

  # optionally load glove, which may modify Sentence itos and stoi
  glove = None
  if args.glove_dir is not None:
    glove, sent_itos, sent_stoi = vocabs.init_glove(args.glove_dir, args.glove_name,
                                                    args.embed_dim, sent_itos,
                                                    strategy=args.glove_strategy,
                                                    with_torchtext=args.with_torchtext)

  ### 2. load PTB
  test_raw = load_ptb_dataset(args.data_dir, C.TEST)
  test = PTB(C.TEST, test_raw, sent_stoi, tree_stoi)

  ### 3. model init
  model = Seq2Seq(
    model=args.model,
    sent_stoi=sent_stoi,
    tree_stoi=tree_stoi,
    embed_dim=args.embed_dim,
    hidden_dim=args.hidden_dim,
    num_layers=args.num_layers,
    dropout=args.dropout,
    rnn_type=args.rnn,
    glove=glove,
    finetune_glove=False,
    device=device
  )

  ### 4. load state dict
  ckpt = torch.load(args.checkpoint, map_location='cpu')  # always load initially to RAM
  model.load_state_dict(ckpt['model'])
  print("Successfully loaded checkpoint from {}: Dev Token-Level ACC {:.3f} | BLEU {:.2f}".format(
    args.checkpoint, ckpt['acc'], ckpt['bleu']))

  # move all model parameters to `device`
  model.to(device)

  ### 5. run inference
  print("\nBegin Inference..")
  preds, acc = predict(model, test, tree_itos, device)

  ### 6. export predictions (this will overwrite existing prediction file)
  preds_path = os.path.join(args.model_dir, C.TREE_PRED_TEMPLATE.format('test', ''))
  print("Exporting test predictions at", preds_path)
  utils.export_txt(preds, preds_path)

  ### 7. bleu
  bleu = utils.raw_corpus_bleu(preds, test_raw[1])
  print("\nTest Token-level Accuracy: {:.3f} | BLEU: {:.2f}".format(acc, bleu))

def main():
  """inference script entry point"""
  begin = time.time()

  pred_args = argparser.parse_args()

  # sentinel
  assert os.path.exists(pred_args.checkpoint), f'{pred_args.checkpoint} does not exist'
  model_dir = os.path.dirname(pred_args.checkpoint)

  # load args from training
  args_path = os.path.join(model_dir, C.ARGS_FNAME)
  prev_args_dict = utils.load_json(args_path)

  # from dict into Namerspace for consistency with usage in `training.py`
  args = argparse.Namespace()
  args.__dict__.update(prev_args_dict)
  args.checkpoint = pred_args.checkpoint
  utils.display_args(args)

  run_inference(args)

  utils.display_exec_time(begin, "PA4 Inference")

if __name__ == '__main__':
  main()
