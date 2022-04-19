#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/14/20 8:59 PM
"""training

the default hyperparams were chosen to keep it as simple as possible; be sure to
try different hyperparameter values for improved results

updated for Fall 2021"""
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import consts as C
import utils
import vocabs
from data_loader import PTB, init_data_loader, load_ptb_dataset
from inference import predict
from seq2seq import Seq2Seq

argparser = argparse.ArgumentParser("PA4 Training Argparser")

# input and output dirs
argparser.add_argument(
  '--data_dir', default='./outputs/ptb', help='path to processed ptb input data')
argparser.add_argument(
  '--model_dir', default='./outputs/model', help='path to model outputs')

# vocab flags
argparser.add_argument(
  '--glove_dir', help='path to glove dir; must be specified when using GloVe')
argparser.add_argument(
  '--glove_name', type=str.upper, choices=C.GLOVE_NAMES,
  help='name of the pre-trained GloVe vectors. If None, will attempt to infer automatically')
argparser.add_argument(
  '--glove_strategy', default=C.KEEP_OVERLAP, choices=C.GLOVE_STRATEGIES,
  help='how to handle vocabs when using GloVe. See `consts.py` for details')
argparser.add_argument(
  '--with_torchtext', action='store_true',
  help='whether to use torchtext when loading GloVe. Note that by default torchtext is not required')
argparser.add_argument(
  '--finetune_glove', action='store_true', help='whether to make GloVe embeddings trainable')
argparser.add_argument(
  '--sent_threshold', type=int, default=5,
  help='minimum number of occurrences for sentence tokens to keep as vocab')
argparser.add_argument(
  '--tree_threshold', type=int, default=5,
  help='minimum number of occurrences for tree tokens to keep as vocab')

# model flags
argparser.add_argument(
  '--model', type=str.lower, default=C.VANILLA, choices=C.SEQ2SEQ_TYPES,
  help='which seq2seq model to train. See `consts.py` for details')
argparser.add_argument(
  '--embed_dim', type=int, default=100, help='embedding dimension')
argparser.add_argument(
  '--rnn', type=str.lower, default=C.RNN, choices=C.RNN_TYPES,
  help='type of rnn to use in encoder and decoder')
argparser.add_argument(
  '--num_layers', type=int, default=1, help='number of rnn layers in encoder and decoder')
argparser.add_argument(
  '--hidden_dim', type=int, default=128, help='rnn hidden dimension')
argparser.add_argument(
  '--dropout', type=float, default=0.1, help='dropout probability')

# experiment flags
argparser.add_argument(
  '--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument(
  '--eval_every', type=int, default=10, help='interval of epochs to perform evaluation on dev set')
argparser.add_argument(
  '--batch_size', type=int, default=128, help='size of mini batch')
argparser.add_argument(
  '--learning_rate', type=float, default=0.005, help='learning rate')
argparser.add_argument(
  '--teacher_forcing_ratio', type=float, default=0.75,
  help='teacher forcing ratio, where higher means more teacher forcing; cannot be 1 with attentional decoders')
argparser.add_argument(
  '--checkpoint', help='if specified, attempts to load model params and resume training')
argparser.add_argument(
  '--seed', type=int, default=1334, help='seed value for replicability')

#################################### TRAIN #####################################
def run_training(args):
  """main training function"""
  # sentinels
  assert os.path.exists(args.data_dir), "be sure to run `prepare_data.py` first"
  assert args.glove_dir is None or os.path.exists(args.glove_dir)

  ### 0. setup
  os.makedirs(args.model_dir, exist_ok=True)
  utils.set_seed(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("\nUsing device:", device)

  # init model state dict filename
  model_path_template = os.path.join(args.model_dir, C.MODEL_PT_TEMPLATE)

  ### 1. load vocab
  sent_itos, sent_stoi, tree_itos, tree_stoi = vocabs.load_vocabs(
    args.data_dir, sent_threshold=args.sent_threshold, tree_threshold=args.tree_threshold)

  # optionally load glove, which may modify Sentence itos and stoi
  glove = None
  if args.glove_dir is not None:
    glove, sent_itos, sent_stoi = vocabs.init_glove(args.glove_dir, args.glove_name,
                                                    args.embed_dim, sent_itos,
                                                    strategy=args.glove_strategy,
                                                    with_torchtext=args.with_torchtext)

  ### 2. load PTB
  dev_raw = load_ptb_dataset(args.data_dir, C.DEV)
  dev = PTB(C.DEV, dev_raw, sent_stoi, tree_stoi)
  training_raw = load_ptb_dataset(args.data_dir, C.TRAIN)
  training = PTB(C.TRAIN, training_raw, sent_stoi, tree_stoi)

  ### 3. training data loaders init
  train_dataloader = init_data_loader(training, sent_stoi, tree_stoi, args.batch_size)

  ### 4. model init
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
    finetune_glove=args.finetune_glove,
    device=device
  )
  num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"  Number of Trainable Params: {num_trainable_params:,}")

  epoch = 0
  if args.checkpoint is not None:
    ckpt = torch.load(args.checkpoint, map_location='cpu') # always load initially to RAM
    model.load_state_dict(ckpt['model'])
    epoch = ckpt['epoch']
    print("Resume training with Token-Level ACC {:.3f} | BLEU {:.2f} at epoch {}".format(
      ckpt['acc'], ckpt['bleu'], epoch))

  # move all model parameters to `device`
  model.to(device)

  ### 5. loss and optimizer init
  criterion = nn.CrossEntropyLoss(ignore_index=tree_stoi[C.PAD])
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  ### 6. training loop
  print("\nBegin Training..")
  target_epochs = epoch + args.epochs
  for epoch in range(epoch, target_epochs):
    epoch_loss = 0
    num_correct = num_tokens = 0

    for batch in tqdm.tqdm(train_dataloader, desc=f'[Training {epoch+1}/{target_epochs}]'):
      optimizer.zero_grad()

      # move data to `device`
      sents, trees, sent_lens = utils.to_device(batch, device)

      # since last index of `trees` is always PAD or EOS, there is no need to
      # predict a token with PAD or EOS as inputs when decoding; hence we omit it
      # output logits: (batch_size, tgt_seq_len-1, vocab_size)
      outputs = model(sents, trees[:,:-1], sent_lens,
                      teacher_forcing_ratio=args.teacher_forcing_ratio)

      # decoder output (gold target) that drops BOS at the beginning since BOS is
      # never part of our predictions, (batch_size, tgt_seq_len-1)
      trees_target = trees[:,1:]

      # for the reason why `outputs` are transposed, see `Shape:` in
      # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
      loss = criterion(outputs.transpose(1,2), trees_target)

      # model predictions as ints: (batch_size, tgt_seq_len-1)
      preds = outputs.argmax(-1)

      # includes EOS which also needs to be predicted correctly
      trees_mask = trees_target==tree_stoi[C.PAD]
      num_correct += (preds==trees_target).masked_fill_(trees_mask, False).sum()
      num_tokens += (~trees_mask).sum()

      # optimizer step
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()

      epoch_loss += loss.item()

    acc = (num_correct.item() / num_tokens.item())
    print(' Training Loss: {:.2f} Token-level Accuracy: {:.3f}'.format(epoch_loss, acc))

    ### 7. eval on dev
    if (epoch+1) % args.eval_every == 0:
      print("Begin Inference on Dev..")
      preds, dev_acc = predict(model, dev, tree_itos, device)

      # bleu
      bleu = utils.raw_corpus_bleu(preds, dev_raw[1])
      print('  Dev Token-level Accuracy: {:.3f} | BLEU: {:.2f}'.format(dev_acc, bleu))

      # export model params and other misc info
      model_fpath = model_path_template.format(epoch+1)
      print("Exporting model params at", model_fpath)
      torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'acc': dev_acc,
                  'bleu': bleu},  model_fpath)

      # export dev predictions
      dev_preds_path = os.path.join(args.model_dir, C.TREE_PRED_TEMPLATE.format('dev', f'_{epoch+1}'))
      print("Exporting dev predictions at", dev_preds_path)
      utils.export_txt(preds, dev_preds_path)

      # sample prediction display
      print("Sample Dev Prediction")
      sample_idx = random.randint(0, len(dev)-1)
      print("  [SENT]", dev_raw[0][sample_idx])
      sample_gold = dev_raw[1][sample_idx]
      print(f"  [GOLD: {len(sample_gold.split())} toks]", sample_gold)
      sample_pred = preds[sample_idx]
      print(f"  [PRED: {len(sample_pred.split())} toks]", sample_pred)

def main():
  """training script entry point"""
  begin = time.time()

  args = argparser.parse_args()
  utils.display_args(args)

  # export args as json for reference
  os.makedirs(args.model_dir, exist_ok=True)
  args_path = os.path.join(args.model_dir, C.ARGS_FNAME)
  utils.export_json(vars(args), args_path)

  run_training(args)

  utils.display_exec_time(begin, "PA4 Training")

if __name__ == '__main__':
  main()

