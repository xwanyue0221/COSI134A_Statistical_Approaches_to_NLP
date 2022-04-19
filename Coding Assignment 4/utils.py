#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 11/14/20 9:06 PM
"""utility functions

updated for Fall 2021"""
import json
import os
import random
import time

import sacrebleu
import torch

#################################### EXPORT ####################################
def export_json(obj, fpath):
  """exports a json obj at fpath"""
  assert type(obj) is dict

  with open(fpath, 'w') as f:
    json.dump(obj, f, indent=4)

def export_txt(obj, fpath, delimiter='\n'):
  """exports a txt obj joined by delimiter at fpath"""
  if type(obj) is list:
    obj = delimiter.join(obj)

  assert type(obj) is str

  with open(fpath, 'w') as f:
    f.write(obj)

##################################### LOAD #####################################
def load_json(fpath):
  """loads a json dict obj from path"""
  assert os.path.exists(fpath) # sentinel

  with open(fpath, 'r') as f:
    out = json.load(f)

  return out

def load_txt(fpath, delimiter='\n'):
  """loads a txt obj from path, which is subsequently split by delimiter"""
  assert os.path.exists(fpath) # sentinel

  with open(fpath, 'r') as f:
    out = f.read().split(delimiter)

  return out

##################################### MISC #####################################
def display_exec_time(begin, msg_prefix=""):
  """displays the script's execution time"""
  exec_time = time.time() - begin

  msg_header = "Execution Time:"
  if msg_prefix:
    msg_header = f"{msg_prefix.strip()} {msg_header}"

  if exec_time > 60:
    et_m, et_s = int(exec_time / 60), int(exec_time % 60)
    print("\n%s %dm %ds" % (msg_header, et_m, et_s))
  else:
    print("\n%s %.2fs" % (msg_header, exec_time))

def display_args(args):
  """displays all flags"""
  print("********* FLAGS *********")
  for k, v in vars(args).items():
    print(f"  {k}: {v}")

def raw_corpus_bleu(hypotheses, references, offset=0.01):
  """
  Simple wrapper around sacreBLEU's BLEU without tokenization and smoothing.

  from https://github.com/awslabs/sockeye/blob/master/sockeye/evaluate.py#L37

  :param hypotheses: Hypotheses stream.
  :param references: Reference stream.
  :param offset: Smoothing constant.
  :return: BLEU score
  """
  return sacrebleu.raw_corpus_bleu(hypotheses, [references], smooth_value=offset).score

def set_seed(seed: int):
  """for replicability"""
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def to_device(data, device):
  """moves incoming tensors to the device"""
  return [x.to(device) if torch.is_tensor(x) else x for x in data]
