#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 9/20/20 10:40 AM
"""Entry point for PA2

Feel free to change/restructure the entire project if you wish
"""
import time
import json
import torch.nn
import numpy as np
from corpus import PDTBCorpus
from torchtext.vocab import GloVe
from model import LR, LR_Dense, LR_Glove, LR_CNN
from scorer import accuracy, prf_for_one_tag, prf

def display_exec_time(begin: float, msg_prefix: str = ""):
  """Displays the script's execution time

  Args:
    begin (float): time stamp for beginning of execution
    msg_prefix (str): display message prefix
  """
  exec_time = time.time() - begin

  msg_header = "Execution Time:"
  if msg_prefix:
    msg_header = msg_prefix.rstrip() + " " + msg_header

  if exec_time > 60:
    et_m, et_s = int(exec_time / 60), int(exec_time % 60)
    print("%s %dm %ds" % (msg_header, et_m, et_s))
  else:
    print("%s %.2fs" % (msg_header, exec_time))

def run_scorer(preds_file: str):
  """Automatically runs `scorer.py` on model predictions. You don't need to use
  this code if you'd rather run `scorer.py` manually.

  Args:
    preds_file (str): path to model's prediction file
  """
  import sys
  import subprocess

  python = 'python3' # TODO: change this to your python command
  scorer = './scorer.py'
  gold = '../data/pdtb/test.json'
  auto = preds_file
  command = "{} {} {} {}".format(python, scorer, gold, auto)

  print("Running scorer with command:", command)
  proc = subprocess.Popen(
    command, stdout=sys.stdout, stderr=sys.stderr, shell=True,
    universal_newlines=True
  )
  proc.wait()

def run():
  begin = time.time()

  '''
  Create padding and trimming functions
  '''
  def pad_trim(dataset, max_len, vocab):
    arg1_L = []
    arg2_L = []
    conn_L = []

    for each in dataset:
      arg1 = np.array([vocab[0][word] for word in each.features[0] if word in vocab[0]])
      if len(arg1) > max_len:
        arg1 = arg1[:max_len]
      arg1 = np.append(arg1, [len(vocab[0])] * (max_len - len(arg1)))
      arg1_L.append(arg1)

      conn = np.array([vocab[1][word] for word in each.features[1] if word in vocab[1]])
      if len(conn) > max_len:
        conn = conn[:max_len]
      conn = np.append(conn, [len(vocab[1])] * (max_len - len(conn)))
      conn_L.append(conn)

      arg2 = np.array([vocab[2][word] for word in each.features[2] if word in vocab[2]])
      if len(arg2) > max_len:
        arg2 = arg2[:max_len]
      arg2 = np.append(arg2, [len(vocab[2])] * (max_len - len(arg2)))
      arg2_L.append(arg2)

    return np.array(arg1_L), np.array(conn_L), np.array(arg2_L)

  def pad_trim_feature_vector(dataset, max_len, vocab):
    arg1_L = []
    arg2_L = []
    conn_L = []

    for each in dataset:
      arg1 = each.feature_vector[0]
      if len(arg1) > max_len:
        arg1 = arg1[:max_len]
      arg1 = np.append(arg1, [len(vocab[0])] * (max_len - len(arg1)))
      arg1_L.append(arg1)

      conn = each.feature_vector[1]
      if len(conn) > max_len:
        conn = conn[:max_len]
      conn = np.append(conn, [len(vocab[1])] * (max_len - len(conn)))
      conn_L.append(conn)

      arg2 = each.feature_vector[2]
      if len(arg2) > max_len:
        arg2 = arg2[:max_len]
      arg2 = np.append(arg2, [len(vocab[2])] * (max_len - len(arg2)))
      arg2_L.append(arg2)

    return np.array(arg1_L), np.array(conn_L), np.array(arg2_L)

  def get_glove_dataset(dataset, max_len, glove_vocab):
    arg1_L = []
    arg2_L = []
    conn_L = []

    for each in dataset:
      arg1 = np.array([glove_vocab[word.lower()] for word in each.features[0] if word.lower() in glove_vocab])
      if len(arg1) > max_len:
        arg1 = arg1[:max_len]
      arg1 = np.append(arg1, [len(glove_vocab)-1] * (max_len - len(arg1)))
      arg1_L.append(arg1)

      conn = np.array([glove_vocab[word.lower()] for word in each.features[1] if word.lower() in glove_vocab])
      if len(arg1) > max_len:
        conn = conn[:max_len]
      conn = np.append(conn, [len(glove_vocab)-1] * (max_len - len(conn)))
      conn_L.append(conn)

      arg2 = np.array([glove_vocab[word.lower()] for word in each.features[2] if word.lower() in glove_vocab])
      if len(arg2) > max_len:
        arg2 = arg2[:max_len]
      arg2 = np.append(arg2, [len(glove_vocab)-1] * (max_len - len(arg2)))
      arg2_L.append(arg2)

    return np.array(arg1_L), np.array(conn_L), np.array(arg2_L)

  def get_glove_vocabs(embedding_module):
    glove_vocab = embedding_module.stoi
    glove_vocab.update({'<unk>': embedding_module.vectors.shape[0]})
    pretrained_embeddings = embedding_module.vectors
    pretrained_embeddings = torch.cat((pretrained_embeddings, torch.zeros(1,pretrained_embeddings.shape[1])))
    return glove_vocab, pretrained_embeddings

  '''
    Load data and create data iterators
  '''
  corpus = PDTBCorpus('../data/pdtb')
  leved_labels = corpus.labels
  num_features = corpus.num_features
  embedding_glove200 = GloVe(name="6B", dim=200)
  glove_vocab, pretrained_embeddings = get_glove_vocabs(embedding_glove200)

  '''
    Initialize Logistic Regression Model
  '''
  learning_rate = 0.01
  num_epochs = 50
  batch_size = 50
  '''
    Write a training loop
  '''
  def shuffle(arg1, conn, arg2, b):
    assert len(arg1) == len(b)
    idx = np.random.permutation(len(arg1))
    return arg1[idx], conn[idx], arg2[idx], b[idx]

  def batchify(instances, batch_size):
    """splits instances into batches, each of which contains at most batch_size"""
    batches = [instances[i:i+batch_size] if i+batch_size <= len(instances)
               else instances[i:] for i in range(0, len(instances), batch_size)]
    return batches

  def train(model, arg1, conn, arg2, labels):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    for i in range(num_epochs):
      total_loss = 0
      arg1_, conn_, arg2_, labels_ = shuffle(arg1, conn, arg2, labels)

      for arg1Batch, connBatch, arg2Batch, y in zip(batchify(arg1_, batch_size), batchify(conn_, batch_size), batchify(arg2_, batch_size), batchify(labels_, batch_size)):
        optimizer.zero_grad()
        arg1Batch = torch.from_numpy(arg1Batch)
        connBatch = torch.from_numpy(connBatch)
        arg2Batch = torch.from_numpy(arg2Batch)
        y = torch.from_numpy(y)

        # forward computation
        logits = model.forward(arg1Batch, connBatch, arg2Batch)
        mb_loss = loss(logits, y)
        total_loss += mb_loss      # accumulate losses

        # backward computation
        mb_loss.backward()
        optimizer.step()
      print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, num_epochs, total_loss))

  def evaluate(model, arg1, conn, arg2, labels):
    with torch.no_grad():
      arg1 = torch.from_numpy(arg1)
      conn = torch.from_numpy(conn)
      arg2 = torch.from_numpy(arg2)
      y = torch.as_tensor(labels)

      logits = model.forward(arg1, conn, arg2)
      print(logits.shape)
      softmax = torch.nn.Softmax(dim=1)
      probas = softmax(logits)
      print(probas.shape)
      preds = probas.argmax(dim=1)
      print(preds.shape)

      num_correct = (preds == y).sum()
      acc = num_correct.item() / len(y)
      print("DEV Acc: {:.3f}".format(acc))
      return preds

  def evaluate2(model, arg1, conn, arg2, labels):
    with torch.no_grad():
      arg1 = torch.from_numpy(arg1)
      conn = torch.from_numpy(conn)
      arg2 = torch.from_numpy(arg2)
      y = torch.as_tensor(labels)

      logits = model.forward(arg1, conn, arg2)
      _, preds = torch.max(logits, dim=1)
      num_correct = (preds == y).sum()
      acc = num_correct.item() / len(y)
      print("TEST Acc: {:.3f}".format(acc))
      return preds

  '''
  Datasets Conversion and Model Training
  '''
  print("Strating Model Training:")
  # padding and trimming the sparse vectors for training datasets and dev datasets
  train_SArg1,train_SConn,train_sArg2 = pad_trim_feature_vector(corpus.train, 300, corpus.vocabs)
  train_labels = np.array([leved_labels.index(each.label) for each in corpus.train])
  dev_SArg1,dev_SConn,dev_sArg2 = pad_trim_feature_vector(corpus.dev, 300, corpus.vocabs)
  dev_labels = np.array([leved_labels.index(each.label) for each in corpus.dev])
  print("LR Model: ")
  LRmodel = LR(labels=leved_labels, num_length=300)
  train(LRmodel, train_SArg1,train_SConn,train_sArg2, train_labels)
  evaluate(LRmodel, dev_SArg1,dev_SConn,dev_sArg2, dev_labels)

  # train_DArg1,train_DConn,train_DArg2 = pad_trim(corpus.train, 11, corpus.vocabs)
  # dev_DArg1,dev_DConn,dev_DArg2 = pad_trim(corpus.dev, 11, corpus.vocabs)
  # print("LR Dense Model: ")
  # LR_DenseModel = LR_Dense(labels=leved_labels, vocab_size = num_features, num_embedding = 200, num_hidden= 300)
  # train(LR_DenseModel, train_DArg1, train_DConn, train_DArg2, train_labels)
  # evaluate(LR_DenseModel, dev_DArg1, dev_DConn, dev_DArg2, dev_labels)

  # # padding and trimming the dense vectors for training datasets and dev datasets using GloVe
  # train_GArg1,train_GConn,train_GArg2 = get_glove_dataset(corpus.train, 30, glove_vocab)
  # dev_GArg1,dev_GConn,dev_GArg2 = get_glove_dataset(corpus.dev, 30, glove_vocab)
  #
  # print("LR Dense Model with Glove Embedding: ")
  # LR_DenseGlove = LR_Glove(labels = leved_labels, num_features = embedding_glove200.dim*3, GloVe_vectors = pretrained_embeddings, num_hidden=300)
  # train(LR_DenseGlove, train_GArg1,train_GConn,train_GArg2, train_labels)
  # evaluate(LR_DenseGlove, dev_GArg1,dev_GConn,dev_GArg2, dev_labels)
  # print("CNN Model: ")
  # LR_CNNModel = LR_CNN(labels = leved_labels, num_features = embedding_glove200.dim*3, GloVe_vectors = pretrained_embeddings, kernel_size=3, hidden_size=300)
  # train(LR_CNNModel, train_GArg1,train_GConn,train_GArg2, train_labels)
  # evaluate2(LR_CNNModel, dev_GArg1,dev_GConn,dev_GArg2, dev_labels)

  '''
  Make predictions on test set and export them as JSON
  '''
  DocID_List = [each.docI for each in corpus.test]
  type_List = [each.type for each in corpus.test]
  # experiemnt 2
  test_SArg1,test_SConn,test_sArg2 = pad_trim_feature_vector(corpus.test, 300, corpus.vocabs)
  test_labels = np.array([leved_labels.index(each.label) for each in corpus.test])
  # experiemnt 2
  test_DArg1,test_DConn,test_DArg2 = pad_trim(corpus.test, 11, corpus.vocabs)
  # experiment 3 & 4
  test_GArg1,test_GConn,test_GArg2 = get_glove_dataset(corpus.test, 30, glove_vocab)

  print("Predicting Test Dataset:")
  def write_json(predict_list, DocID_List, type_List, arg1, conn, arg2, name):
    data = []

    for idx in range(len(predict_list)):
      docID = DocID_List[idx]
      type = type_List[idx]

      data.append(
        {'Arg1': {'TokenList': arg1[idx].tolist()},
        'Arg2': {'TokenList': arg2[idx].tolist()},
        'Connective': {'TokenList': conn[idx].tolist()},
        'DocID': docID,
        'Sense': [leved_labels[predict_list[idx].tolist()]],
        'Type': type})

    with open(name, 'w') as json_file:
      for each in data:
        json.dump(each, json_file)
        json_file.write("\n")

  # '''Experiment 1'''
  # LR_predict = evaluate(LRmodel, test_SArg1, test_SConn, test_sArg2, test_labels)
  # write_json(LR_predict, DocID_List, type_List, test_SArg1, test_SConn, test_sArg2, "LR.json")
  # '''Experiemnt 2'''
  # LR_Dense_predict = evaluate(LR_DenseModel, test_DArg1,test_DConn,test_DArg2, test_labels)
  # write_json(LR_Dense_predict, DocID_List, type_List, test_DArg1,test_DConn,test_DArg2, "LR_Dense.json")
  # '''Experiemnt 3'''
  # LR_Glove_predict = evaluate(LR_DenseGlove, test_GArg1,test_GConn,test_GArg2, test_labels)
  # write_json(LR_Glove_predict, DocID_List, type_List, test_GArg1,test_GConn,test_GArg2, "LR_GloVe.json")
  # '''Experiemnt 4'''
  # LR_CNN_predict = evaluate2(LR_CNNModel, test_GArg1,test_GConn,test_GArg2, test_labels)
  # write_json(LR_CNN_predict, DocID_List, type_List, test_GArg1,test_GConn,test_GArg2, "LR_CNN.json")

  '''
  Run `scorer.py` using exported JSON from Step 5
  '''
  print("Scorer Result For LR Desne Model: ")
  run_scorer("LR_Dense.json")
  print("Scorer Result For LR GloVe Model: ")
  run_scorer("LR_GloVe.json")
  print("Scorer Result For LR CNN: ")
  run_scorer("LR_CNN.json")


if __name__ == '__main__':
  run()
