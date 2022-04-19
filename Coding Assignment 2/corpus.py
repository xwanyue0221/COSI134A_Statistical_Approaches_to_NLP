#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:02 PM
import codecs
import json
import os
import string
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from data import to_level
from collections import Counter

SENSES = [
  'Temporal',
  'Temporal.Asynchronous',
  'Temporal.Asynchronous.Precedence',
  'Temporal.Asynchronous.Succession',
  'Temporal.Synchrony',
  'Contingency',
  'Contingency.Cause',
  'Contingency.Cause.Reason',
  'Contingency.Cause.Result',
  'Contingency.Condition',
  'Comparison',
  'Comparison.Contrast',
  'Comparison.Concession',
  'Expansion',
  'Expansion.Conjunction',
  'Expansion.Instantiation',
  'Expansion.Restatement',
  'Expansion.Alternative',
  'Expansion.Alternative.Chosen alternative', # can ignore `alternative`, will be dropped during processing
  'Expansion.Exception',
  'EntRel'
]

##################################### DATA #####################################
class Document(ABC):
  """Base Document"""
  @abstractmethod
  def featurize(self):
    pass

  @abstractmethod
  def featurize_vector(self, vocab=None) -> np.ndarray:
    pass

class PDTBRelation(Document):
  """A single PDTB relation data instance"""
  def __init__(self,
               docI: str,
               type: str,
               arg1: str,
               arg2: str,
               connective: Optional[str] = None,
               label: str = None):
    # raw data and label
    self.arg1 = arg1
    self.arg2 = arg2
    self.connective = connective # may be empty for implicit relations
    self.label = label
    self.docI = docI
    self.type = type
    # raw features
    self.features = self.featurize() # type: Tuple[List[str], List[str], List[str]]
    # numeric features as numpy ndarray
    self.feature_vector = [] # type: np.ndarray

  def featurize(self) -> Tuple[List[str], List[str], List[str]]:
    """converts each arg1, arg2 and connective into features

    Returns:
      Tuple[List[str], List[str], List[str]]: unigram for each of arg1, arg2 and
      connective
    """
    return self.arg1.split(), self.arg2.split(), self.connective.split()

  def featurize_vector(self, vocab=None) -> np.ndarray:
    """turn the features of this data instance into a feature vector"""

    def get_vector(idx):
      ind = [vocab[idx][word] for word in self.features[idx] if word in vocab[idx]]
      vec = np.zeros(len(vocab[idx]))
      vec[ind] = 1
      return vec

    self.feature_vector = [get_vector(0), get_vector(1), get_vector(2)]

#################################### CORPUS ####################################
class Corpus(ABC):
  """Base Corpus"""
  def __init__(self,
               data_dir: str,
               max_num_data: int = -1,
               shuffle: bool = False):
    """`self.documents` stores all data instances, regardless of the dataset
    split. This is especially helpful when the dataset does not provide the
    pre-defined dataset splits, as in the Names corpus. For PDTB corpus, on the
    other hand, `self.documents` will be empty.

    `shuffle` is only useful for Names Corpus, and should always be set to True
    as the Names data files are stored alphabetically.

    Args:
      data_dir (str): where to load data from
      max_num_data (int): max number of items to load. -1 if no upper limit
      shuffle (bool): whether to shuffle data during dataset split
    """
    self.documents = []
    self.train = []
    self.dev = []
    self.test = []
    self.max_num_data = max_num_data

    # 1. load data
    self.load(data_dir)
    # 1a. dataset split
    self.train, self.dev, self.test = self.split_corpus(shuffle=shuffle)
    # 2. (optional) compile vocabs
    self.vocabs = self.compile_vocab(most_common_n=1000)
    # 3. convert features into feature vectors
    self.featurize_documents(self.vocabs)

  def featurize_documents(self, vocabs=None):
    """for each data instance, create and cache its feature vector"""
    def get_features(dataset):
      for each in dataset:
        each.featurize_vector(vocab=vocabs)

    print("Start Featurizing Train Datset ")
    get_features(self.train)
    print("Start Featurizing Dev Datset ")
    get_features(self.dev)
    print("Start Featurizing Test Datset ")
    get_features(self.test)

  ############################## ABSTRACT METHODS ##############################
  @abstractmethod
  def compile_vocab(self, most_common_n: int = -1):
    pass

  @abstractmethod
  def load(self, data_dir: str):
    pass

  @abstractmethod
  def split_corpus(self, shuffle: bool = False):
    pass

def load_relations(rel_file: str, sense_level: int = 2) -> List['PDTBRelation']:
  """Loads a single relation json file

  Args:
    rel_file (str):  path to a json to be loaded
    sense_level (int): see `to_level` in `utils.py`

  Returns:
    List['BagOfWords']: loaded data as a list of BagOfWords objects
  """
  documents = []
  with codecs.open(rel_file, encoding='utf-8') as pdtb:
    pdtb_lines = pdtb.readlines()
    for pdtb_line in pdtb_lines:
      rel = json.loads(pdtb_line)

      data = rel['Arg1']['RawText']
      data_1 = rel['Connective']['RawText']
      data_2 = rel['Arg2']['RawText']
      label = to_level(rel['Sense'][0], level=sense_level)
      docI = rel['DocID']
      type = rel['Type']

      document = PDTBRelation(docI, type, data, data_1, data_2, label=label)
      documents.append(document)
  return documents

class PDTBCorpus(Corpus):
  """Penn Discourse TreeBank Corpus"""
  def __init__(self,
               data_dir: str,
               max_num_data: int = -1,
               sense_level: int = 2,
               shuffle: bool = False):
    leveled_senses = list(set([to_level(x, level=sense_level) for x in SENSES]))
    self.labels = leveled_senses
    self.sense_level = sense_level

    super().__init__(data_dir, max_num_data, shuffle)
    num_vocabs = sum(len(x) for x in self.vocabs)
    self.num_features = num_vocabs + 1

  def compile_vocab(self, most_common_n: int = -1):
    """
        compile vocabulary from corpus. Using `most_common_n` highly
    """

    print("Starting vocabulary compiling")
    def get_vocab(idx):
      token_list = []
      for each in self.train:
        token_list.extend(each.features[idx])

      if most_common_n > 0:
        return [word for word, idx in Counter(token_list).most_common(most_common_n)]
      else:
        return list(set(token_list))

    # arg1_vocab = get_vocab(0)
    # connective_vocab = get_vocab(1)
    # arg2_vocab = get_vocab(2)

    arg1_vocab_dict = dict([(word, ind) for ind, word in enumerate(get_vocab(0))])
    arg2_vocab_dict = dict([(word, ind) for ind, word in enumerate(get_vocab(1))])
    conn_vocab_dict = dict([(word, ind) for ind, word in enumerate(get_vocab(1))])
    print("vocab size for arg1:", len(arg1_vocab_dict), " - vocab size for arg2:", len(arg2_vocab_dict), " - vocab size for connective:", len(conn_vocab_dict))
    print("Ending vocabulary compiling")

    return arg1_vocab_dict, conn_vocab_dict, arg2_vocab_dict

  def load(self, data_dir: str):
    """loads from `pdtb` data directory"""
    data = {}
    for filename in os.listdir(data_dir):
      if not filename.endswith('.json'):
        continue
      # print(filename)
      dataset_split = os.path.splitext(filename)[0]
      rel_file = os.path.join(data_dir, filename)
      data[dataset_split] = load_relations(rel_file, self.sense_level) # type: List[PDTBRelation]

    self.train = data['train']
    self.dev = data['dev']
    self.test = data['test']

  def split_corpus(self, shuffle: bool = False) -> Tuple[List[PDTBRelation], List[PDTBRelation], List[PDTBRelation]]:
    """PDTB comes with pre-defined dataset split"""
    return self.train, self.dev, self.test

