#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:02 PM
import codecs
import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from utils import to_level

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
    'Expansion.Alternative.Chosen alternative',  # can ignore `alternative`, will be dropped during processing
    'Expansion.Exception',
    'EntRel'
]

for filename in os.listdir("data/pdtb"):
    if filename.endswith('.txt'):
        rel_file = os.path.join("data/pdtb", filename)

        if os.path.splitext(filename)[0] == "stop_words":
            stop_words = []
            with codecs.open(rel_file, encoding='utf-8') as pdtb:
                pdtb_lines = pdtb.readlines()
                for pdtb_line in pdtb_lines:
                    stop_words.append(pdtb_line.replace("\n", ""))
            pdtb.close()

        if os.path.splitext(filename)[0] == "connectives":
            with codecs.open(rel_file, encoding='utf-8') as pdtb:
                connectives = []
                pdtb_lines = pdtb.readlines()
                for pdtb_line in pdtb_lines:
                    connectives.append(pdtb_line.replace("\n", ""))
            pdtb.close()

##################################### DATA #####################################
class Document(ABC):
    """Base Document"""

    @abstractmethod
    def featurize(self):
        pass

    @abstractmethod
    def featurize_vector(self, vocab=None) -> np.ndarray:
        pass


class Name(Document):
    """A single data instance in Names corpus"""

    def __init__(self,
                 data: str,
                 label: str = None):
        # raw data and label
        self.data = data
        self.label = label
        # raw features
        self.features = self.featurize()  # type: Tuple[str, str]
        # numeric features as numpy ndarray, to be initialized when building corpus
        self.feature_vector = self.featurize_vector()  # type: np.ndarray

    def featurize(self) -> Tuple[str, str]:
        """
        converts raw text into a feature
        Returns:
        Tuple[str, str]: first and last letter
        """
        name = self.data
        first, last = name[0], name[-1]
        return first, last

    def featurize_vector(self, vocab=None) -> np.ndarray:
        """
        turn the features of this data instance into a feature vector
        Returns: np.ndarray
        """
        features = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if self.features[1] == letter:
                features[letter] = 1
            else:
                features[letter] = 0

        for LETTER in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if self.features[0] == LETTER:
                features[LETTER] = 1
            else:
                features[LETTER] = 0

        data = list(features.items())
        data = [each[1] for each in data]
        character_array = np.array(data)
        character_array = np.append(character_array, 1)
        return character_array

class PDTBRelation(Document):
    """A single PDTB relation data instance"""

    def __init__(self,
                 arg1: str,
                 arg2: str,
                 connective: Optional[str] = None,
                 label: str = None):
        # raw data and label
        self.arg1 = arg1
        self.arg2 = arg2
        self.connective = connective  # may be empty for implicit relations
        self.label = label

        # raw features
        self.features = self.featurize()  # type: Tuple[List[str], List[str], List[str]]

        # numeric features as numpy ndarray
        self.feature_vector = self.featurize_vector() # type: np.ndarray

    def featurize(self) -> Tuple[List[str], List[str], List[str]]:
        """converts each arg1, arg2 and connective into features

        Returns:
            Tuple[List[str], List[str], List[str]]: unigram for each of arg1, arg2 and connective
        """
        return self.arg1.split(), self.arg2.split(), self.connective.split()

    def featurize_vector(self, vocab=None) -> np.ndarray:
        """turn the features of this data instance into a feature vector"""

        # unigram featralization
        features = {}
        for each in self.features:
            for word in each:
                word = word.lower()
                word = word.replace("'s", '')
                word = word.replace("'t", '')
                word = word.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'))
                if word.isalpha() is True and word is not None:
                    if word not in stop_words:
                        if word not in features:
                            features[word] = 1
                        else:
                            features[word] += 1

        for each in self.features:
            sentence = []
            for word in each:
                word = word.lower()
                word = word.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'))
                sentence.append(word)

            # bigram featralization
            for idx in range(0, len(sentence)-2):
                if sentence[idx] != "" and sentence[idx+1] != "":
                    bigram = sentence[idx] + " " + sentence[idx+1]
                    if bigram not in features:
                        features[bigram] = 1
                    else:
                        features[bigram] += 1
            # trigram featralization
            for idx in range(0, len(sentence)-3):
                if sentence[idx] != "" and sentence[idx+1] != "" and sentence[idx+2] != "":
                    trigram = sentence[idx] + " " + sentence[idx+1] + " " + sentence[idx+2]
                    if trigram not in features:
                        features[trigram] = 1
                    else:
                        features[trigram] += 1

        data = list(features.items())
        character_array = np.array(data)
        return character_array

#################################### CORPUS ####################################
class Corpus(ABC):
    """Base Corpus"""

    def __init__(self,
                 data_dir: str,
                 max_num_data: int = -1,
                 shuffle: bool = False):
        """
        `self.documents` stores all data instances, regardless of the dataset
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

        # 1. load data and stop words
        self.load(data_dir)
        # 1a. dataset split
        self.train, self.dev, self.test = self.split_corpus(shuffle=shuffle)

        # 2. (optional) compile vocabs
        self.vocabs = self.compile_vocab(most_common_n=600)

        # 3. convert features into feature vectors
        self.featurize_documents(self.vocabs)

    def featurize_documents(self, vocabs=None):
        """for each data instance, create and cache its feature vector"""
        if vocabs == None:
            print("Starting Feature Vectorization - Name")
            for each in self.documents:
                each_array = []
                for label in ["male", "female"]:
                    label_idx = ["male", "female"].index(label)
                    featurevectors = np.append(np.zeros(shape=(label_idx, 53), dtype=int), each.feature_vector)
                    featurevectors = np.append(featurevectors, np.zeros(shape=((1-label_idx), 53), dtype=int))
                    each_array.append(featurevectors)
                each.feature_vector = np.array(each_array)

        else:
            for data in [self.train, self.dev, self.test]:
                print("Starting Feature Vectorization - Discourse")
                for each in data:
                    each_array = []
                    fit_array = dict((el, 0) for el in vocabs)
                    for word,count in each.feature_vector:
                        if word in fit_array.keys():
                            fit_array[word] = 1
                    vector = np.array(list(fit_array.values()))
                    vector = np.append(vector, 1)

                    for label in self.labels:
                        label_idx = self.labels.index(label)
                        featurevectors = np.append(np.zeros(shape=(label_idx, len(vocabs)+1), dtype=int), vector)
                        featurevectors = np.append(featurevectors, np.zeros(shape=(len(self.labels)-label_idx-1, len(vocabs)+1)))
                        each_array.append(featurevectors)
                    each.feature_vector = np.array(each_array)

                print("Ending Feature Vectorization")
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


class NamesCorpus(Corpus):
    """Names Corpus"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = ['male', 'female']
        self.num_features = 53  # 52 alphabets + bias
    """A collection of names, labeled by gender. See names/README for copyright and license."""

    def compile_vocab(self, most_common_n: int = -1):
        """
        We don't compile vocab for Names corpus since we know in theory what the
        entire vocab space consists of: 52 upper- and lower-case alphabets
        """
        return

    def load(self, data_dir: str):
        """loads from `names` data directory"""
        for filename in os.listdir(data_dir):
            if not filename.endswith('.txt'):  # skip README
                continue

            data_file = os.path.join(data_dir, filename)
            label = os.path.splitext(filename)[0]

            with open(data_file, "r") as file:
                for i, line in enumerate(file):
                    data = line.strip()
                    document = Name(data, label=label)
                    self.documents.append(document)

                    if 0 < self.max_num_data < i:
                        break

    def split_corpus(self, shuffle: bool = False) -> Tuple[List[Name], List[Name], List[Name]]:
        """
        splits corpus into train, dev and test
        Args:
            shuffle (bool): whether to shuffle data before split
        Returns:
            Tuple[List[Name], List[Name], List[Name]]: split data
        """
        if shuffle:
            random.shuffle(self.documents)
        train = self.documents[:5000]
        dev = self.documents[5000:6000]
        test = self.documents[6000:]
        return train, dev, test

def load_relations(rel_file: str, sense_level: int = 2) -> List['PDTBRelation']:
    """Loads a single relation json file

    Args:
    rel_file (str):  path to a json to be loaded
    sense_level (int): see `to_level` in `utils.py`

    Returns: List['BagOfWords']: loaded data as a list of BagOfWords objects
  """
    documents = []
    with codecs.open(rel_file, encoding='utf-8') as pdtb:
        pdtb_lines = pdtb.readlines()
        for pdtb_line in pdtb_lines:
            rel = json.loads(pdtb_line)

            data = rel['Arg1']['RawText']
            data_1 = rel['Connective']['RawText']
            data_2 = rel['Arg2']['RawText']

            # when there are multiple senses, we will only use the first one
            label = to_level(rel['Sense'][0], level=sense_level)

            document = PDTBRelation(data, data_2, data_1, label=label)
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

        num_vocabs = len(self.vocabs)
        self.num_features = num_vocabs + 1

    def compile_vocab(self, most_common_n: int = -1):
        """
        compile vocabulary from corpus. Using `most_common_n` highly
       """
        print("Starting vocabulary compiling")
        wordCount = {}
        for each in self.train:
            for word, count in each.feature_vector:
                if word not in wordCount:
                    wordCount[word] = int(count)
                else:
                    wordCount[word] = wordCount[word] + int(count)

        filtered_d = dict((k, wordCount[k]) for k in connectives if k in wordCount)
        unwanted = set(filtered_d.keys())
        for unwanted_key in unwanted: del wordCount[unwanted_key]

        if most_common_n > 0:
            most_common = sorted(wordCount, key=wordCount.get, reverse=True)[:most_common_n-len(unwanted)]
        most_common = [each for each in filtered_d.keys()] + most_common

        print("Ending vocabulary compiling")
        return most_common

    def load(self, data_dir: str):
        """loads from `pdtb` data directory"""
        data = {}

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                dataset_split = os.path.splitext(filename)[0]
                rel_file = os.path.join(data_dir, filename)
                data[dataset_split] = load_relations(rel_file, self.sense_level)  # type: List[PDTBRelation]

        self.train = data['train']
        self.dev = data['dev']
        self.test = data['test']

    def split_corpus(self, shuffle: bool = False) -> Tuple[List[PDTBRelation], List[PDTBRelation], List[PDTBRelation]]:
        """PDTB comes with pre-defined dataset split"""
        return self.train, self.dev, self.test
