#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:23 PM
import time

from corpus import NamesCorpus, PDTBCorpus
from maxent import MaxEntClassifier
from utils import accuracy
import matplotlib.pyplot as plt


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


def run_names():
    """Names Corpus Experiment: use this for debugging your code"""
    print("\n Running on Names Corpus")
    begin = time.time()

    corpus = NamesCorpus('./data/names', shuffle=True)
    clf = MaxEntClassifier(corpus.labels, corpus.num_features)
    clf.train(corpus.train, corpus.dev, verbose=True)
    accuracy(clf, corpus.test, verbose=True)

    display_exec_time(begin, msg_prefix="Names MaxEnt")

def run_pdtb():
    """PDTB Experiment"""
    print("\n Running on PDTB Corpus")
    begin = time.time()

    corpus = PDTBCorpus('data/pdtb')
    clf = MaxEntClassifier(corpus.labels, corpus.num_features)
    clf.train(corpus.train, corpus.dev, verbose=True)
    accuracy(clf, corpus.test, verbose=True)

    display_exec_time(begin, msg_prefix="PDTB MaxEnt")

def main():
    """Main entry"""
    print("PA1 Main. Uncomment below to run each experiment")
    run_names()
    run_pdtb()

if __name__ == '__main__':
    main()