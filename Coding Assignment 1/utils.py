#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 9/4/20 8:35 PM

import numpy as np

def accuracy(classifier, test, verbose=False):
    """quick accuracy calculator

  Args:
    classifier: `trained model`
    test: list of test instances
    verbose: whether to display the % result to terminal or not

  Returns:
    float: accuracy in decimals
  """

    X = []
    y = []
    for idx in range(0, len(test)):
        y.append(test[idx].label)
        X.append(test[idx].feature_vector)
    X = np.array(X)

    predict_y = classifier.classify(X)
    correct = 0
    for idx in range(len(predict_y)):
        if predict_y[idx] == y[idx]:
            correct += 1
    if verbose:
        print("Accuracy: %.2d%% " % (100 * correct / len(y)))
    return float(correct)/len(y)

def to_level(sense: str, level: int = 2) -> str:
    """converts a sense in string to a desired level

  There are 3 sense levels in PDTB:
    Level 1 senses are the single-word senses like `Temporal` and `Contingency`.
    Level 2 senses add an additional sub-level sense on top of Level 1 senses,
      as in `Expansion.Exception`
    Level 3 senses adds yet another sub-level sense, as in
      `Temporal.Asynchronous.Precedence`.

  This function is used to ensure that all senses do not exceed the desired
  sense level provided as the argument `level`. For example,

  >>> to_level('Expansion.Restatement', level=1)
  'Expansion'
  >>> to_level('Temporal.Asynchronous.Succession', level=2)
  'Temporal.Asynchronous'

  When the input sense has a lower sense level than the desired sense level,
  this function will retain the original sense string. For example,

  >>> to_level('Expansion', level=2)
  'Expansion'
  >>> to_level('Comparison.Contrast', level=3)
  'Comparison.Contrast'

  Args:
    sense (str): a sense as given in a `relaions.json` file
    level (int): a desired sense level

  Returns:
    str: a sense below or at the desired sense level
  """
    s_split = sense.split(".")
    s_join = ".".join(s_split[:level])
    return s_join
