#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:49 PM

import numpy as np


class MaxEntClassifier:

    def __init__(self, labels, num_features, l2: float = 0):
        self.labels = labels
        self.num_labels = len(labels)
        self.num_features = num_features

        self.gamma = l2
    ################################### TRAIN ####################################
    def train(self,
              instances,
              dev_instances=None,
              learning_rate: float = 0.001,
              batch_size: int = 500,
              num_iter: int = 30,
              verbose: bool = False,
              epsilon: float = 0.00001):
        """
        train MaxEnt model with mini-batch stochastic gradient descent.
        """
        if verbose:
            print("Number of features", len(instances[0].feature_vector))
            print(batch_size)
        self.theta = np.zeros((len(self.labels), self.num_features * len(self.labels)))
        cost_history = []

        if len(instances) % batch_size == 0:
            batchCount = len(instances) // batch_size
        if len(instances) % batch_size != 0:
            batchCount = len(instances) // batch_size + 1

        for i in range(num_iter):
            np.random.shuffle(instances)
            cost_epoch = []

            for j in range(batchCount):

                breakSign = False
                # get mini-batch sample via iteration
                batch_instances = instances[j * batch_size: (j + 1) * batch_size]
                batch_instances = np.asarray(batch_instances)
                splitX, splity = self.splitData(batch_instances)

                # compute gradient and update theta parameter
                gradient = self.compute_gradient(splitX, splity)
                gradient = self.gamma * self.theta + gradient
                self.theta -= learning_rate * gradient

                # compute and cache loss info
                X_dev, y_dev = self.splitData(dev_instances)
                cost_dev = self.gamma/2 * np.sum(self.theta ** 2) + self.compute_cross_entropy_loss(X_dev, y_dev)
                cost_epoch.append(cost_dev)
                cost_history.append(cost_dev)
                if verbose:
                    print("No {} iteration: {}".format(j, cost_dev))

                if len(cost_history) > 2 and ((cost_history[-2] - cost_history[-1]) < epsilon):
                    breakSign = True
                    if verbose and breakSign:
                        print("Average Cost for Epoch {}: {}".format(i, sum(cost_epoch) / len(cost_epoch)))
                    break

            if verbose == True and breakSign == False:
                print("Average Cost for Epoch {}: {}".format(i, sum(cost_epoch) / len(cost_epoch)))

        coef = {"theta": self.theta}
        if verbose:
            print(coef)

    ################################## COMPUTE ###################################
    def oneHot(self, y):
        one_hot = np.zeros((len(y), self.num_labels))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def splitData(self, data):
        '''
        :param data: dataset that includes instances
        :return: X: the feature vector of the input data; y: the target value of the input data
        '''
        X = []
        y = []
        for idx in range(0, len(data)):
            y.append(self.labels.index(data[idx].label))
            X.append(data[idx].feature_vector)
        X = np.array(X)
        y = np.array(y)
        y = self.oneHot(y)
        return X, y

    def compute_softmax(self, X):
        '''
        :param x is the sample instance
        :return: the probability distribution of intance X
        '''
        total_instance = []

        for each in range(len(X)):
            label_array = []
            for label in range(self.num_labels):
                e = np.exp(np.dot(X[each][label] - np.max(X[each][label]), self.theta[label]))
                label_array.append(e)

            each_instance_array = []
            for e in label_array:
                each_instance_array.append(np.sum(e/sum(label_array)))
            total_instance.append(each_instance_array)

        return total_instance

    def compute_gradient(self, X, y):
        """
        Calculate gradient for a single data instance
        """
        total_instance = self.compute_softmax(X)
        gradient = np.zeros(shape=(1, self.num_features * self.num_labels), dtype=int)
        for each in range(len(X)):
            gradient = gradient + (-X[each][np.where(y[each] == 1)])
            for label in range(self.num_labels):
                gradient = gradient + total_instance[each][label] * X[each][label]
        return gradient

    def compute_cross_entropy_loss(self, X, y):
        '''
        compute the negative log likelihood of the regression, which is also called loss computation
        :param X: the feature vector of the input data
        :param y: the target value of the input data
        :return: the cost or loss value
        '''
        softmax = self.compute_softmax(X)
        cross_entropy_loss = np.mean(-np.sum(np.log(softmax) * (y), axis=1))
        return cross_entropy_loss

    ################################## PREDICT ###################################
    def classify(self, X):
        """
        predict the most likely label for the given data instance
        :return: the predicted target value
        """
        softmax = self.compute_softmax(X)
        predict_label = [self.labels[each] for each in np.argmax(softmax, axis=1)]
        return predict_label
