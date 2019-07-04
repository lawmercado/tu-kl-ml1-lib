#! /usr/bin/python

import numpy as np
from ml.supervised.classification.common import LinearClassifier
from ml.data.handler import DataHandler
import random


class SVM(LinearClassifier):

    __initial_w = np.array([])
    __initial_b = 0
    __w = np.array([])
    __b = 0
    __psy_constant = 0
    __batch_size = 0
    __num_features = 0
    __num_iterations = 0

    def __init__(self, num_features, psy_constant, batch_size, initial_w=None, initial_b=None, num_iterations=10000):
        if initial_w is None:
            self.__initial_w = np.array([random.randint(-5, 5) for _ in range(num_features)])
        else:
            self.__initial_w = initial_w

        if initial_b is None:
            self.__initial_b = random.randint(-5, 5)
        else:
            self.__initial_b = initial_b

        self.__num_features = num_features
        self.__psy_constant = psy_constant
        self.__batch_size = batch_size
        self.__num_iterations = num_iterations

    def train(self, train_data_handler):
        """

        :param DataHandler train_data_handler: Data handler for the train data
        """

        assert isinstance(train_data_handler, DataHandler), 'The training data should be an instance of DataHandler'

        self.__w = np.copy(self.__initial_w)
        self.__b = self.__initial_b

        for t in range(1, self.__num_iterations + 1):
            batch = train_data_handler.sample(self.__batch_size)
            learning_rate = 1 / t

            for i in range(self.__batch_size):
                w, b = self.__get_gradient_value(self.__w, self.__b, batch)

                self.__w = self.__w - learning_rate * w
                self.__b = self.__b - learning_rate * b

            if t % 1000 == 0:
                print('Iteration %d' % t)

    def __get_gradient_value(self, w, b, batch):
        sum_w = 0
        sum_b = 0

        for instance in batch:
            xi = instance.get_data()
            yi = instance.get_label()

            if (yi*(np.dot(w, xi) + b)) < 1:
                sum_w += -yi * xi
                sum_b += -yi

        return w + self.__psy_constant * sum_w, self.__psy_constant * sum_b

    def classify(self, test_data_handler):
        """

        :param DataHandler test_data_handler: Data handler for the test data
        :return: The classified test_data_handler
        :rtype: DataHandler
        """

        assert isinstance(test_data_handler, DataHandler), 'The test data should be an instance of DataHandler'

        test_instances = test_data_handler.get_instances(duplicate=True)

        for i, test_instance in enumerate(test_instances):
            xi = test_instance.get_data()
            distance = self.distance(xi)

            label = np.sign(distance)

            test_instances[i].set_label(label if label != 0 else 1)

        return test_instances

    def get_hyperplane_normal(self):
        return self.__w

    def get_hyperplane_independent_term(self):
        return self.__b
