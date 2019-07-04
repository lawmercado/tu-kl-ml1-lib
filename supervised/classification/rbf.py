#! /usr/bin/python

import numpy as np
from ml.supervised.classification.common import Classifier
from ml.data.handler import DataHandler
from sklearn.svm import SVC


class RBF(Classifier):

    def __init__(self, C, gamma):
        self.__svc = SVC(C=C, gamma=gamma)

    def train(self, train_data_handler):
        """

        :param DataHandler train_data_handler: Data handler for the train data
        """

        assert isinstance(train_data_handler, DataHandler), 'The training data should be an instance of DataHandler'

        X, y = train_data_handler.get_vectorized()
        X = np.transpose(X)
        y = np.transpose(y).ravel()

        self.__svc.fit(X, y)

    def classify(self, test_data_handler):
        """

        :param PandasDataHandler test_data_handler: Data handler for the test data
        :return: The classified test_data_handler
        :rtype: DataHandler
        """

        assert isinstance(test_data_handler, DataHandler), 'The test data should be an instance of DataHandler'

        test_instances = test_data_handler.get_instances(duplicate=True)

        X, y = test_data_handler.get_vectorized()
        X = np.transpose(X)

        pred = self.__svc.predict(X)

        for i in range(len(test_instances)):
            test_instances[i].set_label(pred[i])

        return test_instances
