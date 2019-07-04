#! /usr/bin/python

import numpy as np
from ml.data.handler import DataHandler, PandasDataHandler
from ml.supervised.regression.common import Regressor


class RidgeRegressor(Regressor):

    def __init__(self, C):
        self.__C = C
        self.__w = None
        self.__Ainv = None

    def train(self, train_data_handler):
        assert isinstance(train_data_handler, DataHandler) or isinstance(train_data_handler, PandasDataHandler), 'Must be an instance of DataHandler'

        X, y = train_data_handler.get_vectorized()

        self.__Ainv = np.linalg.inv(np.dot(X, X.T) + (1/(2*self.__C)) * np.identity(X.shape[0]))

        self.__w = np.dot(self.__Ainv, (np.dot(X, y.T)))

        se_sum = 0

        for i in range(len(X.T)):
            se_sum += self.__se(X.T[i], y.T[i])

        return float(np.sqrt(se_sum/len(X.T)))

    def predict(self, test_data_handler):
        assert isinstance(test_data_handler, DataHandler) or isinstance(test_data_handler, PandasDataHandler), 'Must be an instance of DataHandler'

        X, y = test_data_handler.get_vectorized()

        return np.dot(X.T, self.__w).T

    def __se(self, xi, yi):
        return np.square((np.dot(xi.T, self.__w) - yi) / (1 - np.dot(np.dot(xi.T, self.__Ainv), xi)))
