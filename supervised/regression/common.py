from abc import ABC, abstractmethod
import numpy as np


class Regressor(ABC):
    """
    Base class for regressors

    """

    @abstractmethod
    def train(self, train_data_handler):
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data_handler):
        raise NotImplementedError


def squared_error(y_correct, y_out):
    return np.square(y_out - y_correct)


def error(y_correct, y_out):
    return y_out - y_correct
