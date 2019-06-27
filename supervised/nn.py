#! /usr/bin/python

import numpy as np
from ml.data.handler import PandasDataHandler
from ml.supervised.common import Classifier
import logging
import copy

logger = logging.getLogger(__name__)


class NN(Classifier):

    def __init__(self, p=5, learning_factor=5, batch_size=50, num_iterations=100000, threshold=0.5):
        self.__num_layers = 3
        self.__layers = [2, 5, 5, 1]
        self.__params = {}
        self.__cache = {}

        self.__p = p
        self.__learning_factor = learning_factor
        self.__batch_size = batch_size
        self.__num_iterations = num_iterations

        self.__threshold = threshold

        self.__Yout = np.zeros((1, batch_size))

        self.__setup()

    def __setup(self):
        self.__params['W1'] = np.random.randn(self.__layers[1], self.__layers[0]) / np.sqrt(self.__layers[0])
        self.__params['b1'] = np.zeros((self.__layers[1], 1))
        self.__params['W2'] = np.random.randn(self.__layers[2], self.__layers[1]) / np.sqrt(self.__layers[1])
        self.__params['b2'] = np.zeros((self.__layers[2], 1))
        self.__params['W3'] = np.random.randn(self.__layers[3], self.__layers[2]) / np.sqrt(self.__layers[2])
        self.__params['b3'] = np.zeros((self.__layers[3], 1))

        return

    @staticmethod
    def __activation(U):
        return 1 / (1 + np.exp(-U))

    @staticmethod
    def __activation_derivative(U):
        s = 1 / (1 + np.exp(-U))
        dU = s * (1 - s)
        return dU

    def __propagate(self, X, Y):
        U1 = self.__params['W1'].dot(X) + self.__params['b1']
        V1 = NN.__activation(U1)
        self.__cache['U1'], self.__cache['V1'] = U1, V1

        U2 = self.__params['W2'].dot(V1) + self.__params['b2']
        V2 = NN.__activation(U2)
        self.__cache['U2'], self.__cache['V2'] = U2, V2

        U3 = self.__params['W3'].dot(V2) + self.__params['b3']
        V3 = NN.__activation(U3)
        self.__cache['U3'], self.__cache['V3'] = U3, V3

        self.__Yout = V3
        loss = self.__loss(Y, V3)
        return self.__Yout, loss

    def __loss(self, Y, Yout):
        loss = (1. / Y.shape[1]) * (-np.dot(Y, np.log(Yout).T) - np.dot(1 - Y, np.log(1 - Yout).T))
        return loss

    def __backpropagate(self, X, Y, lr):
        dJ_Yout = - (np.divide(Y, self.__Yout) - np.divide(1 - Y, 1 - self.__Yout))

        dJ_U3 = dJ_Yout * NN.__activation_derivative(self.__cache['U3'])
        dJ_V2 = np.dot(self.__params['W3'].T, dJ_U3)
        dJ_W3 = 1. / self.__cache['V2'].shape[1] * np.dot(dJ_U3, self.__cache['V2'].T)
        dJ_b3 = 1. / self.__cache['V2'].shape[1] * np.dot(dJ_U3, np.ones([dJ_U3.shape[1], 1]))

        dJ_U2 = dJ_V2 * NN.__activation_derivative(self.__cache['U2'])
        dJ_V1 = np.dot(self.__params['W2'].T, dJ_U2)
        dJ_W2 = 1. / self.__cache['V1'].shape[1] * np.dot(dJ_U2, self.__cache['V1'].T)
        dJ_b2 = 1. / self.__cache['V1'].shape[1] * np.dot(dJ_U2, np.ones([dJ_U2.shape[1], 1]))

        dJ_U1 = dJ_V1 * NN.__activation_derivative(self.__cache['U1'])
        dJ_V0 = np.dot(self.__params['W1'].T, dJ_U1)
        dJ_W1 = 1. / X.shape[1] * np.dot(dJ_U1, X.T)
        dJ_b1 = 1. / X.shape[1] * np.dot(dJ_U1, np.ones([dJ_U1.shape[1], 1]))

        self.__params['W1'] = self.__params['W1'] - lr * dJ_W1
        self.__params['b1'] = self.__params['b1'] - lr * dJ_b1
        self.__params['W2'] = self.__params['W2'] - lr * dJ_W2
        self.__params['b2'] = self.__params['b2'] - lr * dJ_b2
        self.__params['W3'] = self.__params['W3'] - lr * dJ_W3
        self.__params['b3'] = self.__params['b3'] - lr * dJ_b3

        return

    def __label(self, prediction):
        comp = np.zeros((1, prediction.shape[1]))

        for i in range(0, prediction.shape[1]):
            if prediction[0, i] > self.__threshold:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        return comp

    def classify(self, dh):
        """

        :param PandasDataHandler dh: Data handler for the train data
        """

        assert isinstance(dh, PandasDataHandler), 'The training data is not a PandasDataHandler instance'
        X, Y = dh.get_vectorized()

        pred, loss = self.__propagate(X, Y)

        comp = self.__label(pred)

        acc = np.sum((comp == Y) / X.shape[1])

        return comp, acc, loss

    def train(self, dh):
        """

        :param PandasDataHandler dh: Data handler for the train data
        """

        assert isinstance(dh, PandasDataHandler), 'The training data is not a PandasDataHandler instance'

        t = 1
        best_params = {}
        smallest_error = float('inf')
        greater_error_count = 0
        final_error = 0

        accs_train = []
        accs_val = []

        # Split the dataset into validation and training sets
        df = dh.get_data_frame().sample(frac=1)
        limit = round(len(df) * 0.95)
        df_train = df[0:limit]
        dh_train = PandasDataHandler(df_train, dh.get_class_attr(), reset_index=False)
        df_val = df[limit:len(df)]
        dh_val = PandasDataHandler(df_val, dh.get_class_attr(), reset_index=False)

        # Split the training set into batches
        batches = dh_train.get_batches(self.__batch_size)

        while greater_error_count < self.__p and t < self.__num_iterations:
            acc_mean_train = 0

            for batch in batches:
                X, Y = batch.get_vectorized()

                Yout, loss = self.__propagate(X, Y)

                self.__backpropagate(X, Y, self.__learning_factor / t)

                comp = self.__label(Yout)

                acc_mean_train += np.sum((comp == Y) / X.shape[1])

            if t % 250 == 0:
                acc_mean_train = acc_mean_train / len(batches)

                print('Iteration %d' % t)
                print('-- Accuracy in the training set is', acc_mean_train)
                print('-- Loss in the training set is %f' % loss)

                Yout_val, acc_val, loss = self.classify(dh_val)

                loss_val = float(loss)

                print('-- Accuracy in the validation is', acc_val)
                print('-- Loss in the validation set is', loss_val)

                if loss_val < smallest_error:
                    smallest_error = loss_val
                    best_params = copy.deepcopy(self.__params)

                    accs_train.append(acc_mean_train)
                    accs_val.append(acc_val)

                    final_error = loss_val

                    greater_error_count = 0
                else:
                    greater_error_count += 1

            t += 1

        self.__params = best_params

        return accs_train, accs_val, final_error
