#! /usr/bin/python

import numpy as np
from ml.supervised.common import Classifier
from ml.data.handler import DataHandler
import sys


class KNN(Classifier):

    __k = None
    __train_data_handler = None

    def __init__(self, k):
        self.__k = k

    def train(self, train_data_handler):
        """

        :param DataHandler train_data_handler: Data handler for the train data
        """

        assert isinstance(train_data_handler, DataHandler), 'The training data should be an instance of DataHandler'

        self.__train_data_handler = train_data_handler

    def classify(self, test_data_handler):
        """

        :param DataHandler test_data_handler: Data handler for the test data
        :return: The classified test_data_handler
        :rtype: DataHandler
        """

        assert isinstance(test_data_handler, DataHandler), 'The test data should be an instance of DataHandler'

        assert self.__train_data_handler is not None, 'No training data. Please train the algorithm first!'

        test_data_frame = test_data_handler.get_data_frame()
        train_data_frame = self.__train_data_handler.get_data_frame()
        class_attr = self.__train_data_handler.get_class_attr()

        for idx_test_instance, test_instance in test_data_frame.iterrows():
            distances = [sys.maxsize for _ in range(0, self.__k)]
            distances_instances = [None for _ in range(0, self.__k)]

            for idx_train_instance, train_instance in train_data_frame.iterrows():
                pa = np.array(train_instance.loc[train_data_frame.columns != class_attr])
                pb = np.array(test_instance.loc[test_data_frame.columns != class_attr])

                instance_distance = np.linalg.norm(pa-pb)

                for idx_distance, distance in enumerate(distances):
                    if distance > instance_distance:
                        distances.insert(idx_distance, instance_distance)
                        distances_instances.insert(idx_distance, idx_train_instance)
                        distances.pop()
                        distances_instances.pop()

                        break

            knn_classes = [train_data_frame.loc[distances_instances[i], 'Sex'] for i in range(len(distances_instances))]

            counters = {key: knn_classes.count(key) for key in knn_classes}

            winner_counter = 0
            winner = 0

            for key in counters:
                if counters[key] > winner_counter:
                    winner_counter = counters[key]
                    winner = key

            test_data_frame.loc[idx_test_instance, 'Sex'] = winner

        return DataHandler(test_data_frame, test_data_handler.get_class_attr())
