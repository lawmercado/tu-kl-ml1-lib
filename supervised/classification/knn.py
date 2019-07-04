#! /usr/bin/python

import numpy as np
from ml.supervised.classification.common import Classifier
import sys


class KNN(Classifier):

    __k = None
    __train_data_handler = None

    def __init__(self, k):
        self.__k = k

    def train(self, train_data_handler):
        self.__train_data_handler = train_data_handler

    def classify(self, test_data_handler):
        assert self.__train_data_handler is not None, 'No training data. Please train the algorithm first!'

        test_instances = test_data_handler.get_instances(duplicate=True)
        train_instances = self.__train_data_handler.get_instances()

        for idx_test_instance, test_instance in enumerate(test_instances):
            distances = [sys.maxsize for _ in range(0, self.__k)]
            distances_instances = [None for _ in range(0, self.__k)]

            for idx_train_instance, train_instance in enumerate(train_instances):
                pa = np.array(train_instance.get_data())
                pb = np.array(test_instance.get_data())

                instance_distance = np.linalg.norm(pa-pb)

                for idx_distance, distance in enumerate(distances):
                    if distance > instance_distance:
                        distances.insert(idx_distance, instance_distance)
                        distances_instances.insert(idx_distance, idx_train_instance)
                        distances.pop()
                        distances_instances.pop()

                        break

            knn_classes = [train_instances[distances_instances[i]].get_label() for i in range(len(distances_instances))]

            counters = {key: knn_classes.count(key) for key in knn_classes}

            winner_counter = 0
            winner = 0

            for key in counters:
                if counters[key] > winner_counter:
                    winner_counter = counters[key]
                    winner = key

            test_instances[idx_test_instance].set_label(winner)

        return test_instances
