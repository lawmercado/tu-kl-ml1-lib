#! /usr/bin/python

import numpy as np
from ml.supervised.common import LinearClassifier
from ml.data.handler import DataHandler
import sys


class NCC(LinearClassifier):

    __centroids = {}

    def train(self, train_data_handler):
        """

        :param DataHandler train_data_handler: Data handler for the train data
        """

        assert isinstance(train_data_handler, DataHandler), 'The training data should be an instance of DataHandler'

        classes = train_data_handler.get_label_values()

        ns = {}

        for y in classes:
            ns[y] = [instance for instance in train_data_handler.get_instances() if instance.get_label() == y]

        for y in ns:
            nt = ns[y][0].get_data()

            for i in range(1, len(ns[y])):
                nt += ns[y][i].get_data()

            self.__centroids[y] = nt/len(ns[y])

    def classify(self, test_data_handler):
        """

        :param DataHandler test_data_handler: Data handler for the test data
        :return: The classified test_data_handler
        :rtype: DataHandler
        """

        assert isinstance(test_data_handler, DataHandler), 'The test data should be an instance of DataHandler'

        assert self.__centroids is not None, 'You need train the algorithm first!'

        test_instances = test_data_handler.get_instances(duplicate=True)

        for idx_test_instance, test_instance in enumerate(test_instances):
            closer_class = 0
            closer_distance = sys.maxsize

            for y in self.__centroids:
                pa = self.__centroids[y]
                pb = test_instance.get_data()

                distance = np.linalg.norm(pa - pb)

                if distance < closer_distance:
                    closer_distance = distance
                    closer_class = y

            test_instances[idx_test_instance].set_label(closer_class)

        return test_instances

    def get_named_centroids(self):
        positive_centroid = None
        negative_centroid = None

        for y in self.__centroids:
            if y > 0:
                positive_centroid = self.__centroids[y]
            else:
                negative_centroid = self.__centroids[y]

        return positive_centroid, negative_centroid

    def get_hyperplane_normal(self):
        (positive_centroid, negative_centroid) = self.get_named_centroids()

        return 2*(positive_centroid - negative_centroid)

    def get_hyperplane_independent_term(self):
        (positive_centroid, negative_centroid) = self.get_named_centroids()

        return np.linalg.norm(negative_centroid)**2 - np.linalg.norm(positive_centroid)**2
