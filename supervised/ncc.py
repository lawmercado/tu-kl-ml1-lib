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

        classes = train_data_handler.get_class_values()
        class_attr = train_data_handler.get_class_attr()
        data_frame = train_data_handler.get_data_frame()

        ns = []

        for y in classes:
            ns.append(data_frame.loc[data_frame[class_attr] == y])

        for idx_y, y in enumerate(classes):
            self.__centroids[y] = np.array(ns[idx_y].loc[:, ns[idx_y].columns != class_attr].sum())/len(ns[idx_y])

    def classify(self, test_data_handler):
        """

        :param DataHandler test_data_handler: Data handler for the test data
        :return: The classified test_data_handler
        :rtype: DataHandler
        """

        assert isinstance(test_data_handler, DataHandler), 'The test data should be an instance of DataHandler'

        assert self.__centroids is not None, 'You need train the algorithm first!'

        test_data_frame = test_data_handler.get_data_frame()
        class_attr = test_data_handler.get_class_attr()

        for idx_test_instance, test_instance in test_data_frame.iterrows():
            closer_class = 0
            closer_distance = sys.maxsize

            for y in self.__centroids:
                pa = self.__centroids[y]
                pb = np.array(test_instance.loc[test_data_frame.columns != class_attr])

                distance = np.linalg.norm(pa - pb)

                if distance < closer_distance:
                    closer_distance = distance
                    closer_class = y

            test_data_frame.loc[idx_test_instance, 'Sex'] = closer_class

        return DataHandler(test_data_frame, test_data_handler.get_class_attr())

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
