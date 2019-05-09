#! /usr/bin/python

import random
import pandas as pd
import math


class DataHandler:

    __data_frame = None
    __class_attr = None

    def __init__(self, data_frame, class_attr):
        """
        Handler for Pandas DataFrame manipulation

        """

        assert isinstance(data_frame, pd.DataFrame), 'Data frame must be an instance of Pandas DataFrame'

        copy = data_frame.copy(deep=True)
        copy.reset_index(drop=True)

        self.__data_frame = copy
        self.__class_attr = class_attr

    def get_data_frame(self):
        """
        Gets a copy of the DataFrame

        :return: A DataFrame identical to the used in the instantiation
        :rtype: DataFrame

        """

        return self.__data_frame.copy(deep=True)

    def get_class_attr(self):
        return self.__class_attr

    def get_class_values(self):
        """
        Gets the different class values

        :return: All the class values for this dataset
        :rtype: list

        """

        return list(pd.unique(self.__data_frame[self.__class_attr]))

    def set_default_classification(self, value=0):
        self.__data_frame[self.__class_attr] = value

    def stratify(self, num_folds=10):
        """
        Divide the data into num_folds, maintaining the class proportion

        :param integer num_folds: Number of folds to create
        :return: The folds
        :rtype: list of DataFrames
        """

        classes = self.get_class_values()

        options = []

        for y in classes:
            options.append(self.__data_frame.loc[self.__data_frame[self.__class_attr] == y])

        raw_folds = [[] for _ in range(0, num_folds)]

        instances_per_fold = math.ceil(len(self.__data_frame) / num_folds)

        for idx_y, y in enumerate(classes):
            y_proportion = len(options[idx_y]) / len(self.__data_frame)

            counter = math.ceil(y_proportion * instances_per_fold)

            while counter > 0:
                idx_fold = 0

                while idx_fold < num_folds and len(options[idx_y]) > 0:
                    idx_instance = random.randint(0, len(options[idx_y]) - 1)

                    instance_data_frame = options[idx_y].iloc[idx_instance].copy()

                    options[idx_y] = options[idx_y].drop([instance_data_frame.name])

                    raw_folds[idx_fold].append(instance_data_frame)

                    idx_fold = idx_fold + 1

                counter -= 1

        return [pd.DataFrame(data=raw_folds[idx_fold]) for idx_fold in range(0, num_folds)]
