#! /usr/bin/python

import random
import pandas as pd
import math
from liblinearutil import problem


class DataHandler:

    __data_frame = None
    __class_attr = None

    def __init__(self, data_frame, class_attr, reset_index=True):
        """
        Handler for Pandas DataFrame manipulation

        """

        assert isinstance(data_frame, pd.DataFrame), 'Data frame must be an instance of Pandas DataFrame'

        copy = data_frame.copy(deep=True)

        if reset_index:
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

    def normalize_min_max(self, a, b):
        copy = self.get_data_frame()
        target_columns = copy.columns != self.get_class_attr()
        targets = copy.iloc[:, target_columns]

        copy.iloc[:, target_columns] = (((b - a) * (targets - targets.min())) / (targets.max() - targets.min())) + a
        self.__data_frame = copy

    def get_as_liblinear_problem(self):
        data_frame = self.get_data_frame()

        y = data_frame[self.__class_attr].values.tolist()
        x = data_frame.iloc[:, data_frame.columns != self.__class_attr].values.tolist()

        return problem(y, x)

    def get_as_vw_data(self, ignore_label=False):
        data_frame = self.get_data_frame()
        num_features = len(list(data_frame.columns)) - 1  # Discarding the label column

        data = ''

        for idx_row, row in data_frame.iterrows():
            line = ''

            if not ignore_label:
                line = '%s ' % row[self.__class_attr]

            line += 'ex%s|f' % idx_row

            for idx_feature in range(num_features):
                if row[idx_feature] != 0:
                    line += ' %d:%f' % (idx_feature + 1, row[idx_feature])

            data += line + '\n'

        return data
