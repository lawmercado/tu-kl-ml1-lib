#! /usr/bin/python

import random
import pandas as pd
import numpy as np
import math
import copy
import sys
from liblinearutil import problem


class Instance:

    __data = None
    __label = None

    def __init__(self, data, label):
        self.__data = np.array(data, copy=True)
        self.__label = label

    def set_data(self, data):
        self.__data = data

    def set_label(self, value):
        self.__label = value

    def get_data(self):
        return copy.deepcopy(self.__data)

    def get_label(self):
        return self.__label

    def get_raw(self):
        raw = list(list(self.__data) + [self.__label])
        return raw

    def __str__(self):
        return str(list(self.__data)) + ", " + str(self.__label)


class DataHandler:

    def __init__(self, raw_data, columns, label_column, idx_column=None, replace_label=None):
        self.__columns = columns
        self.__label_column = label_column
        self.__instances = []

        idx_label_column = columns.index(label_column)

        for i in range(0, len(raw_data)):
            row = [float(raw_data[i][j]) for j in range(len(raw_data[i])) if j != idx_column]
            label = row.pop(idx_label_column)
            data = row

            if replace_label:
                label = replace_label(label)

            self.__instances.append(Instance(data, label))

    def get_columns(self):
        return copy.deepcopy(self.__columns)

    def get_instances(self, duplicate=False):
        if duplicate:
            return copy.deepcopy(self.__instances)
        else:
            return self.__instances

    def get_label_column(self):
        return self.__label_column

    def get_label_values(self):
        """
        Gets the different label values

        :return: All the label values for this dataset
        :rtype: list

        """

        instances = self.get_instances()
        labels = []

        for instance in instances:
            label = instance.get_label()

            if label not in labels:
                labels.append(label)

        return labels

    def stratify(self, num_folds=10):
        """
        Divide the data into num_folds, maintaining the class proportion

        :param integer num_folds: Number of folds to create
        :return: The folds
        :rtype: list of list with Instance elements
        """

        classes = self.get_label_values()
        instances = self.get_instances()

        options = {}

        for y in classes:
            options[y] = [instance for instance in instances if instance.get_label() == y]

        folds = [[] for _ in range(0, num_folds)]

        instances_per_fold = math.ceil(len(instances) / num_folds)

        for y in classes:
            y_proportion = len(options[y]) / len(instances)

            counter = math.ceil(y_proportion * instances_per_fold)

            while counter > 0:
                idx_fold = 0

                while idx_fold < num_folds and len(options[y]) > 0:
                    idx_instance = random.randint(0, len(options[y]) - 1)
                    instance = options[y].pop(idx_instance)

                    folds[idx_fold].append(instance)

                    idx_fold = idx_fold + 1

                counter -= 1

        return folds

    def sample(self, num_samples, repose=True):
        return DataHandler.sample_from_instances(self.get_instances(), num_samples, repose)

    @staticmethod
    def sample_from_instances(instances, num_samples, repose=True):
        sampled_instances = []
        repose_instances = []

        for i in range(num_samples):
            idx_instance = random.randint(0, len(instances) - 1)
            instance = instances.pop(idx_instance)
            sampled_instances.append(instance)
            repose_instances.append(instance)

        if repose:
            instances += repose_instances

        return sampled_instances

    def __str__(self):
        text = str(self.__columns) + '\n'
        instances = self.get_instances()

        for i in range(len(instances)):
            text += str(instances[i]) + '\n'

        return text

    def normalize_min_max(self, a, b):
        instances = self.get_instances()
        min_value = [sys.maxsize for _ in range(len(self.__columns) - 1)]
        max_value = [0 for _ in range(len(self.__columns) - 1)]

        for i in range(len(self.__columns)):
            if self.__columns[i] != self.__label_column:
                for instance in instances:
                    value = instance.get_data()[i]

                    if value < min_value[i]:
                        min_value[i] = value

                    if value > max_value[i]:
                        max_value[i] = value

        for i in range(len(instances)):
            data = instances[i].get_data()
            normalized_data = [0 for _ in range(len(data))]

            for j in range(len(self.__columns) - 1):
                normalized_data[j] = (((b - a) * (data[j] - min_value[j])) / (max_value[j] - min_value[j])) + a

            instances[i].set_data(np.array(normalized_data))

        self.__instances = instances


class PandasDataHandler:

    __data_frame = None
    __class_attr = None

    def __init__(self, data_frame, class_attr, reset_index=True):
        """
        Handler for Pandas DataFrame manipulation

        """

        assert isinstance(data_frame, pd.DataFrame), 'Data frame must be an instance of Pandas DataFrame'

        if reset_index:
            data_frame.reset_index(drop=True)

        self.__data_frame = data_frame
        self.__class_attr = class_attr

    def get_data_frame(self, copy=True):
        """
        Gets a copy of the DataFrame

        :return: A DataFrame identical to the used in the instantiation
        :rtype: DataFrame

        """

        return self.__data_frame.copy(deep=True) if copy else self.__data_frame

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

    def get_vectorized(self):
        df = self.get_data_frame(copy=False)

        X = df.loc[:, df.columns != self.get_class_attr()].values.transpose()
        Y = df.loc[:, self.get_class_attr():].values.transpose()

        return X, Y

    def get_batches(self, batch_size):
        df = self.get_data_frame().sample(frac=1)
        num_batches = round(len(self.get_data_frame())/batch_size)
        batches = []

        for i in range(num_batches):
            batches.append(PandasDataHandler(df[i*batch_size:(i+1)*batch_size], self.get_class_attr(), reset_index=False))

        return batches
