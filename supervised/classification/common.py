#! /usr/bin/python

from abc import ABC, abstractmethod
from ml.data.handler import DataHandler, PandasDataHandler
import copy
import numpy as np


class Classifier(ABC):
    """
    Base class for classifiers

    """

    @abstractmethod
    def train(self, train_data_handler):
        raise NotImplementedError

    @abstractmethod
    def classify(self, test_data_handler):
        raise NotImplementedError


class LinearClassifier(Classifier):
    """
    Class for linear classifiers

    """

    @abstractmethod
    def get_hyperplane_normal(self):
        raise NotImplementedError

    @abstractmethod
    def get_hyperplane_independent_term(self):
        raise NotImplementedError

    def distance(self, x):
        w = self.get_hyperplane_normal()
        b = self.get_hyperplane_independent_term()

        return (np.dot(w, x) + b) / np.linalg.norm(w)


def crossvalidation(classifier, data_handler, num_folds):
    """
    Apply the crossvalidation for the given algorithm

    :param Classifier classifier: A classification algorithm
    :param DataHandler data_handler: The data to use with the classifier
    :param integer num_folds: The number of folds to stratify the data
    :return: A dict with the measures for 'acc' and 'f-measure'
    :rtype: dict
    """

    assert isinstance(classifier, Classifier), 'The classifier must implement Classifier abc'
    assert isinstance(data_handler, DataHandler) or isinstance(data_handler, PandasDataHandler), 'The dataset must be an instance of DataHandler'

    folds = data_handler.stratify(num_folds)

    measurements = {'acc': [], 'f-measure': []}

    for idx_fold, fold in enumerate(folds):
        test_instances = copy.deepcopy(folds[idx_fold])
        train_instances = []
        train_folds = [copy.deepcopy(fold) for i, fold in enumerate(folds) if i != idx_fold]

        for train_fold in train_folds:
            train_instances += train_fold

        train_data_handler = DataHandler([instance.get_raw() for instance in train_instances], data_handler.get_columns(), data_handler.get_label_column())
        classifier.train(train_data_handler)

        test_data_handler = DataHandler([instance.get_raw() for instance in test_instances], data_handler.get_columns(), data_handler.get_label_column())
        classified_instances = classifier.classify(test_data_handler)

        measure = analyse_results(copy.deepcopy(folds[idx_fold]), classified_instances)

        measurements['acc'].append(measure['acc'])
        measurements['f-measure'].append(measure['f-measure'])

    return measurements


def analyse_results(test_instances, classified_instances):
    """
    Analyse the classification data

    :return: A dict with the main statistics ('acc' and 'f-measure') for this data frames
    :rtype: dict
    """

    correct_classifications = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(len(test_instances)):
        if test_instances[i].get_label() == classified_instances[i].get_label():
            correct_classifications = correct_classifications + 1

        if test_instances[i].get_label() == 1 and classified_instances[i].get_label() == 1:
            true_positive = true_positive + 1

        elif test_instances[i].get_label() == -1 and classified_instances[i].get_label() == 1:
            false_positive = false_positive + 1

        elif test_instances[i].get_label() == 1 and classified_instances[i].get_label() == -1:
            false_negative = false_negative + 1

        else:
            true_negative = true_negative + 1

    # Generate the statistics
    acc = correct_classifications / len(classified_instances)

    try:
        rev = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        rev = 0

    try:
        prec = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        prec = 0

    try:
        f_measure = 2 * (prec * rev) / (prec + rev)
    except ZeroDivisionError:
        f_measure = 0

    return {'acc': acc, 'f-measure': f_measure}


def repeated_crossvalidation(classifier, data_handler, num_folds, num_repetitions):
    """
    Apply the crossvalidation 'num_repetitions' times, and gathers its measurements

    :param Classifier classifier: A classification algorithm
    :param DataHandler data_handler: The data to use with the classifier
    :param integer num_folds: The number of folds to stratify the data
    :param integer num_repetitions: The number of times to apply crossvalidation
    :return: A dict with the measures for 'acc' and 'f-measure'
    :rtype: dict
    """

    measurements = {"acc": [], "f-measure": []}

    for i in range(0, num_repetitions):
        fold_measures = crossvalidation(classifier, data_handler, num_folds)

        measurements['acc'] += fold_measures['acc']
        measurements['f-measure'] += fold_measures['f-measure']

    return measurements


def generate_statistics(measurements):
    """
    With a set of measures, calculates the average and de standard

    :param dict measurements: The name of the measures and a list of measurement
    :return: A tuple containing the average and the standard deviation associated with the measure
    :rtype: dict { measure: (<average>, <standard deviation>), ... }
    """

    statistics = {}

    for id_measure in measurements:
        acc = 0
        for measure in measurements[id_measure]:
            acc += measure

        avg = acc / len(measurements[id_measure])

        f_acc = 0
        for measure in measurements[id_measure]:
            f_acc += (measure - avg) ** 2

        std_deviation = (f_acc / (len(measurements[id_measure]) - 1)) ** 0.5

        statistics[id_measure] = (avg, std_deviation)

    return statistics
