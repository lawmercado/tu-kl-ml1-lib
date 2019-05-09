#! /usr/bin/python

from abc import ABC, abstractmethod
from ml.data.handler import DataHandler
import pandas as pd


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
    assert isinstance(data_handler, DataHandler), 'The dataset must be an instance of DataHandler'

    class_attr = data_handler.get_class_attr()
    folds = data_handler.stratify(num_folds)

    measurements = {'acc': [], 'f-measure': []}

    for idx_fold, fold in enumerate(folds):
        test_data_handler = DataHandler(folds[idx_fold], class_attr)

        train_folds = [fold for i, fold in enumerate(folds) if i != idx_fold]
        train_data_handler = DataHandler(pd.concat(train_folds), class_attr)

        classifier.train(train_data_handler)

        test_data_handler.set_default_classification(0)
        classified_data_handler = classifier.classify(test_data_handler)
        classified_data_frame = classified_data_handler.get_data_frame()

        ref_data_frame = folds[idx_fold].copy()

        measure = analyse_results(ref_data_frame, classified_data_frame, class_attr)

        measurements['acc'].append(measure['acc'])
        measurements['f-measure'].append(measure['f-measure'])

    return measurements


def analyse_results(ref_data_frame, classified_data_frame, class_attr):
    """
    Analyse the classification data

    :param DataFrame ref_data_frame: The reference data frame
    :param DataFrame classified_data_frame: The classified data frame
    :param string class_attr: The class attribute
    :return: A dict with the main statistics ('acc' and 'f-measure') for this data frames
    :rtype: dict
    """

    correct_classifications = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for idx_ref, ref_instance in ref_data_frame.iterrows():
        if ref_instance[class_attr] == classified_data_frame.loc[idx_ref, class_attr]:
            correct_classifications = correct_classifications + 1

        if ref_instance[class_attr] == 1 and classified_data_frame.loc[idx_ref, class_attr] == 1:
            true_positive = true_positive + 1

        elif ref_instance[class_attr] == -1 and classified_data_frame.loc[idx_ref, class_attr] == 1:
            false_positive = false_positive + 1

        elif ref_instance[class_attr] == 1 and classified_data_frame.loc[idx_ref, class_attr] == -1:
            false_negative = false_negative + 1

        else:
            true_negative = true_negative + 1

    # Generate the statistics
    acc = correct_classifications / len(classified_data_frame)

    rev = true_positive / (true_positive + false_negative)

    prec = true_positive / (true_positive + false_positive)

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
