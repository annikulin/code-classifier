__author__ = 'nik'

import os
import bayes_classifier
import sys


def get_train_files(max_number_per_lang, resource_folder):
    train_files = []
    subdirs = [x[0] for x in os.walk(resource_folder)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if len(files) > max_number_per_lang:
            files = files[: max_number_per_lang]
        if len(files) > 0:
            for file in files:
                train_files.append(subdir + "/" + file)
    return train_files


def get_language_name(file_ending):
    options = {'c': 'C', 'cs': 'C#', 'cpp': 'C++', 'java': 'Java', 'js': 'JavaScript', 'pm': 'Perl', 'php': 'PHP',
               'py': 'Python', 'rb': 'Ruby', 'scala': 'Scala'}
    return options[file_ending]


def train_classifier(bayes_classifier, max_number_per_lang=10, resource_folder="naive_bayes/resources"):
    for train_file in get_train_files(max_number_per_lang, resource_folder):
        if "DS_Store" not in train_file:
            bayes_classifier.train(train_file, get_language_name(train_file.split('.')[-1]))