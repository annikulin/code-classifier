from bayes_classifier import BayesTextClassifier
from classifier_trainer import train_classifier
import os
import random


def multinomial_bayes_test(train_files_number=10, lines_per_file=10, feature_selection_method='mutual', feature_selection_count=-1):
    text_classifier = BayesTextClassifier()
    train_classifier(text_classifier, train_files_number, resource_folder="resources")
    if feature_selection_count != -1:
        if feature_selection_method == 'mutual':
            text_classifier.make_feature_selection_mutual_information(feature_selection_count)
        elif feature_selection_method == 'square':
            text_classifier.make_feature_selection_chi_square(feature_selection_count)
    case_count, correct_case_count = 0, 0
    for expected_language, listing in get_test_set(10, lines_per_file):
        actual_language = text_classifier.test(listing)
        case_count += 1
        if expected_language == actual_language:
            correct_case_count += 1
    print 'Multinomial Bayes Test. Model was trained on %d files per language. %d lines per test file. ' \
          '%d out of %d are correctly classifier. Productivity: %f' \
          % (train_files_number, lines_per_file, correct_case_count, case_count, float(correct_case_count / case_count))


def bernoulli_bayes_test(train_files_number=10, lines_per_file=10):
    text_classifier = BayesTextClassifier()
    train_classifier(text_classifier, train_files_number, resource_folder="resources")
    case_count, correct_case_count = 0, 0
    for expected_language, listing in get_test_set(10, lines_per_file):
        actual_language = text_classifier.test(listing, model='bernoulli')
        case_count += 1
        if expected_language == actual_language:
            correct_case_count += 1
    print 'Bernoulli Bayes Test. Model was trained on %d files per language. %d lines per test file. ' \
          '%d out of %d are correctly classifier. Productivity: %f' \
          % (train_files_number, lines_per_file, correct_case_count, case_count, float(correct_case_count / case_count))


def get_testfile_names(max_number_per_lang, resource_folder):
    train_files = []
    subdirs = [x[0] for x in os.walk(resource_folder)]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        random.shuffle(files)
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


def get_test_set(max_number_per_lang=10, lines_per_file=10, resource_folder="resources"):
    test_set = []
    for test_file_name in get_testfile_names(max_number_per_lang, resource_folder):
        with open(test_file_name) as test_file:
            file_lines = test_file.readlines()
        line_number = len(file_lines)
        if line_number > lines_per_file:
            start_line = random.randrange(line_number - lines_per_file)
            file_lines = file_lines[start_line: start_line + lines_per_file]
        if "DS_Store" not in test_file_name:
            language = get_language_name(test_file_name.split('.')[-1])
            test_set.append((language, ' '.join(file_lines)))
    return test_set


if __name__ == "__main__":
    multinomial_bayes_test(train_files_number=20, lines_per_file=20)

    print '________________________________________________________'

    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=5)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=10)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=25)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=50)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=100)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=250)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='mutual', feature_selection_count=500)

    print '________________________________________________________'

    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=5)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=10)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=25)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=50)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=100)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=250)
    multinomial_bayes_test(train_files_number=20, lines_per_file=20, feature_selection_method='square', feature_selection_count=500)


        # multinomial_bayes_test(lines_per_file=1, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=2, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=5, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=10, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=15, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=25, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=50, feature_selection_count=150)

    # print '_________________________________________________________'
    #
    # bernoulli_bayes_test(train_files_number=3, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=5, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=10, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=15, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=20, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=25, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=30, lines_per_file=20)
    # bernoulli_bayes_test(train_files_number=50, lines_per_file=20)


    #
    # print '_________________________________________________________________'
    #
    # multinomial_bayes_test(lines_per_file=1, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=2, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=5, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=10, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=15, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=25, feature_selection_count=150)
    # multinomial_bayes_test(lines_per_file=50, feature_selection_count=150)
    #
    # multinomial_bayes_test(lines_per_file=1, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=2, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=5, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=10, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=15, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=25, feature_selection_count=250)
    # multinomial_bayes_test(lines_per_file=50, feature_selection_count=250)
    # #
    # # print '_________________________________________________________'
    #
    # multinomial_bayes_test(train_files_number=50, lines_per_file=1)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=2)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=5)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=10)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=15)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=25)
    # multinomial_bayes_test(train_files_number=50, lines_per_file=50)







