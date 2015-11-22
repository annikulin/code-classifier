from __future__ import division
import collections
import math
import nltk


class BayesTextClassifier:
    def __init__(self):
        self.labels_word_occurrences = collections.defaultdict(
            lambda: 0)  # contains tuples of the form (label, word_name, word_occurrences)
        self.labels_word_document_occurrences = collections.defaultdict(lambda: 0)
        self.labels_word_number = collections.defaultdict(lambda: 0)  # number of all words from each label
        self.labels_document_number = collections.defaultdict(lambda: 0)
        self.vocabulary = set()
        self.vocabulary_word_number = collections.defaultdict(lambda: 0)
        self.feature_words = set()

    def calculate_label_probability(self, words, label, model):
        label_prob = 0
        if model == 'multinomial':
            for word in words:
                if len(self.feature_words) != 0 and word not in self.feature_words:
                    continue
                prob = math.log((self.labels_word_occurrences[label, word] + 1) /
                                (self.labels_word_number[label] + len(self.vocabulary)))
                label_prob += prob
        if model == 'bernoulli':
            for voc_word in self.vocabulary:
                prob = (self.labels_word_document_occurrences[label, voc_word] + 1) / (
                    self.labels_document_number[label] + 2)
                if voc_word in words:
                    label_prob += math.log(prob)
                else:
                    label_prob += math.log(1 - prob)
        return label_prob

    def test(self, code_listing, model='multinomial'):
        tokens = nltk.word_tokenize(code_listing)
        probability_per_label = {}
        for label in self.labels_document_number.keys():
            label_prob = self.calculate_label_probability(tokens, label, model)
            probability_per_label[label] = math.log(
                self.labels_document_number[label] / sum(self.labels_document_number.values())) + label_prob
        return max(probability_per_label, key=lambda class_label: probability_per_label[class_label])

    def train(self, training_file, label):
        with open(training_file, 'r') as source:
            self.labels_document_number[label] += 1
            code_listing = ' '.join(source.readlines())
            tokens = nltk.word_tokenize(code_listing)
            self.labels_word_number[label] += len(tokens)
            for token in tokens:
                self.vocabulary.add(token)

            tokens_num = collections.Counter(tokens)
            for word in tokens_num.keys():
                self.vocabulary_word_number[word] += 1
                self.labels_word_document_occurrences[label, word] += 1
                self.labels_word_occurrences[label, word] += tokens_num[word]

    def get_trained_set_statistics(self):
        statistics = {"Label count": self.labels_document_number, "Labels": self.labels_word_number.keys(),
                      "Label vocabulary size": self.labels_word_number, "Vocabulary size": len(self.vocabulary)}
        return statistics

    def make_feature_selection_mutual_information(self, feature_count):
        voc = set()
        for label in self.labels_word_number.keys():
            mutual_information = {}
            for label1, word in self.labels_word_document_occurrences:
                if label == label1:
                    n11 = float(self.labels_word_document_occurrences[label, word])
                    n01 = float(self.labels_document_number[label]) - n11
                    n10 = float(self.vocabulary_word_number[word]) - n11
                    n00 = float(sum(self.labels_document_number.values())) - n10
                    n = float(sum(self.labels_document_number.values()))

                    mutual_value = (n11 / n) * math.log((n * n11 + 1) / ((n11 + n10) * (n01 + n11))) + \
                                   (n01 / n) * math.log((n * n01 + 1) / ((n00 + n01) * (n01 + n11))) + \
                                   (n10 / n) * math.log((n * n10 + 1) / ((n11 + n10) * (n10 + n00))) + \
                                   (n00 / n) * math.log((n * n00 + 1) / ((n01 + n00) * (n10 + n00)))
                    mutual_information[word] = mutual_value
            best = sorted(mutual_information.items(), key=lambda x: -x[1])
            best1 = best[: feature_count]
            best2 = best[feature_count:]
            for word, val in best1:
                self.feature_words.add(word)
            for word, val in best2:
                self.labels_word_number[label] -= self.labels_word_occurrences[label, word]
        self.vocabulary = self.feature_words

    def make_feature_selection_chi_square(self, feature_count):
        voc = set()
        for label in self.labels_word_number.keys():
            chi_values = {}
            for label1, word in self.labels_word_document_occurrences:
                if label == label1:
                    n11 = float(self.labels_word_document_occurrences[label, word])
                    n01 = float(self.labels_document_number[label]) - n11
                    n10 = float(self.vocabulary_word_number[word]) - n11
                    n00 = float(sum(self.labels_document_number.values())) - n10
                    n = float(sum(self.labels_document_number.values()))

                    shi_value = (n * (n11*n00 - n10 * n01) * (n11*n00 - n10 * n01)) \
                                / ((n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00));
                    chi_values[word] = shi_value
            best = sorted(chi_values.items(), key=lambda x: -x[1])
            best1 = best[: feature_count]
            best2 = best[feature_count:]
            for word, val in best1:
                self.feature_words.add(word)
            for word, val in best2:
                self.labels_word_number[label] -= self.labels_word_occurrences[label, word]
        self.vocabulary = self.feature_words






