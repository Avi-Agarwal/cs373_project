from pandas import DataFrame
import os
import collections
import numpy

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

# Read in data
spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
training_files = spamFiles + hamFiles


# def loadData():
#     data = DataFrame({'text': [], 'classifier': []})
#
#     spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
#     hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
#
#     spam = []
#     spam_names = []
#     for file in spamFiles:
#         with open(file, 'r', encoding="utf8", errors='ignore') as f:
#             text = f.read()
#             spam.append({'text': text, 'classifier': 'spam'})
#             spam_names.append(file)
#
#     data = data.append(DataFrame(spam, spam_names))
#
#     ham = []
#     ham_names = []
#     for file in hamFiles:
#         with open(file, 'r', encoding="utf8", errors='ignore') as f:
#             text = f.read()
#             ham.append({'text': text, 'classifier': 'not spam'})
#             ham_names.append(file)
#
#     data = data.append(DataFrame(ham, ham_names))
#
#
#     print(data.describe())
#     return data
#
# loadData()


# Go through emails and create a word dictionary
def make_word_dictionary():
    # spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    # hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    # all_files = spamFiles + hamFiles
    emails = training_files
    # emails = spamFiles
    all_words = []
    for mail in emails:
        with open(mail, 'r', encoding="utf8", errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 0: # Checking only subject
                    words = line.split()
                    all_words += words

    dictionary = collections.Counter(all_words)
    return dictionary


# Get rid of non alpha containing words and single digit characters
def dictionary_preprocessing(dictionary):
    check_list = dictionary.keys()
    for word in list(check_list):
        if not word.isalpha() or len(word) == 1:
            del dictionary[word]
    dictionary = dictionary.most_common(2000)  # get most common 2000 words after removing alpha numeric
    return dictionary


# Gets the features in the form of a feature vector matrix whose rows represent the files trained
# and columns the 2000 common words found from the dictionary creation
def get_features(dictionary):
    # spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    # hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    # all_files = spamFiles + hamFiles
    full_files = training_files
    f_matrix = numpy.zeros((len(full_files), len(dictionary)))
    dictionary_index = 0

    for file in full_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as freader:
            for i, l in enumerate(freader):
                if i == 0:  # Using just email Subject
                    words = l.split()
                    for w in words:
                        for index, d_words in enumerate(dictionary):
                            if d_words[0] == w:
                                f_matrix[dictionary_index, index] = words.count(w)
            dictionary_index = dictionary_index + 1

    return f_matrix


# Trains the classifiers using Naive Bayes and Support Vector Machine. Returns two models to test
def train_classifier(dictionary, feature_matrix):
    t_files_len = len(training_files)
    training_labels = numpy.zeros(t_files_len)
    training_labels[int(t_files_len/2):int(t_files_len-1)] = 1

    naive_bayes_model = MultinomialNB()
    vector_machine_model = LinearSVC()

    naive_bayes_model.fit(feature_matrix, training_labels)
    vector_machine_model.fit(feature_matrix, training_labels)

    return naive_bayes_model, vector_machine_model



def test_classifier():
    print()








if __name__ == '__main__':
    dict = make_word_dictionary()
    dict = dictionary_preprocessing(dict)
    fmat = get_features(dict)
    print(fmat)
    print(dict)





