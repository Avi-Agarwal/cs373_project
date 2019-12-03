from pandas import DataFrame
import os
import collections
import numpy

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

# Change this if you want to change the training and testing files
training_file_name = 'enron1'
testing_file_name = 'enron3'

# Read in training data
spamFiles = [training_file_name + '/spam/' + f for f in os.listdir(training_file_name + '/spam')]
hamFiles = [training_file_name + '/ham/' + f for f in os.listdir(training_file_name + '/ham')]
training_files = hamFiles + spamFiles

# Read in test data
spamFiles_test = [testing_file_name + '/spam/' + f for f in os.listdir(testing_file_name + '/spam')]
hamFiles_test = [testing_file_name + '/ham/' + f for f in os.listdir(testing_file_name + '/ham')]
testing_files = hamFiles_test + spamFiles_test

# Models (initially set to 0)
nb_model = 0
svm_model = 0

# Features for testing set (initially set to 0)
test_features_matrix = 0


# Go through email subjects in files and create a word dictionary
def make_word_dictionary(files):
    # spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    # hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    # all_files = spamFiles + hamFiles
    emails = files
    # emails = spamFiles
    all_words = []
    for mail in emails:
        with open(mail, 'r', encoding="utf8", errors='ignore') as m:
            for i, line in enumerate(m):
                # For Checking only subject use if below
                # if i == 0:
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


# Gets the features in the form of a feature vector matrix whose rows represent the files
# and columns represent the 2000 common words found from the dictionary creation
def get_features(files, dictionary):
    # spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    # hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    # all_files = spamFiles + hamFiles
    full_files = files
    f_matrix = numpy.zeros((len(full_files), len(dictionary)))
    dictionary_index = 0

    for file in full_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as freader:
            for i, l in enumerate(freader):
                # if i == 0:  # Using just email Subject
                words = l.split()
                for w in words:
                    for index, d_words in enumerate(dictionary):
                        if d_words[0] == w:
                            f_matrix[dictionary_index, index] = words.count(w)
            dictionary_index = dictionary_index + 1

    return f_matrix


# Trains the classifiers using Naive Bayes and Support Vector Machine. Returns two models to test
def train_classifiers(feature_matrix):
    t_files_len = len(training_files)
    training_labels = numpy.zeros(t_files_len)
    training_labels[len(hamFiles):len(training_files)] = 1  # label all spam files as spam

    naive_bayes_model = MultinomialNB()
    vector_machine_model = LinearSVC()

    naive_bayes_model.fit(feature_matrix, training_labels)
    vector_machine_model.fit(feature_matrix, training_labels)

    # Store models globally
    global nb_model
    nb_model = naive_bayes_model
    global svm_model
    svm_model = vector_machine_model

    return naive_bayes_model, vector_machine_model


# Tests the models created on the training data set.
def test_classifiers():

    test_files_len = len(testing_files)
    test_labels = numpy.zeros(test_files_len)
    test_labels[len(hamFiles_test): len(testing_files)] = 1  # Labels testing files accordingly

    global nb_model
    global svm_model
    global test_features_matrix

    # if test_features_matrix == 0:
    #     print('Test feature matrix needs to be created, please call build_models()')
    #     return

    if nb_model != 0:
        result_nb = nb_model.predict(test_features_matrix)
    else:
        print('Models still need to be trained, please call build_models()')
        return

    if svm_model != 0:
        result_svm = svm_model.predict(test_features_matrix)
    else:
        print('Models still need to be trained, please call build_models()')
        return

    print(confusion_matrix(test_labels, result_nb))
    print(confusion_matrix(test_labels, result_svm))

    return result_nb, result_svm


# Builds the models using the training set and makes features matrix of testing data
def build_models():
    print('\nBuilding Models will take a couple minutes depending on training set size\n')
    dictionary = make_word_dictionary(training_files)
    dictionary = dictionary_preprocessing(dictionary)
    feature_matrix = get_features(training_files, dictionary)
    train_classifiers(feature_matrix)
    print('\nModels Finished Building! Ready for testing!')


# Builds the feature matrix for the training set
def build_test_feature_matrix():
    print('\nNow building feature matrix for training set')
    global test_features_matrix
    test_dict = make_word_dictionary(testing_files)
    test_dict = dictionary_preprocessing(test_dict)
    test_features_matrix = get_features(testing_files, test_dict)
    print('\nTesting Feature Matrix built!')


if __name__ == '__main__':

    # Call this to build classifiers (will take some time)
    build_models()
    # Call this to build test feature matrix (will take time)
    build_test_feature_matrix()

    # Call to test classifiers
    test_classifiers()
    print('\nComplete!')





