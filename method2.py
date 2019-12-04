from pandas import DataFrame
import os
import collections
import numpy

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

# Change this if you want to change the training and testing files
training_file_name = 'enron4'
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

# List of words to use as features (initially set to 0)
word_list = 0

# Features for testing set (initially set to 0)
test_features_matrix = 0


# Go through email subjects in files and create a word list, these will be the features
def make_word_list(files):
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

    wlist = collections.Counter(all_words)
    print('Creating a word list with word frequencies...')
    return wlist


# Get rid of non alpha containing words and single digit characters
def word_list_preprocessing(w_list):
    check_list = w_list.keys()
    for word in list(check_list):
        if not word.isalpha() or len(word) == 1:
            del w_list[word]
    w_list_f = w_list.most_common(1000)  # get most common 1000 words after removing alpha numeric
    print('Reducing word list to only the 1000 most frequently used words...')
    return w_list_f


# Gets the feature matrix in the form of a feature vector matrix whose rows represent the files (samples
# and columns represent the 100 common words found from the word list (features)
def get_feature_matrix(files, w_list):
    # spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    # hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    # all_files = spamFiles + hamFiles
    full_files = files
    f_matrix = numpy.zeros((len(full_files), len(w_list)))
    sample_index = 0
    print('Creating matrix of Samples (frequency of word in email sample n) vs features m (1000 most '
          'common words)...\n')


    for file in full_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as freader:
            for i, l in enumerate(freader):
                # if i == 0:  # Using just email Subject
                words = l.split()
                for w in words:
                    for index, d_words in enumerate(w_list):
                        if d_words[0] == w:
                            f_matrix[sample_index, index] = f_matrix[sample_index, index] + 1
            sample_index = sample_index + 1
    return f_matrix


# Trains the classifiers using Naive Bayes and Support Vector Machine. Returns two models to test
def train_classifiers(feature_matrix):
    print('Training the Classifiers:')
    t_files_len = len(training_files)
    training_labels = numpy.zeros(t_files_len)
    training_labels[len(hamFiles):len(training_files)] = 1  # label all spam files as 1

    naive_bayes_model = MultinomialNB()
    vector_machine_model = LinearSVC()

    naive_bayes_model.fit(feature_matrix, training_labels)
    vector_machine_model.fit(feature_matrix, training_labels)

    # Store models globally
    global nb_model
    nb_model = naive_bayes_model

    global svm_model
    svm_model = vector_machine_model

    print('Classifiers Trained!')

    return naive_bayes_model, vector_machine_model


# Tests the models created on the training data set.
def test_classifiers():

    print('Testing Classifiers...')
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
        print('Models still need to be trained, please call build_models(), please make sure you are not running in '
              'pytest mode')
        return

    if svm_model != 0:
        result_svm = svm_model.predict(test_features_matrix)
    else:
        print('Models still need to be trained, please call build_models(), please make sure you are not running in '
              'pytest mode')
        return

    print('Model Test Outputs:\n')
    nb_matrix_results = confusion_matrix(test_labels, result_nb)
    svm_matrix_results = confusion_matrix(test_labels, result_svm)
    print(nb_matrix_results)
    print(svm_matrix_results)

    nb_ham_classification = int(nb_matrix_results[0, 0])
    nb_spam_classification = int(nb_matrix_results[1, 1])

    svm_ham_classification = int(svm_matrix_results[0, 0])
    svm_spam_classification = int(svm_matrix_results[1, 1])

    total_ham = int(nb_matrix_results[0, 0] + nb_matrix_results[0, 1])
    total_spam = int(nb_matrix_results[1, 0] + nb_matrix_results[1, 1])

    print('\nThe SVM trained Classifier Results:')
    print('Classified ' + str(svm_ham_classification) + '/' + str(total_ham) + ' Ham emails correctly from the testing set')
    print('Classified ' + str(svm_spam_classification) + '/' + str(
        total_spam) + ' Spam emails correctly from the testing set')

    print('\nThe Naive Baiyes trained Classifier Results:')
    print('Classified ' + str(nb_ham_classification) + '/' + str(
        total_ham) + ' Ham emails correctly from the testing set')
    print('Classified ' + str(nb_spam_classification) + '/' + str(
        total_spam) + ' Spam emails correctly from the testing set')

    return result_nb, result_svm


# Builds the models using the training set and makes features matrix of testing data
def build_models():
    print('\nBuilding The Models. This will take a couple minutes depending on the training set size')
    print('Training data set used is: ' + training_file_name + '\n')
    w_list = make_word_list(training_files)
    w_list = word_list_preprocessing(w_list)
    global word_list
    word_list = w_list
    feature_matrix = get_feature_matrix(training_files, w_list)
    train_classifiers(feature_matrix)
    print('\nModels Finished Building! Ready for testing!')


# Builds the feature matrix for the training set
def build_test_feature_matrix():
    print('\nBuilding feature matrix for testing set')
    print('Testing data set used is: ' + testing_file_name + '\n')
    global test_features_matrix
    test_features_matrix = get_feature_matrix(testing_files, word_list)
    print('Testing Feature Matrix built!')


# Given an email path if the models are built then we can predict if that single email is spam or ham
def predict_single_email(email_path):
    file = [email_path]

    test_f_matrix = get_feature_matrix(file, word_list)  # Use the global word list as the features

    if nb_model != 0:
        result_nb = nb_model.predict(test_f_matrix)
    else:
        print('Models still need to be trained, please call build_models(), please make sure you are not running in '
              'pytest mode')
        return

    if svm_model != 0:
        result_svm = svm_model.predict(test_f_matrix)
    else:
        print('Models still need to be trained, please call build_models(), please make sure you are not running in '
              'pytest mode')
        return

    if result_nb[0] == 1:
        nb_prediction = 'Spam'
    else:
        nb_prediction = 'Ham'

    if result_svm[0] == 1:
        svm_prediction = 'Spam'
    else:
        svm_prediction = 'Ham'

    print('The Naive Bayes Classifier Predicts ' + str(email_path) + ' as: ' + nb_prediction)
    print('The Support Vector Machine Predicts ' + str(email_path) + ' as: ' + svm_prediction + '\n')


if __name__ == '__main__':

    # Call this to build classifiers (will take some time)
    build_models()
    # Call this to build test feature matrix (will take time)
    build_test_feature_matrix()

    # Call to test classifiers
    test_classifiers()
    print('\nComplete!')

    # You can manually test the models by entering a single email path and seeing what it is classified as
    print('Demo of using single email prediction method to manually test the models\n')
    predict_single_email('enron2/ham/0015.1999-12-14.kaminski.ham.txt')
    predict_single_email('enron2/spam/0011.2001-06-28.SA_and_HP.spam.txt')





