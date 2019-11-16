from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC


def make_dictionary(root_directory):
    emails_dirs = [os.path.join(root_directory, f) for f in os.listdir(root_directory)]
    all_words = []
    for emails_dir in emails_dirs:
        emails = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]
        # for d in dirs:
        # emails = [os.path.join(d, f) for f in os.listdir(d)]
        for mail in emails:
            with open(mail) as m:
                for line in m:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()

    for item in list_to_remove:
        if not item.isalpha():
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)

    np.save('dict_enron.npy', dictionary)

    return dictionary


def read_data(root_directory):
    spamFiles = [root_directory + '/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = [root_directory + '/ham/' + f for f in os.listdir('enron1/ham')]

    for file in spamFiles:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            print(text)


if __name__ == '__main__':
    root_dir = 'enron1'
    # fname = 'enron1/spam/0032.2003-12-19.GP.spam.txt'
    # f = open(fname)
    # print(f)
    read_data(root_dir)
    # dictionary = make_dictionary(root_dir)
