import os
import sys
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import KFold

import os

def loadData():
    data = DataFrame({'text': [], 'classifier': []})

    spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]

    spam = []
    spam_names = []
    for file in spamFiles:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            spam.append({'text': text, 'classifier': 'spam'})
            spam_names.append(file)

    data = data.append(DataFrame(spam, spam_names))

    ham = []
    ham_names = []
    for file in hamFiles:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            ham.append({'text': text, 'classifier': 'not spam'})
            ham_names.append(file)

    data = data.append(DataFrame(ham, ham_names))

    data = data.reindex(numpy.random.permutation(data.index))
    print(data.describe())

    return data

def train(data, folds = 6):
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    scores = []
    confusion = numpy.array([[0, 0], [0, 0]])
    print("Training with %d folds" % folds)
    for i, (train_indices, test_indices) in enumerate(KFold(folds).split(data)):
        train_text = data.iloc[train_indices]['text'].values
        train_y = data.iloc[train_indices]['classifier'].values.astype(str)

        test_text = data.iloc[test_indices]['text'].values
        test_y = data.iloc[test_indices]['classifier'].values.astype(str)

        print("Training for fold %d" % i)
        pipeline.fit(train_text, train_y)
        print("Testing for fold %d" % i)
        predictions = pipeline.predict(test_text)

        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions, pos_label='spam')
        scores.append(score)
        print("Score for %d: %2.2f" % (i, score))
        print("Confusion matrix for %d: " % i)
        print(confusion)

    print('Total emails classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)
    return pipeline

train(loadData())