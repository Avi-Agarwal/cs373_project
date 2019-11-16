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

def train(data, folds):
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])

    for i, (train_indices, test_indices) in enumerate(KFold(folds).split(data)):
        train_data = data.iloc[train_indices]['text'].values
        train_result = data.iloc[train_indices]['classifier'].values.astype(str)

        pipeline.fit(train_data, train_result)

    print('Number of emails classified:', len(data))
    return pipeline


pipeline = train(loadData(), 2)

print()
print(pipeline.predict(["Hello, the spaceship is ready."]))
print(pipeline.predict(["HOT SINGLES IN YOUR AREA!!"]))