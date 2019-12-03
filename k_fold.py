import os

import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def loadValidationData():
    spam_files = ['enron2/spam/' + f for f in os.listdir('enron2/spam')]
    ham_files = ['enron2/ham/' + f for f in os.listdir('enron2/ham')]

    spam = []
    ham = []

    for file in spam_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            spam.append(f.read())

    for file in ham_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            ham.append(f.read())

    return ham, spam

def loadData():
    load_data = DataFrame({'text': [], 'classifier': []})

    spam_files = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    ham_files = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]

    spam = []
    spam_names = []
    for file in spam_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            spam.append({'text': text, 'classifier': 'spam'})
            spam_names.append(file)

    load_data = load_data.append(DataFrame(spam, spam_names))

    ham = []
    ham_names = []
    for file in ham_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            ham.append({'text': text, 'classifier': 'not spam'})
            ham_names.append(file)

    load_data = load_data.append(DataFrame(ham, ham_names))

    load_data = load_data.reindex(numpy.random.permutation(load_data.index))

    return load_data


def train(training_data, folds=5, algorithm=1):
    pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', KNeighborsClassifier() if algorithm == 1 else SVC(gamma='scale'))
    ])

    for i, (train_indices, test_indices) in enumerate(KFold(folds).split(training_data)):
        print(f"Training fold {i+1}")
        train_values = training_data.iloc[train_indices]['text'].values
        train_result = training_data.iloc[train_indices]['classifier'].values.astype(str)

        pipeline.fit(train_values, train_result)

    return pipeline

def test_accuracy(pipeline, ham, spam):
    ham_results = pipeline.predict(ham).tolist()
    spam_results = pipeline.predict(spam).tolist()

    ham_correct = ham_results.count("not spam")
    spam_correct = spam_results.count("spam")

    print(f"Correctly found {ham_correct} out of {len(ham)} ham")
    print(f"Correctly found {spam_correct} out of {len(spam)} spam")

    return ham_correct / len(ham) * 100, spam_correct / len(spam) * 100


print("Loading data...")
data = loadData()
print("Data loaded.")

algorithm = input("Which algorithm to run? 1=Knn 2=Svm. \n>")

if algorithm == "2":
    algorithm = 2
    print("Running SVM")
else:
    algorithm = 1
    print("Running Knn (default)")

plot = input("Plot results or run test? 1=plot 2=test \n>")
# plot = "1"
if plot == "1":
    print("Loading validation data...")
    ham, spam = loadValidationData()

    test_type = input("Test for k-folds or data size? 1 or 2\n>")

    if test_type == "1":
        k_folds = range(2, 13)
        ham_accuracies = []
        spam_accuracies = []

        for k_fold in k_folds:
            print(f"\nTesting k-folds {k_fold}")
            ham_accuracy, spam_accuracy = test_accuracy(train(data, folds=k_fold, algorithm=algorithm), ham, spam)
            ham_accuracies.append(ham_accuracy)
            spam_accuracies.append(spam_accuracy)

        plt.plot(k_folds, ham_accuracies, label='Ham Accuracy')
        plt.plot(k_folds, spam_accuracies, label='Spam Accuracy')
        plt.xlabel('K-Folds')
        plt.ylabel('Percent Correct')
        plt.title(('SVM' if algorithm == 2 else 'KNN') + " Accuracies vs K-Folds")
        plt.legend()
        plt.show()

    else:
        sizes = range(500, 5500, 500)
        ham_accuracies = []
        spam_accuracies = []

        for size in sizes:
            print(f"\nTesting size {size}")
            ham_accuracy, spam_accuracy = test_accuracy(train(data.sample(size), algorithm=algorithm), ham, spam)
            ham_accuracies.append(ham_accuracy)
            spam_accuracies.append(spam_accuracy)

        plt.plot(sizes, ham_accuracies, label='Ham Accuracy')
        plt.plot(sizes, spam_accuracies, label='Spam Accuracy')
        plt.xlabel('Data Size')
        plt.ylabel('Percent Correct')
        plt.title(('SVM' if algorithm == 2 else 'KNN') + " Accuracies vs Training Data Size")
        plt.legend()
        plt.show()

    exit(0)


folds = input("How many folds to use in k-fold? Higher k takes longer to train. \n>")
if not folds.isdigit():
    folds = "10"

print("Training...")
pipeline = train(data, folds=int(folds), algorithm=algorithm)
print("Trained")

# Test Prediction
while True:
    test_email = input("Enter your email to predict, or type 'test' to test against validation set and get results."
                       " \n>")

    if test_email != "test":
        print(pipeline.predict([test_email]))

    else:
        print("Importing validation set...")
        spam_files = ['enron2/spam/' + f for f in os.listdir('enron2/spam')]
        ham_files = ['enron2/ham/' + f for f in os.listdir('enron2/ham')]

        spam = []
        ham = []

        for file in spam_files:
            with open(file, 'r', encoding="utf8", errors='ignore') as f:
                spam.append(f.read())

        for file in ham_files:
            with open(file, 'r', encoding="utf8", errors='ignore') as f:
                ham.append(f.read())

        print("Predicting results")
        ham_results = pipeline.predict(ham).tolist()
        spam_results = pipeline.predict(spam).tolist()

        ham_correct = ham_results.count("not spam")
        spam_correct = spam_results.count("spam")

        print(f"Correctly found {ham_correct} out of {len(ham)} ham")
        print(f"Correctly found {spam_correct} out of {len(spam)} spam")

        break
