import os

import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def load_validation_data():
    validation_spam_files = ['enron2/spam/' + f for f in os.listdir('enron2/spam')]
    validation_ham_files = ['enron2/ham/' + f for f in os.listdir('enron2/ham')]

    validation_spam = []
    validation_ham = []

    for validation_file in validation_spam_files:
        with open(validation_file, 'r', encoding="utf8", errors='ignore') as f:
            validation_spam.append(f.read())

    for validation_file in validation_ham_files:
        with open(validation_file, 'r', encoding="utf8", errors='ignore') as f:
            validation_ham.append(f.read())

    return validation_ham, validation_spam


def load_data():
    raw_data = DataFrame({'text': [], 'classifier': []})

    load_spam_files = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    load_ham_files = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]

    load_spam = []
    spam_names = []
    for load_file in load_spam_files:
        with open(load_file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            load_spam.append({'text': text, 'classifier': 'spam'})
            spam_names.append(load_file)

    raw_data = raw_data.append(DataFrame(load_spam, spam_names))

    load_ham = []
    ham_names = []
    for load_file in load_ham_files:
        with open(load_file, 'r', encoding="utf8", errors='ignore') as f:
            text = f.read()
            load_ham.append({'text': text, 'classifier': 'not spam'})
            ham_names.append(load_file)

    raw_data = raw_data.append(DataFrame(load_ham, ham_names))

    raw_data = raw_data.reindex(numpy.random.permutation(raw_data.index))

    return raw_data


def train(training_data, num_k_folds=5, train_algorithm=1):
    trained_pipeline = Pipeline([
        ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', KNeighborsClassifier() if train_algorithm == 1 else SVC(gamma='scale'))
    ])

    for i, (train_indices, test_indices) in enumerate(KFold(num_k_folds).split(training_data)):
        print(f"Training fold {i + 1}")
        train_values = training_data.iloc[train_indices]['text'].values
        train_result = training_data.iloc[train_indices]['classifier'].values.astype(str)

        trained_pipeline.fit(train_values, train_result)

    return trained_pipeline


def test_accuracy(test_pipeline, test_ham, test_spam):
    test_ham_results = test_pipeline.predict(test_ham).tolist()
    test_spam_results = test_pipeline.predict(test_spam).tolist()

    test_ham_correct = test_ham_results.count("not spam")
    test_spam_correct = test_spam_results.count("spam")

    print(f"Correctly found {test_ham_correct} out of {len(test_ham)} ham")
    print(f"Correctly found {test_spam_correct} out of {len(test_spam)} spam")

    return test_ham_correct / len(test_ham) * 100, test_spam_correct / len(test_spam) * 100


def plot():
    print("Loading validation data...")
    plot_ham, plot_spam = load_validation_data()

    test_type = input("Test for k-folds or data size? 1 or 2\n>")

    if test_type == "1":
        k_folds = range(2, 13)
        ham_accuracies = []
        spam_accuracies = []

        for k_fold in k_folds:
            print(f"\nTesting k-folds {k_fold}")
            ham_accuracy, spam_accuracy = test_accuracy(train(data, num_k_folds=k_fold, train_algorithm=algorithm),
                                                        plot_ham,
                                                        plot_spam)
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
            ham_accuracy, spam_accuracy = test_accuracy(train(data.sample(size), train_algorithm=algorithm), plot_ham,
                                                        plot_spam)
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


def test():
    folds = input("How many folds to use in k-fold? Higher k takes longer to train. \n>")
    if not folds.isdigit():
        folds = "10"

    print("Training...")
    pipeline = train(data, num_k_folds=int(folds), train_algorithm=algorithm)
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


# MAIN

print("Loading data...")
data = load_data()
print("Data loaded.")

algorithm = input("Which algorithm to run? 1=Knn 2=Svm. \n>")

if algorithm == "2":
    algorithm = 2
    print("Running SVM")
else:
    algorithm = 1
    print("Running Knn (default)")

plot_or_test = input("Plot results or run test? 1=plot 2=test \n>")

if plot_or_test == "1":
    plot()
else:
    test()
