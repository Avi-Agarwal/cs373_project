from pandas import DataFrame
import os
import collections
import numpy

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


    print(data.describe())
    return data

loadData()


def make_Dictionary(train_dir):
    #emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    emails = spamFiles
    all_words = []
    for mail in emails:
        with open(mail, 'r', encoding="utf8", errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 1:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = collections.Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below)
    print(dictionary)
    return dictionary



if __name__ == '__main__':
    print('Hey Team')
    #loadData()
    make_Dictionary('enron1')
