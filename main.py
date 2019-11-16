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


# Go through emails and create a word dictionary
def make_word_dictionary(train_dir):
    spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    all_files = spamFiles + hamFiles
    emails = all_files
    # emails = spamFiles
    all_words = []
    for mail in emails:
        with open(mail, 'r', encoding="utf8", errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 1:
                    words = line.split()
                    all_words += words

    dictionary = collections.Counter(all_words)
    #print(dictionary)
    return dictionary


# Get rid of non alpa containing words and single digit characters
def dictionary_preprocessing(dictionary):
    check_list = dictionary.keys()
    for word in list(check_list):
        if not word.isalpha() or len(word) == 1:
            del dictionary[word]
    dictionary = dictionary.most_common(1000)  # get most common 100 words after removing alpha numerics
    return dictionary


if __name__ == '__main__':
    print('Hey Team')
    dict = make_word_dictionary('enron1')
    dict = dictionary_preprocessing(dict)
    print(dict)
