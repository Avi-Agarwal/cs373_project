from pandas import DataFrame
import os
import collections
import numpy

# Initial work


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
def make_word_dictionary():
    spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    all_files = spamFiles + hamFiles
    emails = all_files
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
    spamFiles = ['enron1/spam/' + f for f in os.listdir('enron1/spam')]
    hamFiles = ['enron1/ham/' + f for f in os.listdir('enron1/ham')]
    all_files = spamFiles + hamFiles
    f_matrix = numpy.zeros((len(all_files), len(dictionary)))
    dictionary_index = 0

    for file in all_files:
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






if __name__ == '__main__':
    print('Hey Team')
    dict = make_word_dictionary()
    dict = dictionary_preprocessing(dict)
    fmat = get_features(dict)
    print(fmat)
    print(dict)
