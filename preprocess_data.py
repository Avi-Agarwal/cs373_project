from pandas import DataFrame
import os

from main import loadData


def preprocessing():
    data = loadData()
    tst = data.iloc[0]
    print(tst)
    print(data.head())
    # Remove non words

if __name__ == '__main__':
    preprocessing()
