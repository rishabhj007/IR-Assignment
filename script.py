from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import string
import re


def preprocessing(doc):
    translator = re.compile('[%s]' % re.escape(string.punctuation))
    doc = translator.sub(' ', doc)
    ps = PorterStemmer()
    tokens = word_tokenize(doc)
    doc2 = []
    for token in tokens:
        token = token.lower()
        token = ps.stem(token)
        doc2.append(token)

    return doc2


def shingling(doc):
    k = int(input("Input shingle size"))
    shingles = set()
    i = k
    while i < len(doc):
        shingles.add(tuple(doc[i-k:i]))
        i+=1
    return shingles


if __name__ == '__main__':
    doc = input("Input doc")
    doc = preprocessing(doc)
    print(shingling(doc))
