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


def DNA_shingling(doc):
    dataset = open(doc, encoding='utf-8')
    lines = [line.rstrip('\n') for line in dataset]
    dataset.close()

    k = input("Input shingle size")

    lines = lines[1:]
    dict = {}
    count = 0
    for line in lines:
        dna = line.split(" ")[0]
        n = len(dna)
        i = k
        shingles = set()
        while i < n:
            shingles.add(dna[(i-k):i])
            i+=1
        dict[count] = shingles
        count+=1
    return dict


def minhash(set):

    return


if __name__ == '__main__':
    doc = input("Input doc")
    doc = preprocessing(doc)
    print(DNA_shingling(doc))
