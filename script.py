from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import string
import re
import random
import sympy
import numpy as np

global shingle_dict
global lines


def generate_hash_func(size):
    a = random.randint(1, size)
    b = random.randint(1, size)
    c = sympy.nextprime(size)

    hash_list = []
    for i in range(0, size):
        hash_list.append((a*i+b) % c)

    return hash_list


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

# DNA_shingling:
# input : doc - the link to the dataset
#         k - the shingle size usually between 5 - 9
# returns : a dictionary with key as the DNA string and
#           the set of shingles as the paired value


def DNA_shingling(doc,k):
    dataset = open(doc, encoding='utf-8')
    global lines
    lines = [line.rstrip('\n') for line in dataset]
    dataset.close()

    lines = lines[1:]
    hashmap = {}
    count = 0
    for line in lines:
        dna = line.split(" ")[0]
        n = len(dna)
        i = k
        shingles = set()
        while i < n:
            shingles.add(dna[(i-k):i])
            i+=1
        hashmap[count] = shingles
        count+=1
    return hashmap

# vectorizer:
# input : a dictionary of with key as integer denoting a document
#         and value as a set of the shingles present in the document
# func : creates a numpy array with shingles as rows and documents
#        as columns, the array has 0 or 1 value based on whether the
#        shingle is present in document or not
# return type : a 2D numpy array which is basically a document vector


def vectorizer(hashmap):
    mega_set = set()
    for shingles in hashmap.values():
        mega_set.update(shingles)
    n = len(mega_set)
    k = len(hashmap)
    arr = np.empty((n,k),dtype=np.int8)
    global shingle_dict
    count = 0
    for shingle in mega_set:
        shingle_dict[count] = shingle
        for i in range (0,k):
            if shingle in hashmap[i]:
                arr[count,i] = 1
            else:
                arr[count,i] = 0
        count+=1
    return arr

#TODO
def min_hash(arr):
    x,y = arr.shape() # x = number of shingles,y = number of documents
    sig_mat = np.empty((100,y),dtype=np.int32)

    for i in range (0,100):
        generate_hash_func(x)



if __name__ == '__main__':
    for i in range (0,10):
        print(generate_hash_func(8))
