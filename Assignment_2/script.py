from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import string
import re
import random
import numpy as np
import time


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


# sattolo_algo is used to generate a random shuffling of
# an array of elements 0 to size-1.
# input : int size which is the number of shingles
# return : a numpy array in of elements in range 0 to size - 1
#          both inclusive shuffled in a random order.


def sattolo_algo(size):
    hash_list = np.arange(size)
    for i in range(0, size - 1):
        j = random.randrange(i + 1, size)  # i+1 instead of i
        hash_list[i], hash_list[j] = hash_list[j], hash_list[i]
    return hash_list


# sattolo_algo_arr is used to generate a random shuffling of
# an array of elements 0 to size-1 but the input is different then
# the other function
# input : a numpy array of elements in range 0 to size - 1
# return : a numpy array in of elements in range 0 to size - 1
#          both inclusive shuffled in a random order.


def sattolo_algo_arr(hash_list):
    size = np.size(hash_list)
    random.seed(time.time())
    for i in range(0, size - 1):
        j = random.randrange(i + 1, size)  # i+1 instead of i
        hash_list[i], hash_list[j] = hash_list[j], hash_list[i]
    return hash_list


# DNA_shingling:
# input : doc - the link to the dataset
#         k - the shingle size usually between 5 - 9
# returns : a dictionary with key as the DNA string and
#           the set of shingles as the paired value


def DNA_shingling(doc, k):
    dataset = open(doc, encoding='utf-8')
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
            shingles.add(dna[(i - k):i])
            i += 1
        hashmap[count] = shingles
        count += 1
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
    arr = np.empty((n, k), dtype=np.int8)
    shingle_dict = {}
    count = 0
    for shingle in mega_set:
        shingle_dict[count] = shingle
        for i in range(0, k):
            if shingle in hashmap[i]:
                arr[count, i] = 1
            else:
                arr[count, i] = 0
        count += 1
    return arr


def min_hash(arr):
    x, y = arr.shape  # x = number of shingles,y = number of documents

    sig_mat = np.empty((100, y), dtype=np.int32)
    sig_mat.fill(2147483647)
    hash_mat = np.empty((100, x), dtype=np.int32)

    hash_list = sattolo_algo(x)

    for i in range(0, 100):
        hash_list = sattolo_algo_arr(hash_list)
        hash_mat[i] = np.array(hash_list)

    hash_mat = np.transpose(hash_mat)

    # hash_mat has a dimensions of x rows and 100 columns
    # sig_mat has a dimensions of 100 rows and y columns

    for i in range(0, x):
        for j in range(0, y):
            if arr[i, j] == 1:
                for k in range(0, 100):
                    val = hash_mat[i, k]
                    if sig_mat[k, j] > val:
                        sig_mat[k, j] = val

    return sig_mat


def jaccard_sim(vec1,vec2):
    a = 0
    for i in range (0,100):
        if vec1[i] == vec2[i]:
            a += 1
    return a/100
