# import hashlib
#
#
# f = open('ml-1m.zip',"rb")
# hash_md5 = hashlib.md5()
# for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#
#
# if hash_md5.hexdigest() == "c4d9eecfca2ab87c1945afe126590906":
#     print("Match")
import numpy as np
import cmath


def cos_simi(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def corroborative_normalize(vec):
    return vec - np.mean(vec)


def corroborative(filename):
    f = open(filename)


def SVD(matrix, energy_factor):
    AAT = np.matmul(matrix, matrix.T)
    ATA = np.matmul(matrix.T, matrix)

    w1, v1 = np.linalg.eig(AAT)
    w2, v2 = np.linalg.eig(ATA)

    index = w1.argsort(kind='mergesort')[::-1]
    w1 = w1[index]
    v1 = v1[:, index]

    index = w2.argsort(kind='mergesort')[::-1]
    w2 = w2[index]
    v2 = v2[:, index]
    v2_T = v2.T

    S = np.zeros(matrix.shape)

    for i in range(len(w1)):
        if i == len(w1) or i == len(w2):
            break
        S[i, i] = cmath.sqrt(w1[i]).real

    return v1, S, v2_T


A = np.array([[1, 1, 1, 0, 0], [3, 3, 3, 0, 0], [4, 4, 4, 0, 0], [5, 5, 5, 0, 0], [0, 2, 0, 4, 4], [0, 0, 0, 5, 5],
              [0, 1, 0, 2, 2]])
print(A)
U, S, Vt = SVD(A, 4)
print(np.matmul(U, np.matmul(S, Vt)))
