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
import random
import math
from scipy import sparse as sp


def cos_simi(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def corroborative_normalize(vec):
    return vec - np.mean(vec)


def corroborative(filename):
    f = open(filename)


def SVD(matrix):
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


def CUR(matrix, r):
    m, n = matrix.shape
    matrix_T = matrix.T
    mat = np.square(matrix)
    f = np.sum(mat)
    mat_T = mat.T
    p_R = []
    p_C = []
    C = np.empty([r,m])
    R = np.empty([r,n])
    for i in range(0,m):
        p_R.append(np.sum(mat[i]) / f)
    for i in range(0,n):
        p_C.append(np.sum(mat_T[i]) / f)
    tmp_r = [j for j in range(m)]
    r_select = random.choices(tmp_r, weights=p_R, k=r)
    tmp_c = [j for j in range(n)]
    c_select = random.choices(tmp_c, weights=p_C, k=r)
    i = 0
    for j in r_select:
        R[i] = (matrix[j] / math.sqrt(r * p_R[j]))
        i+=1
    i = 0
    for j in c_select:
        C[i] = (matrix_T[j] / math.sqrt(r * p_C[j]))
        i += 1
    C = C.T

    W = np.empty([r,r])
    a = 0
    b = 0
    for i in r_select:
        b = 0
        for j in c_select:
            W[a][b] = matrix[i][j]
            b += 1
        a += 1

    X, S, YT = np.linalg.svd(W)
    XT = X.T
    Y = YT.T
    SI = np.zeros([S.size,S.size])
    for i in range(0,S.size):
        if S[i] != 0:
            SI[i,i] = 1/S[i]

    SI = np.matmul(SI, SI)
    U = np.matmul(XT, np.matmul(SI, Y))

    return C, U, R


A = np.array([[1, 1, 1, 0, 0], [3, 3, 3, 0, 0], [4, 4, 4, 0, 0], [5, 5, 5, 0, 0], [0, 0, 0, 4, 4], [0, 0, 0, 5, 5],
              [0, 0, 0, 2, 2]])
print(A)
U, S, Vt = CUR(A, 2)
print(U)
print(S)
print(Vt)

print(np.matmul(U, np.matmul(S, Vt)))
