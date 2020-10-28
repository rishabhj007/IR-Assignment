import script
import numpy

if __name__ == '__main__':

    file = 'dataset.txt'

    k = input('Input shingle size, mostly between 4-8')
    hashmap = script.DNA_shingling(file, k)
    arr = script.vectorizer(hashmap)
    sim_mat = script.min_hash(arr)
    numpy.save("Similarity Matrix", sim_mat, allow_pickle=False, fix_imports=False)

    commonset = set()
    sim_mat = numpy.load('Similarity Matrix.npy')
    band = int(input("Input the band size"))  # take = 5
    sim_threshold = input("Input the similarity threshold. A value between \'0.7\'-\'0.9\'")  # take = 0.8

    for i in range(0, int(100 / band)):
        hashtable = {}
        for j in range(0, sim_mat.shape[1]):
            k = tuple(sim_mat[band * i:band * (i + 1), j])
            if k in hashtable:
                sim = script.jaccard_sim(sim_mat[:, hashtable[k]], sim_mat[:, j])
                if sim > sim_threshold:
                    commonset.add((hashtable[k], j))
            else:
                hashtable[k] = j

    print(commonset)
