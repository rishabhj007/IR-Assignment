import script

if __name__ == '__main__':
    file = 'test.txt'
    k = 7
    hashmap = script.DNA_shingling(file, k)
    arr = script.vectorizer(hashmap)
    sim_mat = script.min_hash(arr)
