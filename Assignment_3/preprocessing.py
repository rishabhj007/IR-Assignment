import numpy as np


def get_array():

    ratings = open('./ml-1m/ratings.dat', encoding="utf-8")
    data = np.zeros((6041, 3953), dtype=np.int8)
    for line in ratings:
        user_id, movie_id, rating, timestamp = [int(i) for i in line.split("::")]
        data[user_id,movie_id] = rating
    ratings.close()

    return data.T


if __name__ == '__main__':
    get_array()
