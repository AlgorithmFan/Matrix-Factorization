#!usr/bin/env python
#coding:utf-8

from GeneralModel.SVD import SVD
import numpy as np

def readData(filename):
    fp = open(filename)
    users_items_rates = []
    while True:
        line = fp.readline()
        temp = line.split('::')
        if len(temp) < 4: break
        user_id, item_id, rate = int(temp[0]), int(temp[1]), int(temp[2])
        users_items_rates.append([user_id, item_id, rate])
    fp.close()
    return np.array(users_items_rates)

def main(users_items_rates):
    mSVD = SVD()
    LAMBDA, GAMMA = 0.15, 0.04
    K = 30
    num_steps = 100
    mSVD.learn(users_items_rates, K, num_steps, LAMBDA, GAMMA)

if __name__ == '__main__':
    filename = 'Data/ratings.dat'
    print 'Read data from {Filename}.'.format(Filename = filename)
    users_items_rates = readData(filename)

    print 'The number of records is {num}.'.format(num=users_items_rates.shape[0])

    print 'Training data.'
    main(users_items_rates)

