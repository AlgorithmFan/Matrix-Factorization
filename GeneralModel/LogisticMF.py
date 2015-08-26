#!usr/bin/env python
#coding:utf-8

'''
https://github.com/MrChrisJohnson/logistic-mf/blob/master/logistic_mf.py
Input:
    users_items_rates: user_id * item_id * rate
    K: the number of latent features
    num_steps: the number of iterations
    LAMBDA: the learning rate
    GAMMA: the regulation rate
Output:
    Predict rates on users_items_rates.
'''
from __future__ import division
import numpy as np

class LogisticMF():
    def __init__(self):
        self.bu = {}
        self.bi = {}
        self.pu = {}
        self.qi = {}
        self.threshold = 0.00001

    def initParameters(self, users_items_rates, K):
        # calculate the user bias and item bias
        for user_id, item_id, rate in users_items_rates:
            self.bu.setdefault(user_id, 0)
            self.bi.setdefault(item_id, 0)
            self.pu.setdefault(user_id, np.random.normal(size=K))
            self.qi.setdefault(item_id, np.random.normal(size=K))

    def learn(self, users_items_rates, K, num_steps, Gamma, Lambda):

        for step in range(num_steps):
            for user_id, item_id, rate in users_items_rates:
                self.bu[user_id]