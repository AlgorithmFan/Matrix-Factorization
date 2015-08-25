#!usr/bin/env python
#coding:utf-8

'''
Input:
    R: a matrix to be factorized, dimension N*M
    P: a left matrix, dimension N*K
    Q: a right matrix, dimension M*K
    K: the number of latent features
    num_steps: the number of iterations
    alpha: the learning rate
    beta: the regulation rate
Output:
    the left matrix P and the right matrix Q.
'''

import numpy as np
import scipy

class NonNegativeMatrixFactorization:
    def __init__(self):
        self.N, self.M = 0, 0

    def learn(self, R):
        pass