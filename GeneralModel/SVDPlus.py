#!usr/bin/env python
#coding:utf-8

'''
Input:
    users_items_rates: user_id * item_id * rate
    K: the number of latent features
    num_steps: the number of iterations
    LAMBDA: the learning rate
    GAMMA: the regulation rate
Output:
    Predict rates on users_items_rates.
'''

import numpy as np

class SVDPlus():
    def __init__(self):
        self.bu = {}
        self.bi = {}
        self.pu = {}
        self.qi = {}
        self.mu = 0
        self.threshold = 0.00001

    def initParameters(self, users_items_rates, K):
        # calculate the average rate
        self.mu = np.mean(users_items_rates[:, 2])
        # calculate the user bias and item bias
        for user_id, item_id, rate in users_items_rates:
            self.bu.setdefault(user_id, 0)
            self.bi.setdefault(item_id, 0)
            self.pu.setdefault(user_id, np.random.normal(size=K))
            self.qi.setdefault(item_id, np.random.normal(size=K))


    def _calRate(self, user_id, item_id):
        rate = self.mu + self.bu[user_id] + self.bi[item_id] + np.sum(np.dot(self.pu[user_id], self.qi[item_id]))
        if rate > 5:
            rate = 5
        if rate < 1:
            rate = 1
        return rate

    def learn(self, users_items_rates, K, num_steps, LAMBDA, GAMMA):
        self.initParameters(users_items_rates, K)

        #
        for step in range(num_steps):
            old_rmse, rmse = 0.0, 0.0
            for user_id, item_id, rate in users_items_rates:
                predict_rate = self._calRate(user_id, item_id)
                error_ui = rate - predict_rate
                rmse += error_ui * error_ui
                self.bu[user_id] = self.bu[user_id] + GAMMA*(error_ui - LAMBDA*self.bu[user_id])
                self.bi[item_id] = self.bi[item_id] + GAMMA*(error_ui - LAMBDA*self.bi[item_id])
                self.pu[user_id] = self.pu[user_id] + GAMMA*(error_ui*self.qi[item_id] - LAMBDA*self.pu[user_id])
                self.qi[item_id] = self.qi[item_id] + GAMMA*(error_ui*self.pu[user_id] - LAMBDA*self.qi[item_id])
            rmse = np.sqrt(rmse/users_items_rates.shape[0])
            print 'The RMSE of Iteration {STEP} is {RMSE}.'.format(STEP=step, RMSE=rmse)
            if abs(old_rmse-rmse) < self.threshold:
                break
            else:
                old_rmse = rmse

    def predict(self, users_items_rates):
        output = list()
        rmse = 0.0
        for user_id, item_id, rate in users_items_rates:
            error_ui = rate - self._calRate(user_id, item_id)
            rmse += error_ui * error_ui
            output.append([user_id, item_id, rate])
        rmse = np.sqrt(rmse/users_items_rates.shape[0])
        print 'The RMSE is {RMSE}.'.format(RMSE=rmse)
        return output


