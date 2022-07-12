#!/usr/bin/env python
# coding: utf-8

# ### 随机追踪（Random Pursuit）

# In[1]:


# TODO: not using forward_losses, but use MSE which calculates error for each line to a target line
# every 7 points padding 0 to 49 points
# every 49 points use FFT, DCT and then calculate cov matrix to weight
# data reverse then use weight to predict


# In[2]:


import random
import warnings
import numpy as np
import copy
from sklearn import linear_model
from multiprocessing import Pool
from functools import partial


# In[3]:


warnings.filterwarnings("ignore", category=DeprecationWarning)

thread_num = 8


# In[4]:


def random_pursuit(data_x, data_y, threshold, inx_h, l_inx, N, inx_p = -1, n_pred = 1, s_size = -1, percent_value = -1, with_self_resid = False, method = 'inverse'):
    """random pursuit core function.
    # Arguments
        data_x: A ndarray, a series x value.
        data_y: A ndarray, corresponding series of y value.
        threshold: Decimal, if abs(data_y - pred_y) >= threshold, will replace data_y with pred_y.
        inx_h: Integer, max index of history data.
        l_inx: Integer, length of history data used for pursuit. Or say, the size of sliding window.
        N: Interger, number of points in each regression.
        inx_p: Integer, the index of the data point to predict. If set to -1, the point with index (inx_h+1) will be predicted.
        n_pred: Interger, number of points to predict with sliding window.
        s_size: Interger, the moving step size of sliding window.
        percent_value: Float, max percentage for weight recaculation.
        method: String, 'inverse' or 'softmax', how weights are calculated from losses
    # Returns
        pred: Prediction value for the data point you specified.
        alert_idxs: The indexes of the alerted data points.
        data_y_copy: The refined series of y value.
    # Raises
        ValueError: Input error OR calculation error.
    """

    if len(data_x) == 0 or len(data_y) == 0 or len(data_x) < len(data_y): # len(data_x) can be bigger than len(data_y), which means you can have some x for prediction, you don't know there corresponding y value.
        raise ValueError('Input data is invalid, Exit.')
    if inx_p >= len(data_x) or (inx_p == -1 and inx_h + 1 >= len(data_x)): # Make sure data_x[inx_h + 1] exits
        raise ValueError('inx_p provided is out of the bound of data_x, Exit.')
    if inx_h - l_inx + 1 < 0: # Make sure data_x[inx_h - l_inx + 1] exits
        raise ValueError('l_inx provided is out of the bound of data_x, Exit.')
    if inx_p == -1: # If inx_p is set to -1, the point with index (inx_h + 1) will be predicted
        inx_p = inx_h + 1
    if inx_p + n_pred - 1 >= len(data_x): # Make sure data_x[inx_p + n_pred - 1] exits
        raise ValueError('n_pred provided is out of the bound of data_x, Exit.')


    pool = Pool(thread_num)
    
    history_inx = list(range(inx_h - l_inx + 1, inx_h + 1))
    random.shuffle(history_inx)
    n_lines = len(history_inx) // N # TODO: if cannot be completely divided
    shuffle_inx = np.resize(np.array(history_inx), (N, n_lines)) # Generate a shuffled ndarray of size: N * n_lines
    #print("Initiate - Shuffled inx is: ")
    #print(shuffle_inx)
    print("Initiate - Shuffled inx with shape: %s" % str(shuffle_inx.shape))

    if s_size > 0 and n_pred > 0: # Do slide prediction
        n_iter = n_pred
    else: # Predict once
        n_iter = 1
    all_iter = n_iter

    reg_results = []
    pred_results = {}
    alert_idxs = []
    real_jumps = set()
    
    data_y_copy = copy.deepcopy(data_y) # As we may need to replace some values in data_y, deep copy it first

    while n_iter > 0:
        slide_time = all_iter - n_iter
        if (slide_time + 1) % 500 == 0:
            print("... ... This is iteration %s ... ..." % (slide_time + 1))
        #print("====================================================")
        if slide_time == 0: # The 1st time, run all n_lines regressions
            # 1. Prepare data
                # train_sets is (train_x, train_y, line_idx)
                # rest_sets is a combination of (rest_x, rest_y)
                # test_set is a combination of (test_x, test_y)
            train_sets = [(data_x[list(shuffle_inx[i])], data_y_copy[list(shuffle_inx[i])], i) for i in range(n_lines)]
            rest_sets = list()
            for index in range(len(shuffle_inx)):
                shuffle_inx_copy = list(copy.deepcopy(shuffle_inx))
                del shuffle_inx_copy[index]
                rest_inxs = list(np.array(shuffle_inx_copy).flat)
                rest_sets.append((data_x[rest_inxs], data_y_copy[rest_inxs]))
            if len(data_y_copy) <= inx_p:
                test_set = (data_x[inx_p], None)
            else:
                test_set = (data_x[inx_p], data_y_copy[inx_p])

            # 2. Partial function Preparasion and Map to run regression
            func = partial(fit_linear_model, test_set)
            args = zip(train_sets, rest_sets) # Feed (train_set, rest_set), train_set is used to do train regression, rest_set is used to calculate residual
            reg_results = pool.map(func, args)
            #print("Regression result - inx, (regr.coef_[0,0], regr.intercept_[0]), regr._residues[0], rest_resid_sum[0], pred_y[0,0], pred_resid:")
            #print(reg_results) # It looks like: [(0, (0.99, 2.00), 2.07e-12, 5.861236525792118e-28, 51.00, None), (...), (...)]
            #print("====================================================")

            # 3. Calculate weights using forward_losses (either with/without the prediction resid)
            if with_self_resid:
                forward_losses = [reg_result[2] + reg_result[3] for reg_result in reg_results]
            else:
                forward_losses = [reg_result[3] for reg_result in reg_results]
            weights = calculate_weight(forward_losses, method)
            #print("Weights are %s" % str(weights))

            # 4. Final prediction using weighted avg
            preds = [reg_result[4] for reg_result in reg_results]
            pred = np.average(preds, weights = weights)
            pred_results[slide_time] = pred
            #print("Prediction is %e" % pred)
            #print("====================================================")
            
            if np.abs(pred - data_y_copy[inx_p]) >= threshold:
                # Replace data_y_copy if it exceeds pred to much
                data_y_copy[inx_p] = pred
                idx_alert = inx_p - l_inx
                alert_idxs.append(idx_alert)
                # print("ALERT - Data at index %s has exceeded threshold" % idx_alert)
            
            # Record real jump point
            data_y_this_loop = data_y[history_inx]
            real_jumps_this_loop = np.array(history_inx)[np.abs(data_y_this_loop - np.mean(data_y_this_loop)) >= threshold]
            if len(real_jumps_this_loop) > 0:
                # print("REAL - Data at index %s has exceeded threshold" % list(real_jumps_this_loop))
                real_jumps.update(list(real_jumps_this_loop))
                

        else: # Window slides, re-calculate forward_losses, weights and pred
            # 1. Check if there are history data available for new iteration
            if inx_h + 1 >= len(data_x):
                raise ValueError('After window slides, inx_h is out of the bound of data_x, Exit.')
            if inx_h + 1 >= len(data_y_copy):
                raise ValueError("WARN: After window slides, no real y value is available for index inx_h.") # TODO: may use predicted y value here
                
            # 2. Slide the window and refactor shuffle_inx
            inx_in = inx_h
            inx_out = inx_h - l_inx
            #print("After window slides, point at %s is involved, point at %s is eliminated" % (inx_in, inx_out))
            #print("====================================================")
            inx_replace = np.where(shuffle_inx == inx_out)
            shuffle_inx[inx_replace] = inx_in
            #print("After window slides, Shuffled inx changes to: ")
            #print(shuffle_inx)
            #print("====================================================")

            # 3. Training - we will not do delta training here
            # train_sets is (train_x, train_y, line_idx)
            # rest_sets is a combination of (rest_x, rest_y)
            # test_set is a combination of (test_x, test_y)
            train_sets = [(data_x[list(shuffle_inx[i])], data_y_copy[list(shuffle_inx[i])], i) for i in range(n_lines)]
            rest_sets = list()
            for index in range(len(shuffle_inx)):
                shuffle_inx_copy = list(copy.deepcopy(shuffle_inx))
                del shuffle_inx_copy[index]
                rest_inxs = list(np.array(shuffle_inx_copy).flat)
                rest_sets.append((data_x[rest_inxs], data_y_copy[rest_inxs]))
            if len(data_y_copy) <= inx_p:
                test_set = (data_x[inx_p], None)
            else:
                test_set = (data_x[inx_p], data_y_copy[inx_p])

            func = partial(fit_linear_model, test_set)
            args = zip(train_sets, rest_sets) # Feed (train_set, rest_set), train_set is used to do train regression, rest_set is used to calculate residual
            reg_results = pool.map(func, args)
            #print("Regression result - inx, (regr.coef_[0,0], regr.intercept_[0]), regr._residues[0], rest_resid_sum[0], pred_y[0,0], pred_resid:")
            #print(reg_results) # It looks like: [(0, (0.99, 2.00), 2.07e-12, 5.861236525792118e-28, 51.00, None), (...), (...)]
            #print("====================================================")

            # 4. Calculate weights using forward_losses (either with/without the prediction resid)
            if with_self_resid:
                #print("Include self residual into loss")
                forward_losses = [reg_result[2] + reg_result[3] for reg_result in reg_results]
            else:
                #print("NOT Include self residual into loss")
                forward_losses = [reg_result[3] for reg_result in reg_results]
            weights = calculate_weight(forward_losses, method)
            #print("Weights are %s" % str(weights))

            # 5. Final prediction using weighted avg
            preds = [reg_result[4] for reg_result in reg_results]
            pred = np.average(preds, weights = weights)
            pred_results[slide_time] = pred
            #print("Prediction is %e" % pred)
            #print("----------------------------------------------------")

            if np.abs(pred - data_y_copy[inx_p]) >= threshold:
                # Replace data_y_copy if it exceeds pred to much
                data_y_copy[inx_p] = pred
                idx_alert = inx_p - l_inx
                alert_idxs.append(idx_alert)
                # print("ALERT - Data at index %s has exceeded threshold" % idx_alert)
            
            # Record real jump point
            new_history_inx = list(range(inx_h - l_inx + 1, inx_h + 1))
            data_y_this_loop = data_y[new_history_inx]
            real_jumps_this_loop = np.array(new_history_inx)[np.abs(data_y_this_loop - np.mean(data_y_this_loop)) >= threshold]
            if len(real_jumps_this_loop) > 0:
                # print("REAL - Data at index %s has exceeded threshold" % list(real_jumps_this_loop))
                real_jumps.update(list(real_jumps_this_loop))
                
        # Update iterators
        inx_h += 1
        inx_p += 1
        n_iter -= 1

    # Close the pool
    pool.close()
    return pred_results, alert_idxs, data_y_copy, real_jumps


# In[5]:


def fit_linear_model(test_set, train_rest_pair):
    """fit linear regression core function.
    # Arguments
        test_set: A combination, (test_x, test_y).
        train_rest_pair: A combination, (train_sets, rest_sets). Here 'train_sets' is a combination of (train_x, train_y, line_idx), 'rest_sets' is a combination of (rest_x, rest_y)
    """

    # TODO: Delta training
    # - Perform linear regression only for new line
    # - Calculate rest_resid for all the lines (New line resid calculate, Old line resid updated by removing 1 point and adding 1 point)

    train_sets, rest_sets = train_rest_pair

    # 1. Train the regression model
    regr = linear_model.LinearRegression()
    train_x = np.resize(train_sets[0], (-1, 1))
    train_y = np.resize(train_sets[1], (-1, 1))
    inx = train_sets[2] # The uid of this line, corresponding to its row id in shuffle_inx
    test_x = np.array(test_set[0]).reshape(-1, 1)
    test_y = np.array(test_set[1]).reshape(-1, 1)
    regr.fit(train_x, train_y) # training

    # 2. Predict the test_set
    pred_y = regr.predict(test_x) # predict - test_x should be 2D
    if test_y != None: # If know the real y value
        pred_resid = (pred_y - test_y) ** 2 # MSE
    else: # If do not know the real y value
        pred_resid = None

    # 3. calculate the projection residual by predicting the rest_x
    rest_x = np.array(rest_sets[0]).reshape(-1, 1)
    rest_y = np.array(rest_sets[1]).reshape(-1, 1)
    pred_rest = regr.predict(rest_x)
    rest_resid_sum = np.sum((pred_rest - rest_y) ** 2) # MSE

    # 4. Each thread (a line) returns something like this: (0, (0.99, 2.00), 2.07e-12, 5.861236525792118e-28, 51.00, None)
    return inx, (regr.coef_[0,0], regr.intercept_[0]), regr._residues[0], rest_resid_sum, pred_y[0,0], pred_resid


# In[11]:


def calculate_weight(forward_losses, method='inverse'):
    """calculate weights given losses
    # Arguments
        forward_losses: losses list of each sub-model.
        method: 'inverse' or 'softmax', determine how to calculate weights from losses.
    """
    #print("Forward losses to calculate weights are %s" % forward_losses)
    # calulate inverse
    forward_losses = np.array(forward_losses)
    if method == 'inverse':
        inverse_losses = (1.0 + 1e-100) / (forward_losses + 1e-100) # avoid divided by 0
        weights = scale_one(inverse_losses) # scale to one
    elif method == 'softmax':
        weights = np.exp(forward_losses) / np.sum(np.exp(forward_losses), axis=0)
    else:
        raise ValueError('Weight calculation method is invalid, Exit.')
    return weights


# In[7]:


def scale_one(x):
    return x / np.sum(x, axis=0)
