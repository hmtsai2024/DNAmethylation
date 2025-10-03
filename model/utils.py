#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def mse_result(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    return {
        "mse_train": mean_squared_error(y_train, pred_train),
        "mse_val": mean_squared_error(y_val, pred_val),
        "mse_test": mean_squared_error(y_test, pred_test),
    }

def pearson_result(y_train, y_val, y_test, pred_train, pred_val, pred_test):
    results = []
    for j in range(y_train.shape[1]):
        results.append({
            "rppa": y_train.columns[j],
            "pearsonr_train": pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[0],
            "pearsonp_train": pearsonr(y_train.iloc[:, j], pred_train.iloc[:, j])[1],
            "pearsonr_val": pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[0],
            "pearsonp_val": pearsonr(y_val.iloc[:, j], pred_val.iloc[:, j])[1],
            "pearsonr_test": pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[0],
            "pearsonp_test": pearsonr(y_test.iloc[:, j], pred_test.iloc[:, j])[1],
        })
    return pd.DataFrame(results)

def save_weights(model, file_name):
    weights = [layer.get_weights() for layer in model.layers]
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)

