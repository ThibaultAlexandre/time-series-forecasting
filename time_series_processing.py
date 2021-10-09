# time_series_processing : This file contains functions for processing time series

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


def average(l,average_size):
    res = np.zeros(len(l)-average_size+1)
    for i in range(len(l)-average_size+1):
        res[i] = (sum(l[i:i+average_size])/len(l[i:i+average_size]))
    return res

def sliding_windows(data, seq_length, average_size):
    X = []
    y = []
    data = average(np.array(data).reshape(-1),average_size).reshape(-1,1)
    for i in range(len(data)-seq_length):
        Xi = data[i:(i+seq_length)]
        yi = data[i+seq_length]
        X.append(Xi)
        y.append(yi)
    return np.array(X),np.array(y)

def process_data(data, seq_length,average_size, train_proportion, scaler):
    data = np.array(data)
    
    #Fitting on training data
    train_size = int(len(data) * train_proportion)
    scaler.fit(data[0:train_size])
    
    #Transforming all data
    data_normalized = scaler.transform(data)
    X,y = sliding_windows(data_normalized,seq_length, average_size)
    
    #Raw Target
    y_raw = data_normalized[seq_length-1+average_size:]
    y_raw = Variable(torch.Tensor(y_raw))
    
    #All data
    X_data = Variable(torch.Tensor(X))
    y_data = Variable(torch.Tensor(y))
    
    #Training 
    X_train = Variable(torch.Tensor(X[0:train_size]))
    y_train = Variable(torch.Tensor(y[0:train_size]))
    
    #Testing
    X_test = Variable(torch.Tensor(X[train_size:len(X)]))
    y_test = Variable(torch.Tensor(y[train_size:len(X)]))

    return y_raw, X_data, y_data, X_train, y_train, X_test, y_test

def process_data_diff(data, seq_length, average_size, train_proportion, scaler):
    data = np.array(data)
    
    #Differencing the time series
    data_diff = np.diff(data.reshape(data.shape[0])).reshape(-1,1)

    #Fitting on training data
    train_size = int(len(data_diff) * train_proportion)    
    scaler.fit(data_diff[0:train_size])
    
    #Transforming all data
    data_normalized = scaler.transform(data_diff)
    X,y = sliding_windows(data_normalized,seq_length,average_size)
    
    #Raw Target
    y_raw = data_normalized[seq_length-1+average_size:]
    y_raw = Variable(torch.Tensor(y_raw))
    
    #All data
    X_data = Variable(torch.Tensor(X))
    y_data = Variable(torch.Tensor(y))
    
    #Training 
    X_train = Variable(torch.Tensor(X[0:train_size]))
    y_train = Variable(torch.Tensor(y[0:train_size]))
        
    #Test
    X_test = Variable(torch.Tensor(X[train_size:len(X)]))
    y_test = Variable(torch.Tensor(y[train_size:len(X)]))

    return y_raw, X_data, y_data, X_train, y_train, X_test, y_test