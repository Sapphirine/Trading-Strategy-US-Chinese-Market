import csv
# import re
# import sys
import pickle
# import dateutil.parser
# import datetime
import numpy as np
# import os
# import random


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        # score = random.random()
        normalised_window = [[((float(p[1]) / float(window[0][1])) - 1),((float(p[2]) / float(window[0][2])) - 1),
                              ((float(p[3]) / float(window[0][3])) - 1),((float(p[4]) / float(window[0][4])) - 1),
                              ((float(p[5]) / float(window[0][5])) - 1)] for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def getdata(filepath):
    data = np.genfromtxt(filepath, delimiter=',')
    seqlen = 30
    sequencelen = seqlen + 1
    result = []
    z_list = []
    for id in range(len(data) - sequencelen+1):
        result.append(data[id: id + sequencelen])
        z_list.append(data[id][1])
    result = normalise_windows(result)
    train = np.array(result)
    X = train[:, :-1]
    Y = train[:, -1, 0:1]
    Z = np.array(z_list).reshape((len(z_list),1))


    return X,Y,Z


X = None
Y = None
Z = None

data = ['us_test','ch_test'] 
for item in data:
    X, Y, Z = getdata('./data/%s.csv'%item)
    print (X.shape)
    print (Y.shape)
    print (Z.shape)
    dict ={}
    dict['X']=X
    dict['Y']=Y
    dict['Z']=Z
    with open ('./dict/%s.pickle'%item,'wb') as f:
        pickle.dump(dict, f)













