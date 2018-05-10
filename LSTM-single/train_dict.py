import csv
import pickle
import numpy as np
import os

folder = './data'
print (os.getcwd())

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
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
    for id in range(len(data) - sequencelen+1):
        result.append(data[id: id + sequencelen])
    result = normalise_windows(result)
    train = np.array(result)
    np.random.shuffle(train)
    X = train[:, :-1]
    Y = train[:, -1, 0:1]
    return X,Y


X = None
Y = None
cnt =0
for f in os.listdir(folder):
    if 'csv' in f:
        csvpath = os.path.join(folder,f)
        # print (csvpath)
        tX,tY = getdata(csvpath)
        # print (type(tX))
        if type(X) is not np.ndarray:
            X = tX
            Y = tY
        else:
            X = np.concatenate((X, tX))
            Y = np.concatenate((Y, tY))
        print ("at %d turn"%cnt)
        cnt+=1
        print (X.shape)
        print (Y.shape)
dict ={}
dict['X']=X
dict['Y']=Y

with open ('./dict/alldata.pickle','wb') as f:
    pickle.dump(dict, f)









