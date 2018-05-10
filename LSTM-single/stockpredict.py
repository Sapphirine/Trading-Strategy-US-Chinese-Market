from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import Sequential
import time
import numpy as np
import csv
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd

with open ('./dict/alldata_final.pickle','rb') as f:
    dicts = pickle.load(f)
X = dicts['X']
Y = dicts['Y']
# print ('Successfully load the X and Y dataset')
# print (X.shape)
# print (Y.shape)

split = int(len(X)*1)
X_train = X[:split,:,:]
Y_train = Y[:split,:]
# X_test = X[split:,:,:]
# Y_test = Y[:split,:]
# print (X_train.shape)
# print (Y_train.shape)
# print (X_test.shape)
# print (Y_test.shape)

# # X_train = np.random.randn(10,20,3)
# # Y_train = np.random.randn(10,20)
# print (X_train.shape)
# print ("++++++++")
# print (Y_train.shape)


learning_rate = 0.001
model = Sequential()
model.add(LSTM(
    input_dim=5,
    output_dim=512,
    return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(
    512,
    return_sequences=False))
model.add(Dropout(0.5))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate))
print ('compilation time : ', time.time() - start)

print(model.summary())


checkpointer = ModelCheckpoint('./model/lstm_1.h5', 
                           verbose=1,monitor='val_loss', save_best_only=True)
history = model.fit(
    X_train,
    Y_train,
    batch_size=50,
    nb_epoch=100,
    validation_split=0.1, callbacks=[checkpointer])


summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss as steps')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model = load_model('./model/lstm_1.h5')


def predict(model, data, Z, Y):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    predicted_seqs = []
    predicted_prices = []
    truth = []
    pdv = []
    tv = []
    for i in range(len(data)):
        #item = np.array(data[i]).reshape((1,20,3))
        onepredict = model.predict(np.array(data[i])[np.newaxis,:,:])[0,0]
        predicted_prices.append((onepredict+1)*Z[i])
        pdv.append((onepredict+1)*Z[i][0])
        truth.append((Y[i]+1)*Z[i])
        tv.append(((Y[i]+1)*Z[i])[0])
        print ((onepredict+1)*Z[i][0], ((Y[i]+1)*Z[i])[0])
        predicted_seqs.append(onepredict)

        
    return predicted_seqs, predicted_prices, truth,pdv,tv

with open ('./dict/ch_test.pickle','rb') as f:
    dicts = pickle.load(f)

X_test = dicts['X']
Y_test = dicts['Y']
Z_test = dicts['Z']

print (X_test.shape)
print (Z_test.shape)


predicted, predprices, truths, pdv, tv = predict(model, X_test, Z_test, Y_test)
# dict_price = {}
# dict_price['predict'] = pdv
# dict_price['trueValue'] = tv
# df = pd.DataFrame(dict_price,columns = ['predict','trueValue'])
# df.to_csv('./lstm_result_open_ch_sample.csv')
# exit()
# predprices is the predicted result

plt.plot(np.arange(len(X_test))+1,list(np.squeeze(predprices)), label='predict')
plt.plot(np.arange(len(X_test))+1,list(truths), label='truth')
plt.title('predicted sequence - CHINA')
plt.ylabel('Next day price')
plt.xlabel('days')
plt.legend()

# plt.savefig('test_f5.png')
plt.show()


pred = np.array(predicted).reshape([len(X_test),1])
predprices_new = np.array(predprices).reshape([len(X_test),1])
truths_new = np.array(truths).reshape([len(X_test),1])


print (np.sum(np.abs(pred-Y_test))/len(X_test))
print (np.sum(np.abs(predprices_new-truths_new))/len(X_test))

cnt = 0
sumcnt = 0
for i in range(len(truths)):
    if i==0:
        continue
    sumcnt+=1
    a = truths[i]>truths[i-1]
    b = predprices[i]>truths[i-1]
    if a==b:
        cnt+=1
print (sumcnt)
print (cnt/sumcnt)


