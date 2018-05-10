from keras.models import Model
from keras.layers import Dense, Input, Add, merge, Conv2D,Flatten
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU


class MarketDeepQLearningModelBuilder():
   
    def __init__(self, action_size = 3):
        self.action_size = action_size
        self.learning_rate = 0.001

    def buildModel(self):

        input1 = Input(shape=(2,))
        input2 = Input(shape=(5,10,1))

        pre = []

        net1 = Dense(8, activation='relu')(input1)

        net2 = Conv2D(filters=1024, kernel_size=(1, 10), strides=1, data_format='channels_last')(input2)
        net2 = LeakyReLU(alpha = 0.001)(net2)
        net2 = Flatten()(net2)
        net2 = Dense(48)(net2)
        net2 = LeakyReLU(alpha = 0.001)(net2)
        
        pre.append(net1)
        pre.append(net2)

        hidden1 = concatenate(pre, axis=1)
        hidden2 = Dense(96, activation='relu')(hidden1)
        hidden3 = Dense(128, activation='relu')(hidden2)
        hidden4 = Dense(128, activation='relu')(hidden3)
        hidden5 = Dense(96, activation='relu')(hidden4)
        output_main = Dense(self.action_size, activation='linear')(hidden5)


        model = Model(inputs = [input1,input2], outputs = output_main)


        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # print (model.summary())

        return model





