import os
import numpy as np
import random
import copy
import csv
from env_full import MarketEnv
from model_builder import MarketDeepQLearningModelBuilder
from collections import deque
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from os import walk
from datetime import datetime
import matplotlib
import pandas as pd



class DeepQ:

    def __init__(self, env, gamma=0.85, model_file_name=None, test = False, random = False):
        self.env = env
        self.gamma = gamma
        self.model_filename = model_file_name
        self.memory = deque(maxlen=1000000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.005
        self.random = random

        self.model = MarketDeepQLearningModelBuilder().buildModel()
        self.fixed_model = MarketDeepQLearningModelBuilder().buildModel()


        if test:
            self.model.load_weights(self.model_filename)

    def epsilon_reset(self):
        self.epsilon = 1.0


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):

        batchs = min(batch_size, len(self.memory))
        batchs = np.random.choice(len(self.memory), batchs)
        losses = []

        true_q = []
        train_states = []
        self.fixed_model.set_weights(self.model.get_weights())
        for i in batchs:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                predict_f = self.fixed_model.predict(next_state)[0]

                target = reward + self.gamma * np.amax(predict_f)

            target_f = self.model.predict(state)

            target_f[0][action] = target
            

            train_states.append(state[0])
            true_q.append(target_f)

            # checkpointer = ModelCheckpoint(filepath=self.model_filename, mode='auto', verbose=1, monitor='val_loss', save_best_only=True)
        train_states = np.array(train_states)
        true_q = np.array(true_q)
        history = self.model.fit(state, target_f, epochs=1, verbose=0)
        losses=history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return losses

    def act(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= self.epsilon:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] 

    def act_predict_new(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']


        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        
        if np.random.rand() <= 0.1:
            sample = self.env.action_space.sample()
            
            while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()

            return sample


        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]

        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] 


    def act_random(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)
        
        sample = self.env.action_space.sample()
        
        while (cur_cash < trading_fee_general and sample == 0) or \
            (cur_stock <= 0 and sample == 1):
                sample = self.env.action_space.sample()
        return sample


    def act_predict(self, state, info):

        cur_cash = info["current_asset"]['cash']
        cur_stock = info["current_asset"]['stock']

        trading_fee_general = info["current_trading_price"]*(info["trading_feeRate"]+1)

        act_values = self.model.predict(state)[0]

        act_index = np.argsort(act_values)[::-1]
        flag = 0

        while (cur_cash < trading_fee_general and act_index[flag] == 0) or \
            (cur_stock <= 0 and act_index[flag] == 1):
            flag+=1


        return act_index[flag] 


    def train(self, code_stocks, max_episode=500, verbose=0):

        history = open('./record/history.txt', 'a')

        for e in range(max_episode):
            self.env._reset(code_stocks)
            state = self.env._render()

            game_over = False
            reward_sum = 0

            holds = 0
            buys = 0
            sells = 0
            print ("----",self.env.targetCode,"----")
            info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": self.env.current_trading_price, "current_asset_value":self.env.current_asset_value,\
                "current_asset":self.env.current_asset}
            while not game_over:

                action = self.act(state, info)

                if self.env.actions[action] == 'Hold':
                    holds += 1
                elif self.env.actions[action] == 'Buy':
                    buys += 1
                elif self.env.actions[action] == 'Sell':
                    sells += 1

                next_state, reward, game_over, info = self.env._step(action)

                current_asset_value = info['current_asset_value']

                self.remember(state, action, reward, next_state, game_over)

                state = copy.deepcopy(next_state)


                if game_over:
                    toPrint = '----episode----', e,'totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                    print (toPrint)
                    history.write(str(toPrint))
                    history.write('\n')
                    history.flush()

            self.model.save_weights(self.model_filename)
            print ('model weights saved')

            losses = self.replay(100)

            with open("./record/train_loss.txt","a") as file:
                file.write(str(losses)+'\n')

        history.close()

    def predict(self,code_stocks,return_1,acts):
        if os.path.exists('./record/test_history.txt'):
            os.remove('./record/test_history.txt')

        history = open('./record/test_history.txt', 'a')

        self.env._reset(code_stocks)
        state = self.env._render()

        game_over = False
        reward_sum = 0

        holds = 0
        buys = 0
        sells = 0

        if self.random == True:
            methods = 'random'
        else:
            methods = 'DQN'


        print ("----",self.env.targetCode,"----")
        info = {"trading_feeRate": self.env.trading_feeRate,"current_trading_price": self.env.current_trading_price, "current_asset_value":self.env.current_asset_value,\
                "current_asset":self.env.current_asset}

        while not game_over:

            if self.random == True:
                action = self.act_random(state,info)
            else:
                action = self.act_predict(state,info)


            if self.env.actions[action] == 'Hold':
                holds += 1
            elif self.env.actions[action] == 'Buy':
                buys += 1
            elif self.env.actions[action] == 'Sell':
                sells += 1

            return_1.append(round((info['current_asset_value']-self.env.startAssetValue)/self.env.startAssetValue,3))
            acts.append(self.env.actions[action])
            next_state, reward, game_over, info = self.env._step(action)

            current_asset_value = info['current_asset_value']


            state = copy.deepcopy(next_state)


            if game_over:
                toPrint = '----episode----', 'DQN','totalgains:', round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3), 'holds:', holds, 'buys:', buys, 'sells:', sells, 'mem size:', len(self.memory), '\n'
                print (toPrint)
                history.write(str(toPrint))
                history.write('\n')
                history.flush()

        history.close()
        return return_1, acts, round((current_asset_value-self.env.startAssetValue)/self.env.startAssetValue,3)

def exploreFolder(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for f in filenames:
            files.append(f.replace(".csv", ""))
        break
    return files

def to_date(initial):
    a = initial.split('/')
    return a[2]+'-'+a[1]+'-'+a[0]


if __name__ == "__main__":

    if os.path.exists("./record/train_loss.txt"):
        os.remove("./record/train_loss.txt")

    if os.path.exists("./record/history.txt"):
        os.remove("./record/history.txt")

    train = ['us_train','ch_train']
    test = ['us_test','ch_test']

    env = MarketEnv(dir_path="./data/", target_codes=train, sudden_death_rate=0.3, endDate = '2017-12-29') #370
    pg = DeepQ(env, gamma=0.80, model_file_name="./model/1.h5")
    
    for train_item in train:
        pg.train(code_stocks = train_item)


    with open('./data/us_test.csv','r') as ifile:
        reader = csv.reader(ifile)
        date = []
        price = []
        for row in reader:
            date.append(to_date(row[0]))
            price.append(float(row[1]))


    with open('./ALL/ch_test.csv','r') as ifile:
        reader = csv.reader(ifile)
        date_ch = []
        price_ch = []
        for row in reader:
            date_ch.append(to_date(row[0]))
            price_ch.append(float(row[1]))


    date = date[11:-1]
    date_ch = date_ch[11:-1]
    price = price[11:-1]
    price_ch = price_ch[11:-1]
    price_ = []
    price_ch_ = []
    
    for item in price:
        price_.append((item-price[0])/price[0])

    for item in price_ch:
        price_ch_.append((item-price_ch[0])/price_ch[0])

    
    env = MarketEnv(dir_path="./ALL/",target_codes=test, test = True, sudden_death_rate=0.3, endDate ='2018-04-27') #131
    test_obj = DeepQ(env, gamma=0.80, model_file_name="./model/1.h5", test = True)

    returns = {}
    returns_ ={} 


    for stock in test:
        return_1 = []
        acts = []
        return_1, acts, final = test_obj.predict(stock,return_1,acts)
        returns[stock] = acts
        returns_[stock] = return_1
        print('final',final)
    
    
    print('price_us',price_[-1])
    print('price_ch',price_ch_[-1])
    
        
    date_new = []
    date_new_ch = []
    
    for item in date:
        date_new.append(matplotlib.dates.date2num(datetime.strptime(item, '%Y-%m-%d')))

    for item in date_ch:
        date_new_ch.append(matplotlib.dates.date2num(datetime.strptime(item, '%Y-%m-%d')))
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))


    plt.plot(date_new, returns_[test[0]])
    plt.plot(date_new, price_)
    plt.title("Returns along timeline-us")
    plt.xlabel("trading timeline")
    plt.ylabel("return")
    plt.legend(['DQN','Price'], loc='upper left')
    plt.show()


    plt.plot(date_new_ch, returns_[test[1]])
    plt.plot(date_new_ch, price_ch_)
    plt.title("Returns along timeline-ch")
    plt.xlabel("trading timeline")
    plt.ylabel("return")
    plt.legend(['DQN','Price'], loc='upper left')
    plt.show()

