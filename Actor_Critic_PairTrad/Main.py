from environment import Simulator
from agent import PolicyGradientAgent, CriticsAgent
import datetime as dt
import pandas as pd
import numpy as np
def main():
    #rule = {'Finance':['price_ICBC','price_taibao'], 'Electrical Engineering':['ee_600482','ee_600560'],
    #        'Real Estate':['housing_600240','housing_600743'],'Bio Medicine':['bio_600436','bio_600566']}
    rule = {'SP500 & HS300':['china_index_jijin','us_index_sp'] }

    def trade(pair,tick,iter):

        actions = ["buy", "sell", "hold"]

        n_iter = iter
        env_train = Simulator(pair, dt.datetime(2017, 1, 2), dt.datetime(2018, 2,14) - dt.timedelta(50))
        agent = PolicyGradientAgent(lookback=env_train.init_state())

        for i in range(n_iter):

            env_train = Simulator(pair, dt.datetime(2017, 1, 2), dt.datetime(2018, 2,14)- dt.timedelta(50))

            agent.reset(lookback=env_train.init_state())
            #critic_agent = CriticsAgent(lookback=env.init_state())
            action = agent.init_query()


            while env_train.has_more():
                action = actions[action]
                print("#########Runner: Taking action################", env_train.date, action)
                reward, state = env_train.step(action)
                action = agent.query(state, reward)

            if i == n_iter - 1:
                env_train.visualize(item,n_iter,'In Sample')


        env_test = Simulator(pair, start_date = dt.datetime(2018, 2,14) - dt.timedelta(73), end_date = dt.datetime(2018, 4, 27))
        #print(env_test.dateIdx)
        agent.reset(lookback=env_test.init_state())
        #print(env_test.prices.index[-1])
        #print(env_test.)
        while env_test.has_more():
            print(env_test.date)
            action = actions[action] # map action from id to name
            print("#############################################################")
            print ("Runner Test: Taking action", env_test.date, action)
            print("##########################################################")
            reward, state = env_test.step(action)
            action = agent.query(state, reward)

        date, price, date_sell, price_sell, date_buy, price_buy, date_hold, price_hold= env_test.visualize(item,n_iter, 'Out Sample')
        return date,price,date_sell,price_sell,date_buy,price_buy,date_hold,price_hold

    for iter in [50]:
        for item in rule:
            date, price, date_sell, price_sell, date_buy, price_buy, date_hold, price_hold= trade(rule[item],item, iter)
            pd.DataFrame(date).to_csv("date.csv")
            pd.DataFrame(price).to_csv("pair-re.csv")
            pd.DataFrame(date_sell).to_csv("date_sell.csv")
            pd.DataFrame(price_sell).to_csv("price_sell.csv")
            pd.DataFrame(date_buy).to_csv("date_buy.csv")
            pd.DataFrame(price_buy).to_csv("price_buy.csv")
            pd.DataFrame(date_hold).to_csv("date_hold.csv")
            pd.DataFrame(price_hold).to_csv("price_hold.csv")

if __name__ == '__main__':
    main()
