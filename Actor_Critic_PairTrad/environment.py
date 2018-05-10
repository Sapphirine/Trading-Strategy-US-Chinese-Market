import numpy as np
import pandas as pd

import datetime as dt
import csv

verbose = False
'''
Define the get_data function to store data
'''
def get_data(symbols, dates_range, update):
    import os
    import pandas as pd
    cwd = os.getcwd()
    dir = cwd + "/data/"
    df = pd.DataFrame()
    symbols.append('spy')
    for symbol in symbols:
        print("symbol", symbol)
        filename = dir + symbol + '.csv'
        data = pd.read_csv(filename, index_col=["Date"], parse_dates=['Date'])
        #print(data.index)
        if df.empty:
            if 'Close' in data:
                df[symbol] = data['Close']
            else:
                df[symbol] = data['Rate']
        else:
            if 'Close' in data:
                df[symbol] = data['Close']
            else:
                df[symbol] = data['Rate']
    return df[df.index.isin(dates_range)].sort_index()





class Simulator(object):

    def __init__(self, symbols,
        start_date=dt.datetime(2014,1,1),
        end_date= dt.datetime(2018,1,1)):

        # Define the training time period

        self.dates_range = pd.date_range(start_date, end_date)
        print(self.dates_range)

        self.init_cash = 100000

        #keep the information of the portfolio for visualization
        self.data_out = []

        # Take the stocks symbols
        stock_symbols = symbols[:]

        # Store the prices into the DataFrame
        prices_all = get_data(symbols, self.dates_range, True)
        print(prices_all.index)
        self.stock_A = stock_symbols[0]
        self.stock_B = stock_symbols[1]



        # first trading day
        self.dateIdx = 0
        self.date = prices_all.index[0]
        self.start_date = start_date
        self.end_date = end_date

        self.prices = prices_all[stock_symbols]
        self.prices_SPY = prices_all['spy']

        self.portfolio = {'cash': self.init_cash, 'a_vol': [], 'a_price': [], 'b_vol': [], 'b_price': [], 'longA': 0}
        self.port_val = self.port_value_for_output()



    def init_state(self, lookback=50):
        """
        return init states of the market
        """
        states = []
        for _ in range(lookback):
            states.append(self.get_state(self.date))
            self.dateIdx += 1
            self.date = self.prices.index[self.dateIdx]

        return states

    def step(self, action):



        buy_volume = 100
        abs_return_A = 0
        pct_return_A = 0
        abs_return_B = 0
        pct_return_B = 0

        if (action == 'buy'):
            if (self.portfolio['longA'] >= 0):

                if verbose: print('---BUY WITH longA greater/equal 0')
                long_cost = buy_volume * self.prices.ix[self.date, self.stock_A]

                if verbose: print('Buying ' + str(buy_volume) + ' shares of ' + self.stock_A + ' at a price of $' + str(self.prices.ix[self.date, self.stock_A]) + ' per share, for a total cost of $' + str(long_cost) + '.')
                short_cost = buy_volume * self.prices.ix[self.date, self.stock_B]

                if verbose: print('Shorting ' + str(buy_volume) + ' shares of ' + self.stock_B + ' at a price of $' + str(self.prices.ix[self.date, self.stock_B]) + ' per share, for a total cost of $' + str(short_cost) + '.')
                total_cost = short_cost + long_cost

                if verbose: print('Total cost is $' + str(total_cost))

                if verbose: print('Pre-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['cash'] -= total_cost

                if verbose: print('Post-transaction cash is $' + str(self.portfolio['cash']))

                self.portfolio['a_vol'].append(buy_volume)
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['b_vol'].append(buy_volume)
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longA'] = 1

                if verbose: print(self.portfolio)
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()

                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('---END OF BUY WITH longA greater/equal 0')

            else: #longA < 0 --> sell in reverse
                if verbose: print('---BUYING (ACTUALLY SELLING) WITH longA < 0')
                if verbose: print('Selling our long investment of ' + str(self.portfolio['b_vol'][0]) + ' shares for $' + str(self.prices.ix[self.date, self.stock_B]))
                long_initial = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                long_return = self.portfolio['b_vol'].pop(0) * self.prices.ix[self.date, self.stock_B]
                abs_return_B = long_return - long_initial
                pct_return_B = float(abs_return_B) / long_initial
                if verbose:print('Long initial is ' + str(long_initial))
                if verbose:print('Long return is ' + str(long_return))
                if verbose:print('Long return - long initial = ' + str(abs_return_B))
                if verbose:print('(long return - long initial) / long initial = ' + str(pct_return_B))
                self.portfolio['b_price'].pop(0)
                if verbose: print('Return is $' + str(long_return))
                if verbose: print('Cover our long investment of ' + str(self.portfolio['a_vol'][0]) + ' shares that we bought for $' + str(self.portfolio['a_price'][0]) + ' and add to it a gain of $' + str((self.portfolio['a_price'][0] - self.prices.ix[self.date, self.stock_A])) + ' (' + str((self.portfolio['a_price'][0])) + ' - ' + str(self.prices.ix[self.date, self.stock_A]) + ') for each of our ' + str(self.portfolio['a_vol'][0]) + ' stocks.')
                short_initial = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                abs_return_A = (self.portfolio['a_vol'][0] * (self.portfolio['a_price'][0] - self.prices.ix[self.date, self.stock_A]))
                short_return = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                short_return += (self.portfolio['a_vol'].pop(0) * (self.portfolio['a_price'].pop(0) - self.prices.ix[self.date, self.stock_A]))
                pct_return_A = float(abs_return_A) / short_initial
                if verbose:print('Short initial is ' + str(short_initial))
                if verbose:print('Short return is ' + str(short_return))
                if verbose:print('Absolute return for short is ' + str(abs_return_A))
                if verbose:print('Percetn return for short is ' + str(pct_return_A))
                if verbose: print('Short return is $' + str(short_return))
                if verbose: print('Old cash is $' + str(self.portfolio['cash']))
                new_cash = self.portfolio['cash'] + long_return + short_return
                self.portfolio['cash'] = new_cash
                if verbose: print('New cash is $' + str(self.portfolio['cash']))
                self.portfolio['longA'] = -1 if (len(self.portfolio['a_vol']) > 0) else 0
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('Old portfolio value is $' + str(old_port_val))
                if verbose: print('New portfolio value is $' + str(self.port_val))
                if verbose: print('Reward is $' + str(reward))

        elif (action == 'sell'):
            if (self.portfolio['longA'] > 0):
                if verbose: print('---SELLING WITH longA > 0')
                if verbose: print('Selling our long investment of ' + str(self.portfolio['a_vol'][0]) + ' shares for $' + str(self.prices.ix[self.date, self.stock_A]))
                long_initial = self.portfolio['a_vol'][0] * self.portfolio['a_price'][0]
                long_return = self.portfolio['a_vol'].pop(0) * self.prices.ix[self.date, self.stock_A]
                abs_return_A = long_return - long_initial
                pct_return_A = float(abs_return_A) / long_initial
                self.portfolio['a_price'].pop(0)
                if verbose: print('Return is $' + str(long_return))
                if verbose: print('Cover our long investment of ' + str(self.portfolio['b_vol'][0]) + ' shares that we bought for $' + str(self.portfolio['b_price'][0]) + ' and add to it a gain of $' + str((self.portfolio['b_price'][0] - self.prices.ix[self.date, self.stock_B])) + ' (' + str((self.portfolio['b_price'][0])) + ' - ' + str(self.prices.ix[self.date, self.stock_B]) + ') for each of our ' + str(self.portfolio['b_vol'][0]) + ' stocks.')
                short_initial = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                abs_return_B = (self.portfolio['b_vol'][0] * (self.portfolio['b_price'][0] - self.prices.ix[self.date, self.stock_B]))
                short_return = self.portfolio['b_vol'][0] * self.portfolio['b_price'][0]
                short_return += (self.portfolio['b_vol'].pop(0) * (self.portfolio['b_price'].pop(0) - self.prices.ix[self.date, self.stock_B]))
                pct_return_B = float(abs_return_B) / short_initial
                if verbose: print('Short return is $' + str(short_return))
                if verbose: print('Old cash is $' + str(self.portfolio['cash']))
                new_cash = self.portfolio['cash'] + long_return + short_return
                self.portfolio['cash'] = new_cash
                if verbose: print('New cash is $' + str(self.portfolio['cash']))
                self.portfolio['longA'] = 1 if (len(self.portfolio['a_vol']) > 0) else 0
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('Old portfolio value is $' + str(old_port_val))
                if verbose: print('New portfolio value is $' + str(self.port_val))
                if verbose: print('Reward is $' + str(reward))
            else: # longA <= 0 --> buy in reverse
                if verbose: print('---SELLING (ACTUALLY BUYING) WITH long <= 0')
                long_cost = buy_volume * self.prices.ix[self.date, self.stock_B]
                if verbose: print('Buying ' + str(buy_volume) + ' shares of ' + self.stock_B + ' at a price of $' + str(self.prices.ix[self.date, self.stock_B]) + ' per share, for a total cost of $' + str(long_cost) + '.')
                short_cost = buy_volume * self.prices.ix[self.date, self.stock_A]
                if verbose: print('Shorting ' + str(buy_volume) + ' shares of ' + self.stock_A + ' at a price of $' + str(self.prices.ix[self.date, self.stock_A]) + ' per share, for a total cost of $' + str(short_cost) + '.')
                total_cost = short_cost + long_cost
                if verbose: print('Total cost is $' + str(total_cost))
                if verbose: print('Pre-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['cash'] -= total_cost
                if verbose: print('Post-transaction cash is $' + str(self.portfolio['cash']))
                self.portfolio['a_vol'].append(buy_volume)
                self.portfolio['a_price'].append(self.prices.ix[self.date, self.stock_A])
                self.portfolio['b_vol'].append(buy_volume)
                self.portfolio['b_price'].append(self.prices.ix[self.date, self.stock_B])
                self.portfolio['longA'] = -1
                if verbose: print(self.portfolio)
                old_port_val = self.port_val
                self.port_val = self.port_value_for_output()
                if verbose: print(self.portfolio)
                reward = self.port_val - old_port_val
                if verbose: print('---END OF SELLING (ACTUALLY BUYING) WITH long <= 0')
        else: #hold
            if verbose: print('---HOLDING')
            old_port_val = self.port_val
            self.port_val = self.port_value_for_output()
            if verbose: print(self.portfolio)
            reward = self.port_val - old_port_val
        print ("port value", self.port_val)
        self.data_out.append(self.date.isoformat()[0:10]  + ',' + str(self.port_val) + ',' + action)
        if verbose: print(self.data_out)
        state = self.get_state(self.date)
        if verbose: print(state)
        self.dateIdx += 1
        if self.dateIdx < len(self.prices.index):
            self.date = self.prices.index[self.dateIdx]

        if verbose: print(self.get_state(self.date))
        if verbose: print('New date is')
        if verbose: print(self.date)
        if verbose: print('Reward is ' + str(reward))
        return (reward, state)

    def get_state(self, date):

        if date not in self.dates_range:
            if verbose: print('Date was out of bounds.')
            if verbose: print(date)
            exit

        return [self.prices.ix[date, self.stock_A]/self.prices.ix[0, self.stock_A] - self.prices.ix[date, self.stock_B]/self.prices.ix[0, self.stock_B],
            self.port_val / self.init_cash - 1,
            ]


    def port_value(self):
        value = self.portfolio['cash']
        if (len(self.portfolio['a_vol']) > 0):
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.portfolio['a_price'][i])
        if (len(self.portfolio['b_vol']) > 0):
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.portfolio['b_price'][i])
        return value


    def port_value_for_output(self):
        value = self.portfolio['cash']
        if (self.portfolio['longA'] > 0):
            value += (sum(self.portfolio['a_vol']) * self.prices.ix[self.date, self.stock_A])
            for i in range(len(self.portfolio['b_vol'])):
                value += (self.portfolio['b_vol'][i] * self.portfolio['b_price'][i])
                value += (self.portfolio['b_vol'][i] * (self.portfolio['b_price'][i] - self.prices.ix[self.date, self.stock_B]))
        if (self.portfolio['longA'] < 0):
            value += (sum(self.portfolio['b_vol']) * self.prices.ix[self.date, self.stock_B])
            for i in range(len(self.portfolio['a_vol'])):
                value += (self.portfolio['a_vol'][i] * self.portfolio['a_price'][i])
                value += (self.portfolio['a_vol'][i] * (self.portfolio['a_price'][i] - self.prices.ix[self.date, self.stock_A]))
        return value

    def has_more(self):
        if ((self.dateIdx < len(self.prices.index)) == False):
            print('\n\n\n*****')
            print(self.baseline())
            print('*****\n\n\n')
        return self.dateIdx < len(self.prices.index)

    def baseline(self):
        num_shares = self.init_cash / self.prices_SPY[0]
        return num_shares * self.prices_SPY[-1]


    def visualize(self, tick, niter, type):
        data = self.data_out

        store = {'buy': [], 'sell': [], 'hold': []}
        date = []
        price = []

        for i in range(len(data)):
            temp = data[i].split(',')
            date.append(temp[0])
            price.append(temp[1])
            store[temp[2]].append(i)

        date_buy = []
        price_buy = []
        for item in store['buy']:
            date_buy.append(date[item])
            price_buy.append(price[item])
        date_sell = []
        price_sell = []
        for item in store['sell']:
            date_sell.append(date[item])
            price_sell.append(price[item])
        date_hold = []
        price_hold = []
        for item in store['hold']:
            date_hold.append(date[item])
            price_hold.append(price[item])

        import numpy as np
        from bokeh.plotting import figure, output_file, show

        # output to static HTML file
        output_file("{}: {}_Sector_{}iters.html".format(type,tick,niter))

        def datetime(x):
            return np.array(x, dtype=np.datetime64)

        p1 = figure(x_axis_type="datetime", title="Portfolio Value",plot_width=800, plot_height=400)
        p1.grid.grid_line_alpha = 0.3
        p1.xaxis.axis_label = 'Date'
        p1.yaxis.axis_label = 'Net Value Of Portfolio'

        p1.line(datetime(date), price, color='#A6CEE3', legend='RL Portfolio')
        p1.triangle(datetime(date_buy), price_buy, legend="Long Position", fill_color="red", size=5)
        p1.triangle(datetime(date_sell), price_sell, legend="Short Position", fill_color="green", size=5)
        p1.circle(datetime(date_hold), price_hold, legend="Hold Position", fill_color="blue", size=5)

        show(p1)
        return date,price,date_sell,price_sell,date_buy,price_buy,date_hold,price_hold

