# Trading-Strategy-US-Chinese-Market

Mao Guan (mg3844)
Anlu Chen (ac4218)
Zijun Nie (zn2146)
   
There are two parts of the project: the first part is the causal analysis procedure; the second part is the three strategies including LSTM based single trading, DQN based single trading; and actor critic based pair trading. 

Apart from these four folders, the dataset folder would contain stock price data of US and Chinese markets in various sectors. For both markets, Electonic and Information, Finance, Energy and Household sectors are selected and there are approaximately 10 stocks in each sector. And In four main folders, there are also some related datasets, which majorly contain information of US and Chinese markets' general stock index price. 

To run the causal analysis,

In DQN based single trading, you shall run the file named as "dqn_full.py", which is the main code; "env_full.py" is the trading environment; "model_builder.py" is the neural network builder part. In some codes, in order to record the training process or created model, users may need to create two folders named as "record" and "model".

In LSTM based single trading, files named "train_dict.py" and "test_dict.py" serve as the scripts for creating training and test pickle files respectively. File of "stockpredict.py" is the main code for stock prediction and contains both training and testing process.

For the actor critic pair trading strategy,
