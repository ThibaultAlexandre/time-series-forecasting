# time-series-forecasting
This is a tentative to train a Long Short Term Memory (LSTM) neural network for time series forecasting. 

- For the Tesla stock values from 2010 to 2020, see tesla.ipynb
- For the Covid-19 cases in France, see covid-19.ipynb

The file lstm.py contains a class LstmNeuralNetwork that defines a pytorch neural network with lstm layers
The file processing.py contains some functions to process a time series like a moving average, a sliding window and a differencing
