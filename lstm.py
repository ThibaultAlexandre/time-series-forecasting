# LstmNeuralNetwork : This class defines a Long Short Term Memory neural network with built-in train, plot and accuracy method
# that allows you to train and evaluate your model

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import trange

class LstmNeuralNetwork(nn.Module):
    def __init__(self,input_size, num_layers, hidden_size, seq_length, output_size = 1):
        super(LstmNeuralNetwork, self).__init__()
        #Attributes from nn.Module
        self.input_size = input_size #input size
        self.output_size = output_size #output size
        self.num_layers = num_layers #number of layers
        self.hidden_size = hidden_size #hidden state
        
        #New attributes
        self.seq_length = seq_length #sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first = True) #lstm
        
        self.fc =  nn.Linear(hidden_size, output_size) #fully connected linear


    def forward(self,X):
        h_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size)) #internal state
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(X, (h_0, c_0)) #lstm with input, hidden, and internal state
        final_state = hn.view(self.num_layers, X.size(0), self.hidden_size)[-1]
                
        # Propagate input through fully connected linear neuron
        out = self.fc(final_state)
        
        return out
    
    
    def _train(self, num_epochs, learning_rate, criterion, X_train, y_train, X_test, y_test):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) 
        t = trange(num_epochs+1)
        for epoch in t:
            #Pass through the neural network
            train_outputs = self.forward(X_train) 
            test_outputs = self.forward(X_test) 

            #Reset gradients to zero  
            optimizer.zero_grad() 

            train_loss = criterion(train_outputs, y_train)
            test_loss = criterion(test_outputs,y_test)

            #Backprogagation step
            train_loss.backward()

            #Update weights and bias of the network
            optimizer.step()

            #Print train and test loss
            t.set_description("Epoch: %d, Train loss: %1.5f, Test loss: %1.5f" % (epoch, train_loss.item(),test_loss.item()))
    
    def get_horizon_predictions(self, X_data, y_data, scaler, horizon):
        total_days = X_data.shape[0]
        predictions = np.ndarray(shape=(total_days-horizon+1,horizon, self.output_size), dtype=float)
        for i in range(total_days-horizon+1):
            X_temp = X_data[i].unsqueeze(0)
            for j in range(horizon):
                next_prediction = self(X_temp)
                predictions[i,j,:] = scaler.inverse_transform(next_prediction.detach().numpy())
                X_temp_shifted = X_temp[:,1:].reshape((1,self.seq_length-1,self.input_size))
                X_temp = torch.cat((X_temp_shifted,next_prediction.unsqueeze(0)),1)
        return predictions
    
    def get_accuracy(self, X_data, y_data, naive_prediction, scaler, last_day):
        lstm_prediction = scaler.inverse_transform(self(X_data).detach().numpy())[:,:1]
        real_values = scaler.inverse_transform(y_data.numpy())[:,:1]

        naive_error = mean_squared_error(real_values[:last_day:1],naive_prediction[:last_day:1])
        lstm_error = mean_squared_error(real_values[:last_day:1],lstm_prediction[:last_day:1])

        naive_r2 = r2_score(real_values[:last_day:1], naive_prediction[:last_day:1])
        lstm_r2 = r2_score(real_values[:last_day:1], lstm_prediction[:last_day:1])

        print('Mean squared error up to the ' + str(last_day) + ' day using the naive prediction : ' + str(naive_error))
        print('Mean squared error up to the ' + str(last_day) + ' day using the lstm prediction : ' + str(lstm_error))
        print('R2 up to the ' + str(last_day) + ' day using the naive prediction : ' + str(naive_r2))
        print('R2 up to the ' + str(last_day) + ' day using the lstm prediction : ' + str(lstm_r2))
    
    def get_accuracy_with_horizon(self, X_data, y_data, naive_prediction_without_horizon, scaler, last_day, horizon):
        lstm_prediction = self.get_horizon_predictions(X_data, y_data, scaler, horizon)[:,:,0]
        total_days = lstm_prediction.shape[0]
        real_values_without_horizon = scaler.inverse_transform(y_data.numpy())[:,:1]
        real_values = np.ndarray(shape=(total_days,horizon), dtype=float)
        naive_prediction = np.ndarray(shape=(total_days,horizon), dtype=float)
        for i in range(total_days):
            for j in range(horizon):
                real_values[i,j] = real_values_without_horizon[i+j]
                naive_prediction[i,j] = naive_prediction_without_horizon[i]
                
        naive_error = mean_squared_error(real_values[:last_day:1].flatten(),naive_prediction[:last_day:1].flatten())
        lstm_error = mean_squared_error(real_values[:last_day:1].flatten(),lstm_prediction[:last_day:1].flatten())

        naive_r2 = r2_score(real_values[:last_day:1].flatten(), naive_prediction[:last_day:1].flatten())
        lstm_r2 = r2_score(real_values[:last_day:1].flatten(), lstm_prediction[:last_day:1].flatten())

        print('Mean squared error up to the ' + str(last_day) + ' day using the naive prediction : ' + str(naive_error))
        print('Mean squared error up to the ' + str(last_day) + ' day using the lstm prediction : ' + str(lstm_error))
        print('R2 up to the ' + str(last_day) + ' day using the naive prediction : ' + str(naive_r2))
        print('R2 up to the ' + str(last_day) + ' day using the lstm prediction : ' + str(lstm_r2))


    def plot(self, X_data, y_data, scaler):
        lstm_prediction = scaler.inverse_transform(self(X_data).detach().numpy())[:,:1]
        real_values = scaler.inverse_transform(y_data.numpy())[:,:1]
        plt.plot(real_values, label='Actual value', c = 'b')
        plt.plot(lstm_prediction, label='LSTM Prediction', c = 'r')
        plt.legend()
        plt.show()
        
    def plot_with_horizon(self, X_data, y_data, scaler, horizon, drawing_jump):
        colors = ['r','g','k','y']
        real_values = scaler.inverse_transform(y_data.numpy())[:,:1]
        predictions = self.get_horizon_predictions(X_data, y_data, scaler, horizon)[:,:,0]
        plt.plot(real_values, label='Actual value', c = 'b')
        for i in range(0,predictions.shape[0], drawing_jump):
            plt.plot(np.arange(i,i+horizon),predictions[i], c = colors[int(i/drawing_jump) % len(colors)])
        plt.legend()
        plt.show()