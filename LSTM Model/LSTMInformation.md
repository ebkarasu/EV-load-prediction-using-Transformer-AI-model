This script is written in order to implement LSTM deep learning model and test it by generating predictions of Electric Vehicle charging demands each hour.

Importing required libraries

```
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
```

# Implementing LSTM Model
   
   The LSTM class takes input size which is the number of the features, hidden dimension which is what dimension we want to be in middle there, and number of stacked layers is how many layers we want the LSTM model be. Then the nn.LSTM function is called to operation be done. It simply maintains a "memory" over long sequences, enabling them to learn patterns from time series data to give an estimation.
   
   ```
   class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
   ```


   Training the model, the model is trained with approximately ∼ 85% with our data with whatever epoch number is specified. It calculates and prints loss during each step.

   ```
   def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
   ```

   Validating or also can be called as testing of the model. It's done by approximately ∼ 15% with our data and validation loss is calculated.
    
   ```
    def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
   ```
   Lastly, calling the functions after defining the parameters.
   ```
    learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()
   ```




# Reading Data & Calculations
    
The data is first read from json file, then charging load is calculated for each hour on the given time interval.

   ```
f = open('acn-data.json')

dataList = json.load(f)

# Closing the file
f.close()

# Initializing the object array
classArray = [chargeInfo() for i in range(len(dataList))]
   ```

We calculate kW for each hour, normalize the data reshape and convert it to tensor to be able to give as input to the LSTM block.

   ```
scaler = MinMaxScaler(feature_range=(-1,1))
dataMatrix = scaler.fit_transform(dataMatrix)

X = dataMatrix[:,1:]
y = dataMatrix[:,0]
X = np.flip(X,axis=1)

split_index = 650

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

X_train = X_train.reshape((-1,4,1))
X_test = X_test.reshape((-1,4,1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

X_train = torch.from_numpy(X_train.copy()).float()
X_test = torch.from_numpy(X_test.copy()).float()
y_train = torch.from_numpy(y_train.copy()).float()
y_test = torch.from_numpy(y_test.copy()).float()
   ```

We denormalize the output data, flatten it and do required operations to plot it .

   ```
train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], 4+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dummies[:, 0]

dummies = np.zeros((X_train.shape[0], 4+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dummies[:, 0]

plt.title("Training data prediction")
plt.plot(new_y_train, label='Original')
plt.plot(train_predictions, label='Predictions')
plt.xlabel('Hours')
plt.ylabel('kW')
plt.legend()
   ```

Validation loss is also calculated, original data and the model output is shown on the same plot for comparation.

   ```
with torch.no_grad(): # Disables gradient computation, as we don't need to compute gradients during validation

    val_output = transformer(sourceTensor_test, targetTensor_test[:, :-1]) # Passes the validation source data and the validation target data through the transformer
    val_loss = criterion(val_output.contiguous().view(-1, 1000), targetTensor_test[:, 1:].contiguous().view(-1)) # Computes the loss between the model's predictions and the validation target data
    print(f"Validation Loss: {val_loss.item()}") # Prints the validation loss value
   ```
![screenshot](https://github.com/user-attachments/assets/19a242c4-5677-4db0-9342-19ebf84ae3f4)

Lastly, the outputs of the model with the original data and future prediction are processed on our trained transformer model and plotted, seperatively.

   ```
# Creating and filling array to hold the predictions
predictedDataSet = np.empty((64, 10),dtype=int) # Creating an array to hold the predictions
for i in range(val_output.size()[0]):
    for j in range(val_output.size()[1]):
        predictedDataSet[i][j] = torch.argmax(val_output[i][j])

# Since transformer operations exclude the first token in each sequence the last column is filled with zeros. To avoid discontinuity, we assign previous values to the zero column.
predictedDataSet[:,9] = predictedDataSet[:,8]

# Converting predicted array into 1-Dimensional
predictedDataSet = predictedDataSet.reshape(-1)

# We multiplied the original dataset by 100 to convert it into integers, now dividing by 100 to get original values
predictedDataSet = np.array(predictedDataSet, dtype=float) / 100
   ```

   ```
plt.rcParams["figure.figsize"] = [15, 5]
plt.title("Average Power")
plt.plot(y, dataSet, color="red")
plt.plot(y2, predictedDataSet, color="blue")
plt.legend(["Original", "Predicted (e="+str(ep)+")"], loc="lower right")
plt.xlabel("Hours from May 9, 2020 to Jun 11, 2020 (747 total)")
plt.ylabel("kW")
plt.gca().yaxis.label.set(rotation='horizontal', ha='right');
   ```
![Compare Predictions (e=15)](https://github.com/user-attachments/assets/974c1e7a-029f-44b0-a80d-302d76e85416)

   ```
y3 = np.arange(681,1001)
plt.figure()
plt.rcParams["figure.figsize"] = [15, 5]
plt.title("Average Power Future Predicts")
plt.plot(y3, futurePredict, color="red")
plt.xlabel("Hours after Jun 11, 2020 (320 total)")
plt.ylabel("kW")
plt.gca().yaxis.label.set(rotation='horizontal', ha='right');
plt.show()
   ```

![Future Predicts (e=15)](https://github.com/user-attachments/assets/87bfa34c-6271-42f8-a2cb-2d66cf3e1737)


