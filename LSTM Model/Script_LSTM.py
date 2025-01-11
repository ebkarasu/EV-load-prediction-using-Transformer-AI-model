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

# A Sample class to hold data
class chargeInfo:

    # init method or constructor
    def __init__(self,id="",clusterID="",connectionTime="",disconnectTime="",doneChargingTime="",kWhDelivered=0.0,sessionID="",siteID="",spaceID="",stationID="",timezone="",userID="",userInputs=""):
         self.id=id
         self.clusterID=clusterID
         self.connectionTime=connectionTime
         self.disconnectTime=disconnectTime
         self.doneChargingTime=doneChargingTime
         self.kWhDelivered=kWhDelivered
         self.sessionID=sessionID
         self.siteID=siteID
         self.spaceID=spaceID
         self.stationID=stationID
         self.timezone=timezone
         self.userID=userID
         self.userInputs=userInputs

    # Method to print an object
    def print(self):
         print(self.id + '\n' + self.clusterID + '\n' + self.connectionTime + '\n' + self.disconnectTime + '\n' + self.doneChargingTime + '\n' + str(self.kWhDelivered) + '\n' + self.sessionID + '\n' + self.siteID + '\n' + self.spaceID + '\n' + self.stationID + '\n' + self.timezone + '\n' + self.userID + '\n' + self.userInputs)

# Reading the ACN data needed
f = open('acn-data.json')

dataList = json.load(f)

# Closing the file
f.close()

# Initializing the object array
classArray = [chargeInfo() for i in range(len(dataList))]

# Pushing the data to the object array
for i in range(len(dataList)):
     classArray[i].id = str(dataList[i].get('id'))
     classArray[i].clusterID = str(dataList[i].get('clusterid'))
     classArray[i].connectionTime = str(dataList[i].get('connectiontime'))
     classArray[i].disconnectTime = str(dataList[i].get('disconnecttime'))
     classArray[i].doneChargingTime = str(dataList[i].get('donechargingtime'))
     classArray[i].kWhDelivered = float(dataList[i].get('kwhdelivered'))
     classArray[i].sessionID = str(dataList[i].get('sessionid'))
     classArray[i].siteID = str(dataList[i].get('siteid'))
     classArray[i].spaceID = str(dataList[i].get('spaceid'))
     classArray[i].stationID = str(dataList[i].get('stationid'))
     classArray[i].timezone = str(dataList[i].get('timezone'))
     classArray[i].userID = str(dataList[i].get('userid'))
     classArray[i].userInputs = str(dataList[i].get('userinputs'))

# Array to hold all charge connecting and disconnecting times
connectTimes = [datetime.datetime(2000,1,1) for i in range(len(dataList))]
disconnectTimes = [datetime.datetime(2000,1,1) for i in range(len(dataList))]

monthToNumber = {
    "Jan":1,
    "Feb":2,
    "Mar":3,
    "Apr":4,
    "May":5,
    "Jun":6,
    "Jul":7,
    "Aug":8,
    "Sep":9,
    "Oct":10,
    "Nov":11,
    "Dec":12
}

# Assigning values to the connectTimes and disconnectTimes array
for i in range(len(classArray)):
    connectTimes[i] = datetime.datetime(int(classArray[i].connectionTime[12:16]),monthToNumber.get(classArray[i].connectionTime[8:11]),int(classArray[i].connectionTime[5:7]),int(classArray[i].connectionTime[17:19]),int(classArray[i].connectionTime[20:22]))
    disconnectTimes[i] = datetime.datetime(int(classArray[i].disconnectTime[12:16]),monthToNumber.get(classArray[i].disconnectTime[8:11]),int(classArray[i].disconnectTime[5:7]),int(classArray[i].disconnectTime[17:19]),int(classArray[i].disconnectTime[20:22]))

# Finding first starting point of the first charge connection in time
minTime = datetime.datetime(min(connectTimes).year,min(connectTimes).month,min(connectTimes).day,min(connectTimes).hour)

# Creating our dataset
dataSet = np.array([], dtype=float)

flag = 0
dummy = 0

# Our dataset is going to show average power usage each hour of the entire time interval in the original data
while(minTime < max(disconnectTimes)):
    for i in range(len(classArray)):
        if connectTimes[i] < minTime < disconnectTimes[i]:
            flag = 1
            dummy = i
    if flag:
        dataSet = np.append(dataSet, classArray[dummy].kWhDelivered / ((disconnectTimes[dummy] - connectTimes[dummy]).seconds / 3600.0))
    else:
        dataSet = np.append(dataSet, 0.0)
    flag = 0
    minTime = minTime + datetime.timedelta(minutes=60)

# Converting our dataset to integers to be given transformer as an input

dataSet = dataSet.round(decimals=2)

dataMatrix = np.empty([743,5])

for i in range(len(dataSet)-4):
    dataMatrix[i][0] = dataSet[i + 4]
    dataMatrix[i][1] = dataSet[i + 3]
    dataMatrix[i][2] = dataSet[i + 2]
    dataMatrix[i][3] = dataSet[i + 1]
    dataMatrix[i][4] = dataSet[i + 0]

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

class TimeSeriesDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train,y_train)
test_dataset = TimeSeriesDataset(X_test,y_test)

batch_size = 16

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    break

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

model = LSTM(1, 4, 1)
model.to(device)

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
    print()

learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

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

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], 4+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dummies[:, 0]

dummies = np.zeros((X_test.shape[0], 4+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dummies[:, 0]

plt.figure()
plt.title("Test data prediction")
plt.plot(new_y_test, label='Original')
plt.plot(test_predictions, label='Predictions')
plt.xlabel('Hours')
plt.ylabel('kW')
plt.legend()
plt.show()
