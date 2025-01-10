#Importing used libraries
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import json
import timedelta

# This is a class for making Multi Head Attention processes and defined as a subclass of PyTorch's nn.Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, query, and value

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

# This is a class for making Feed Forwarding processes
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# This is a class for making Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Encoder in transformer model encapsulates a multi-head self-attention mechanism followed by position-wise feed-forward neural network, with residual connections, layer normalization, and dropout applied
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Decoder part consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism, a position-wise feed-forward neural network, and the corresponding residual connections, layer normalization and dropout layers.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Attaining the transformer model by combining decoder, encoder and all required parameters.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

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
    disconnectTimes[i] = datetime.datetime(int(classArray[i].disconnectTime[12:16]),
                                        monthToNumber.get(classArray[i].disconnectTime[8:11]),
                                        int(classArray[i].disconnectTime[5:7]),
                                        int(classArray[i].disconnectTime[17:19]),
                                        int(classArray[i].disconnectTime[20:22]))

# Finding first starting point of the first charge connection in time
minTime = datetime.datetime(min(connectTimes).year,min(connectTimes).month,min(connectTimes).day,min(connectTimes).hour)

# Creating our dataset
dataSet = np.array([], dtype=float)

flag = 0
dummy = 0

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

dataSet = dataSet.round(decimals=3)

print(dataSet,'\n',len(dataSet))

# Determining specifications to be given into the transformer as an input
embeddingDim = 512
noHeads = 8
noEncLayers = 2
noDecLayers = 1
FFND = 2048 # Feeding forward network dimension
dropout = 0.05
batchSize = 32










