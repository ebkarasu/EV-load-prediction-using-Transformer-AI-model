This script is written in order to implement transformer deep learning model and test it by generating predictions of Electric Vehicle charging demands each hour.

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
import timedelta
import matplotlib.pyplot as plt
from sympy.stats.sampling.sample_numpy import numpy
```

# Implementing Transformer

**1. Multi-head Attention**
   
   Multi-Head Attention mechanism computes the attention between each pair of positions in a sequence. It consists of multiple “attention heads” that capture different aspects of the input sequence.
   
   ```
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
   ```

**2. Feed forward Network**

   FeedForward class defines a position-wise feed-forward neural network that consists of two linear layers with a ReLU activation function in between. This feed-forward network is applied to each position separately and identically.

   ```
   class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
   ```
   
**3. Encoder Block**

   Encoder defines a single layer of the transformer's encoder. It encapsulates a multi-head self-attention mechanism followed by position-wise feed-forward neural network. Typically, multiple such encoder layers are stacked to form the complete encoder part of a          transformer model. 
    
   ```
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
   ```

**4. Decoder Block**

   Decoder class defines a single layer of the transformer's decoder. It consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism, a position-wise feed-forward neural network and the corresponding residual connections, layer                normalization, and dropout layers. As with the encoder, multiple decoder layers are typically stacked to form the complete decoder part of a transformer model.

   ```
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
   ```  

**5. Transformer**

   The Transformer class brings together the various components of a Transformer model, including the embeddings, positional encoding, encoder layers, and decoder layers.
  
   ```
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

After the transformer model is implemented using PyTorch, the 64 batch-sized samples is taken from our data are divided into three portions to train, validate and test the model, respectively.

   ```
for i in range(64):
    targetData[i] = dataSetInt[10*i : 50 + 10*i]

# Filling source array with samples batchSize = 64
c = np.arange(1,len(dataSet))
for i in range(64):
    sourceData[i] = c[10*i : 50 + 10*i]

# Dividing the target data into portions training, validating and testing
targetData_training = targetData[:,0:35]
targetData_validation = targetData[:,35:40]
targetData_test = targetData[:,40:50]
   ```

The parameters for transformer are defined. 

   ```
embeddingDim = 512 # Embedding dimension
noHeads = 8 # Number of heads
noEncLayers = 6 # Number of encoding layers
noDecLayers = 6 # Number of decoding layers
FFND = 2048 # Feeding forward network dimension
dropout = 0.05

   ```

The epoch (iteration number for training) is prompted, loss of training process for each epoch level is calculated and printed on the prompt screen.

   ```
print('Enter training epoches:', end=' \t')
ep = int(input())

# Iterating over 5 training epochs
for epoch in range(ep):
    optimizer.zero_grad() # Clears the gradients from the previous iteration
    output = transformer(sourceTensor_training, targetTensor_training[:, :-1]) # Passes the source data and the target data through the transformer
    loss = criterion(output.contiguous().view(-1, 1000), targetTensor_training[:, 1:].contiguous().view(-1)) # Computes the loss between the model's predictions and the target data.
    loss.backward() # Computes the gradients of the loss with respect to the model's parameters
    optimizer.step() # Updates the model's parameters using the computed gradients
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}") # Prints the current epoch number and the loss value for that epoch
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

![errors](https://github.com/user-attachments/assets/381bb89f-96c4-4f48-8172-09e9e16fa19f)


