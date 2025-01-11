This script is written in order to implement transformer deep learning model and test it by generating predictions of Electric Vehicle charging demands each hour.

The data is first read from json file, then charging load is calculated for each hour on the given time interval.

After the transformer model is implemented using PyTorch, our data is divided into three portions to train, validate and test the model, respectively.

The epoch (iteration number for training) is prompted, loss of training process for each epoch level is calculated and printed on the prompt screen.

Validation loss is also calculated, original data and the model output is shown on the same plot for comparation.

Lastly, future prediction is made on our trained transformer model and plotted.
