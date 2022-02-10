import os
import numpy as np 
import torch
import torch.nn as nn 
from torch.optim import SGD

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/checkpts'):
    os.mkdir(f'{path}/checkpts')

#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'Dims': [ 12, 10, 7, 5, 4, 3, 2],
                'LR': 1e-3,
                'MaxEpochs': 10000,
                'Verbose': True,
                'BatchSize': 32, 
                } 
eps_ = 1e-8

#---------------------------------
#    NeuralNetwork architecture
#---------------------------------

class NeuralNet(nn.Module):

    def __init__( self, dims):
        super().__init__()
        '''Network in the paper

        The model that is simulated in the paper.
        We train the model using cpu, because the network
        scale is small. 

        Activation: tanh, sigmoid(last layer)
        '''
        layers = []
        for i in range(len(dims)-1):
            layers.append( nn.Linear( dims[i], dims[i+1]))
            if i == len(dims)-2:
                layers.append( nn.Sigmoid())
            else: 
                layers.append( nn.Tanh())
        self.layers = nn.Sequential( *layers)

    def forward( self, x):
        return self.layers(x)


def trainNN( train_data, **kwargs):
    '''Train a neural network 

        Architecture: fully connected 12-10-7-5-4-3-2
        Activation: tanh, sigmoid(last layer)
        Optimizer: SGD
        Loss fn: cross entropy
    '''
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]
    # preprocess the data
    x, y = train_data
    n_batch = int( len(x))
    x_tensor = x.type( torch.FloatTensor)
    y_tensor = y.type( torch.FloatTensor)
    _dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    # init model
    model = NeuralNet( HyperParams['Dims'])
    # decide optimizer 
    optimizer = SGD( model.parameters(), lr=1e-2)   
    # init loss 
    BCE = nn.BCELoss()    
    ## get batch_size
    losses = []
    # start training
    model.train()
    for epoch in range( HyperParams['MaxEpochs']):

        ## train each batch 
        loss_ = 0        
        for _, (x_batch, y_batch) in enumerate(_dataloader):

            # reshape the image
            x = torch.FloatTensor( x_batch).view( 
                x_batch.shape[0], -1)
            # reconstruct x
            y_hat =  model.forward( x)
            # calculate the loss 
            loss = BCE( y_hat, y_batch)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store the te losses for plotting 
            loss_ += loss.data.numpy() / n_batch

        # track training
        losses.append(loss_)

        # save the model 
        if (epoch%100 ==0) and HyperParams['Verbose']:
            print( f'Epoch:{epoch}, Loss:{loss_}')
            torch.save( model.state_dict(), f'{path}/checkpts/model-{epoch}.pkl')

if __name__ == '__main__':

    model = NeuralNet()

    print(model)