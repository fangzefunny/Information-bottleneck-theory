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
#        Data generation
#--------------------------------

# a function to an int to the input of the NN
to_x = lambda x: list(map(int,list(f'{int(bin(x)[2:]):010d}')))

def get_data( n_sample, groups, rng=None):
    '''Generate artificial data

    We need a experiemnt material to examine the data
    Because I did not understand the example used in
    Tishby's paper. I use the example in:
    https://github.com/stevenliuyi/information-bottleneck

    This is simple example, the input is a binary 
    representation of number ranged in [0, 1023] in total
    1024 possible input states. The states are divided into
    16 groups, decided by their mod to 16. Each group output
    a binary number either 0 or 1. 
    '''
    ## Decide the random generator 
    rng = rng if rng is not None else np.random.RandomState(2022)

    ## Get the input data X
    x_int = rng.choice( 1023, size=n_sample)
    x     = np.vstack( [ to_x(i) for i in x_int])
    y0    = np.array([ groups[i%16] for i in x_int])
    y1    = 1 - y0
    y     = (np.vstack( [ y0, y1])).T

    return x, y

def get_train_test( n_train, n_test, seed=2022):
    
    ## Decide the random generator 
    rng = np.random.RandomState( seed)

    ## decide the group assignment
    groups = rng.permutation( [0]*8+[1]*8)
    x_train, y_train = get_data( n_train, groups, rng)
    x_test , y_test  = get_data( n_test, groups, rng)

    return x_train, y_train, x_test, y_test  

#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'LR': .1,
                'MaxEpochs': 2000,
                'Verbose': True,
                'BatchSize': 32, 
                } 
eps_ = 1e-14

#---------------------------------
#    NeuralNetwork architecture
#---------------------------------

class NeuralNet(nn.Module):

    def __init__( self, dims, gpu=True):
        super().__init__()
        '''Network in the paper

        The model that is simulated in the paper.
        We train the model using cpu, because the network
        scale is small. 

        Activation: tanh, sigmoid(last layer)
        '''
        # assign device
        if gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        layers = []
        for i in range(len(dims)-1):
            layers.append( nn.Linear( dims[i], dims[i+1]))
            if i == len(dims)-2:
                layers.append( nn.Sigmoid())
            else: 
                layers.append( nn.Tanh())
        self.layers = nn.Sequential( *layers)
        self.to(self.device)

    def forward( self, x):
        return self.layers(x)

#---------------------------------
#        Network Training 
#---------------------------------

def ACC( y_hat, y):
    return (y_hat.argmax(dim=1) == y.argmax(dim=1)
                ).float().mean().data.numpy()

def trainNN( train_data, test_data, model, **kwargs):
    '''Train a neural network 

        Architecture: fully connected 10-7-5-4-3-2
        Activation: tanh, sigmoid(last layer)
        Optimizer: SGD
        Loss fn: cross entropy
    '''
    ## Hyperparameter
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]

    ## Preprocessing 
    # preprocess the train data
    x_train, y_train = train_data
    n_batch = int( len(x_train))
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    _dataset = torch.utils.data.TensorDataset(x_train, y_train)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    # preprocess the test data 
    x_test, y_test = test_data
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)

    ## Start training 
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
        loss_, train_acc = 0, 0        
        for _, (x_batch, y_batch) in enumerate(_dataloader):

            # foward the model 
            y_hat =  model.forward( x_batch.to(model.device))
            # calculate the loss 
            loss = BCE( y_hat, y_batch)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store the te losses for plotting 
            loss_ += loss.data.numpy() / n_batch
            train_acc += ACC( y_hat, y_batch.to(model.device)) / n_batch

        # track training
        losses.append(loss_)

        # Tracking the training condition 
        if (epoch%100 ==0) and HyperParams['Verbose']:
            y_test_hat =  model.forward( x_test.to(model.device))
            test_acc   = ACC( y_test_hat, y_test.to(model.device))
            print( f'Epoch:{epoch}: Train acc:{train_acc:.4f}-Test acc:{test_acc:.4f}')
            #torch.save( model.state_dict(), f'{path}/checkpts/model-{epoch}.pkl')

if __name__ == '__main__':

    ## GET DATA
    n_train, n_test = 50000, 10000 
    data = get_train_test( n_train, n_test, seed=2022)
    x_train, y_train, x_test, y_test = data

    ## train model
    dims = [ 10, 7, 5, 4, 3, 2]
    model = NeuralNet( dims, gpu=True)
    trainNN( (x_train, y_train), (x_test, y_test), model, 
                LR=.1)
    