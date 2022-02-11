import os
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
from torch.optim import SGD
from collections import Counter
from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns 

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')
if not os.path.exists(f'{path}/data'):
    os.mkdir(f'{path}/data')

# A dictionary to hold hidden layers
hid_layers = {}

#--------------------------------
#      Visualization Setting
#--------------------------------

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
dpi = 250
fontsize = 16

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

    return x, y, x_int

def get_train_test( n_train, n_test, seed=2022):
    
    ## Decide the random generator 
    rng = np.random.RandomState( seed)

    ## decide the group assignment
    groups = rng.permutation( [0]*8+[1]*8)
    x_train, y_train, x_int = get_data( n_train, groups, rng)
    x_test , y_test, _  = get_data( n_test, groups, rng)

    return x_train, y_train, x_test, y_test, x_int   

#--------------------------------
#        Hyperparameters
#--------------------------------

DefaultParams = { 
                'LR': .001,
                'MaxEpochs': 250,
                'Verbose': True,
                'BatchSize': 2048, 
                'InfoPlane': True,
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

#---------------------------------------
#       Mutual information estimation
#---------------------------------------

def cal_MI( hidden, n_bins, data):
    '''Estimate the mutual information

        Bin the output activations,
        so that the hidden layer random variables
        can be a discrete categorical distribution.
        Then we
    '''
    hidden = hidden.cpu().numpy()
    x_train, y_train, x_int = data
    n_train = x_train.shape[0] 

    # discreteization 
    bins = np.linspace( -1, 1, n_bins+1)
    ind  = np.digitize( hidden, bins)

    # get the probability of each distribution 
    P_x = Counter(); P_y = Counter(); P_t = Counter()
    P_xt = Counter(); P_ty = Counter()
    for i in range(n_train):
        P_x[x_int[i]] += 1./n_train
        P_y[y_train[i,0]] += 1./n_train      
        P_xt[(x_int[i],)+tuple(ind[i,:])] += 1./n_train
        P_ty[(y_train[i,0],)+tuple(ind[i,:])] += 1./n_train
        P_t[tuple(ind[i,:])] += 1./n_train

    # calcuate encoder mutual information I(X;T)
    # I(X;T) = ∑_i p(xi,ti) log [p(xi,ti) / (p(x)p(t))]
    I_XT = 0
    for i in P_xt:
        # P(xi,ti), P(xi) and P(ti)
        p_xt = P_xt[i]; p_x = P_x[i[0]]; p_t = P_t[i[1:]]
        # I(xi;ti)
        I_XT += p_xt * np.log( p_xt / (p_x*p_t))

    I_TY = 0
    for i in P_ty:
        # P(ti,yi), P(ti) and P(yi)
        p_ty = P_ty[i]; p_y = P_y[i[0]]; p_t = P_t[i[1:]]
        # I(xi;ti)
        I_TY += p_ty * np.log( p_ty / (p_t*p_y))
    
    return I_XT, I_TY

def get_MI( dims, data):
    layers = [ hid_layers[f'l{i*2}'] for i in range(len(dims)-2)]
    I_XT_lst, I_TY_lst = [], []
    for layer in layers:
        I_xt, I_ty = cal_MI(layer, 30, data)
        I_XT_lst.append(I_xt)
        I_TY_lst.append(I_ty)
    return I_XT_lst, I_TY_lst

#---------------------------------
#        Network Training 
#---------------------------------

def ACC( y_hat, y):
    return (y_hat.argmax(dim=1) == y.argmax(dim=1)
                ).float().mean().detach().cpu().numpy()

def get_activation(name):
    def hook(model, input, output):
        hid_layers[name] = output.detach()
    return hook

def reg_hook( model, dims):
    for i in range(len(dims)-2):
        model.layers[i*2].register_forward_hook(
            get_activation(f'l{i*2}'))

def trainNN( train_data, test_data, x_int, model, dims, **kwargs):
    '''Train a neural network 

        Architecture: fully connected 10-7-5-4-3-2
        Activation: tanh, sigmoid(last layer)
        Optimizer: SGD
        Loss fn: cross entropy

    Inputs:
        train_data: train data
        test_data: test data
        x_int: train data int 
        model: the NN model 
        dims: dims for the mdoel 
    
    Args:
        LR: .001,
        MaxEpochs: 250,
        Verbose: True,
        BatchSize: 2048, 
        InfoPlane: True,
    '''
    ## Hyperparameter
    HyperParams = DefaultParams
    for key in kwargs.keys():
        HyperParams[key] = kwargs[key]

    ## Preprocessing 
    # preprocess the train data
    x_train_np, y_train_np = train_data
    n_batch = int( len(x_train_np) / HyperParams['BatchSize'])
    x_train = torch.FloatTensor(x_train_np)
    y_train = torch.FloatTensor(y_train_np)
    _dataset = torch.utils.data.TensorDataset(x_train, y_train)
    _dataloader = torch.utils.data.DataLoader( _dataset, 
                batch_size=HyperParams['BatchSize'], drop_last=True)
    x_train = x_train.to(model.device)
    # preprocess the test data 
    x_test, y_test = test_data
    x_test = torch.FloatTensor(x_test).to(model.device)
    y_test = torch.FloatTensor(y_test).to(model.device)

    ## Start training 
    # decide optimizer 
    optimizer = SGD( model.parameters(), lr=HyperParams['LR'])   
    # init loss 
    BCE = nn.BCELoss()    
    ## get batch_size
    losses = []
    if HyperParams['InfoPlane']: I_XTs, I_TYs, epochs = [], [], []
    # start training
    model.train()
    for epoch in tqdm(range( HyperParams['MaxEpochs'])):

        ## train each batch 
        loss_, train_acc = 0, 0        
        for _, (x_batch, y_batch) in enumerate(_dataloader):
            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)
            # foward the model 
            y_hat =  model.forward( x_batch)
            # calculate the loss 
            loss = BCE( y_hat, y_batch)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store the te losses for plotting 
            loss_ += loss.detach().cpu().numpy() / n_batch
            train_acc += ACC( y_hat, y_batch) / n_batch

        # track training
        losses.append(loss_)
        if HyperParams['InfoPlane']:
            _ =  model.forward( x_train)
            I_xt, I_ty = get_MI( dims, (x_train_np, y_train_np, x_int))
            I_XTs.append( I_xt)
            I_TYs.append( I_ty)
            epochs.append( epoch)

        # Tracking the training condition 
        if (epoch%50 ==0) and HyperParams['Verbose']:
            y_test_hat =  model.forward( x_test)
            test_acc   = ACC( y_test_hat, y_test)
            print( f'Epoch:{epoch}: Train acc:{train_acc:.4f} Test acc:{test_acc:.4f}')
            #torch.save( model.state_dict(), f'{path}/checkpts/model-{epoch}.pkl')
        
    if HyperParams['InfoPlane']:
        layers = [ d+1 for d in range(len(dims)-2)]
        fla_epochs = np.repeat( epochs, len(layers))
        fla_I_XY   = np.hstack(I_XTs)
        fla_I_TY   = np.hstack(I_TYs)
        fla_layers = layers*250        
        outcome = { 'I_XT': fla_I_XY,
                    'I_TY': fla_I_TY,
                    'Layer': fla_layers,
                    'Epochs': fla_epochs}
        return pd.DataFrame( outcome)

#------------------------------
#      Information plane 
#------------------------------

def info_plane(n):

    # Average over data
    I_XTs, I_TYs = 0, 0
    for i in range(n):
        data = pd.read_csv( f'{path}/data/Mutual_info-s{i}.csv')
        I_XTs += data['I_XT'] / n
        I_TYs += data['I_TY'] / n
    avg_data = { 'I_XT': I_XTs,
                 'I_TY': I_TYs,
                 'layer': data['Layer'],
                 'Epochs': data['Epochs']}
    layers = avg_data['Layer'].unique()

    nr = nc = 2
    fig, axs = plt.subplots( nr, nc, figsize=(nr*4,nc*4))
    for i, layer in enumerate(layers):
        ax = axs[ i//nr, i%nr]
        subdata = avg_data[ avg_data['Layer']==layer]
        ifleg = True if i==0 else False
        sns.scatterplot( x='I_XT', y='I_TY', data=subdata, 
                         hue='Epochs', sizes=30, 
                         legend=ifleg, ax=ax)
        ax.set_xlabel('I(X;T)', fontsize=fontsize)
        ax.set_xlim( [ 0, 7])
        ax.set_ylabel('I(T;Y)', fontsize=fontsize)
        ax.set_ylim( [ 0,.7])
        ax.set_title(f'Layer {layer}', fontsize=fontsize)
    fig.tight_layout()
    plt.savefig( f'{path}/figures/Info_plane.png')

if __name__ == '__main__':

    ## Train or not?
    train = True
    n = 20

    if train:
        dims = [ 10, 7, 5, 4, 3, 2]
        seed = 20412
        for i in range(20):
            seed +=1
            ## Get Data
            n_train, n_test = 50000, 10000 
            data = get_train_test( n_train, n_test, seed=seed)
            x_train, y_train, x_test, y_test, x_int = data

            ## Train model 
            torch.manual_seed(seed)
            model = NeuralNet( dims, gpu=True)
            reg_hook( model, dims)
            infop=trainNN( (x_train, y_train), (x_test, y_test), x_int, 
                        model, dims, 
                        LR=.1, InfoPlane=True
                    )
            infop.to_csv( f'{path}/data/Mutual_info-s{i}.csv')
    
    ## Visualize 
    info_plane(n=20)