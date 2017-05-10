import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

#set up plotting
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#helper function to plot training loss, and train+val accuracy
def plot_results(results_dict):
    #results_dict has keys train_loss, train_acc, val_acc for lists of value vs iteration number
    loss_history = results_dict['train_loss']
    train_acc_history = results_dict['train_acc']
    val_acc_history = results_dict['val_acc']
    
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.plot([0.7] * len(val_acc_history), 'k--')
    plt.xlabel('iteration_num')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()
    
def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train(model, loss_fn, optimizer, num_epochs = 1):
    loss_history = [] #per iteration
    train_acc_history = []#per epoch
    val_acc_history = []#per epoch
    
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.data[0])
        train_acc_history.append(check_accuracy(model,loader_train))
        val_acc_history.append(check_accuracy(model,loader_val))
    results_dict = {
        'train_loss':loss_history,
        'train_acc':train_acc_history,
        'val_acc':val_acc_history
    }
    return results_dict
 
#calculate f2 score

#integrate accuracy check inside training func
