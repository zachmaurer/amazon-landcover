import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import numpy as np

# Function: check_accuracy
# ----
# Args:
#   model: the model object
#   loader: DataLoader in pytorch
#  
# TODO: fix config references, such that config.loader_train or whatever is used.
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


# TODO: Integrate commented code into train model
# def train(model, loss_fn, optimizer, num_epochs = 1, dtype = gpu_dtype, eval_every = 500, overfit = None):
#     for epoch in range(num_epochs):
#         print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
#         model.train()
#         for t, (x, y) in enumerate(loader_train):
#             x_var = Variable(x.type(dtype))
#             y_var = Variable(y.type(dtype).long())

#             scores = model(x_var)
            
#             loss = loss_fn(scores, y_var)
#             #  loss.data[0] is the loss 
            
#             if (t + 1) % print_every == 0:
#                 print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             if (t + 1) % eval_every == 0:
#                 check_accuracy(model, loader_train, dtype = dtype, overfit = overfit, msg = "Train")
#                 check_accuracy(model, loader_val, dtype = dtype, overfit = overfit, msg = "Val")
#                 model.train()
            
#             if (overfit and t > overfit): 
#                 break

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
 
