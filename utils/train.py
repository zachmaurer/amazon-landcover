import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gc
from scipy.special import expit
from sklearn.metrics import fbeta_score
import numpy as np



# Function: check_accuracy
# 
# Evaluates the model on a dataset
# Always takes a model in TRAIN and returns a model in TRAIN
# ----
# Args:
#   model: the model object
#   loader: DataLoader in pytorch
#  

def f2_score(model, config, loader, label=""):
    sum_f2 = 0.0 
    model.eval()
    num_samples = 0
    for x, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.numpy())
        # multiply by num examples to get sum, not average
        sum_f2 += fbeta_score(scores > 0.5, y.numpy(), beta=2, average='samples')*y.size(0)
        num_samples += y.numpy().shape[0]
    f2_score = float(sum_f2)/num_samples
    config.log('F2 score {%s} : Got %.2f' % (label, 100.0 * f2_score))
    model.train()
    return f2_score

def check_global_recall(model, config, loader, label = ""):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += (preds == y.numpy()).sum()
        num_samples += preds.shape[0]*17
    acc = float(num_correct) / num_samples
    config.log('Global recall {%s} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100.0 * acc))
    model.train()
    return acc

def check_all_or_none_accuracy(model, config, loader, label = ""):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += np.sum([1 for i in range(preds.shape[0]) if np.array_equal(preds[i], y[i])])
        num_samples += preds.shape[0]
    acc = float(num_correct) / num_samples
    config.log('All or none acc {%s} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
    model.train()
    return acc

def check_per_class_accuracy(model, config, loader, label = ""):
    num_correct = np.zeros((17,))
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += (preds == y.numpy()).sum(axis=0)
        num_samples += preds.shape[0]
    acc = num_correct / num_samples
    # TODO: Not printing this right now because it would print 17 scores at every step.
    #config.log('{} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
    model.train()
    return acc

# Function: train
# 
# Evaluates the model on a dataset
# Always takes a model in TRAIN and returns a model in TRAIN
# ----
# Args:
#   model: the model object
#   loader: DataLoader in pytorch
#  
def train(model, config, loss_fn = None, optimizer = None):
    if not loss_fn:
        loss_fn = nn.MultiLabelSoftMarginLoss().type(config.dtype) # TODO: should the loss function run on the CPU or GPU?
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr = config.lr) 

    loss_history = [] # per iteration
    train_f2_history = []
    val_f2_history = []
    train_all_or_none_acc_history = [] # per epoch
    val_all_or_none_acc_history = [] # per epoch
    train_per_class_acc_history = [] # TODO
    val_per_class_acc_history = []  # TODO
    train_global_recall_history = []
    val_global_recall_history = []

    
    model.train()
    for epoch in range(config.epochs):
        config.log('Starting epoch %d / %d' % (epoch + 1, config.epochs))
        for t, (x, y) in enumerate(config.train_loader):
            # Train
            x_var = Variable(x.type(config.dtype))
            y_var = Variable(y.type(config.dtype)) # removed .long() ?
            scores = model(x_var)            
            loss = loss_fn(scores, y_var)
            loss_history.append(loss.data[0])
            
            # Print Loss
            if config.print_every and (t + 1) % config.print_every == 0:
                config.log('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            # Evaluate on train and val sets
            if config.eval_every and (t + 1) % config.eval_every == 0:
                if config.train_loader:
                    train_f2_history.append(f2_score(model, config, config.train_loader, "train"))
                    train_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.train_loader, "train"))
                    train_global_recall_history.append(check_global_recall(model, config, config.train_loader, "train"))
                if config.val_loader:
                    val_f2_history.append(f2_score(model, config, config.val_loader, "val"))
                    val_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.val_loader, "val"))
                    val_global_recall_history.append(check_global_recall(model, config, config.val_loader, "val"))
                             
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()
        gc.collect()

    # Final Evaluation
    if config.train_loader:
        train_f2_history.append(f2_score(model, config, config.train_loader, "train"))
        train_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.train_loader, "train"))
        train_global_recall_history.append(check_global_recall(model, config, config.train_loader, "train"))
    if config.val_loader:
        val_f2_history.append(f2_score(model, config, config.val_loader, "val"))
        val_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.val_loader, "val"))
        val_global_recall_history.append(check_global_recall(model, config, config.val_loader, "val"))
    results_dict = {
        'train_loss': loss_history,
        'train_f2': train_f2_history,
        'train_all_or_none': train_all_or_none_acc_history,
        'train_global_recall': train_global_recall_history,
        'val_f2': val_f2_history, 
        'val_all_or_none': val_all_or_none_acc_history, 
        'val_global_recall': val_global_recall_history
    }
    return results_dict
