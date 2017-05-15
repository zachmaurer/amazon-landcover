import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Function: check_accuracy
# 
# Evaluates the model on a dataset
# Always takes a model in TRAIN and returns a model in TRAIN
# ----
# Args:
#   model: the model object
#   loader: DataLoader in pytorch
#  
def check_accuracy(model, config, loader, label = ""):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    config.log('{} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
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
    train_acc_history = [] # per epoch
    val_acc_history = [] # per epoch
    
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
                    train_acc_history.append(check_accuracy(model, config, config.train_loader, "train"))
                if config.val_loader:
                    val_acc_history.append(check_accuracy(model, config, config.val_loader, "val"))
         
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Final Evaluation
    if config.train_loader:
                    train_acc_history.append(check_accuracy(model, config, config.train_loader, "train"))
    if config.val_loader:
        val_acc_history.append(check_accuracy(model, config, config.val_loader, "val"))
    results_dict = {
        'train_loss': loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history
    }
    return results_dict
 
