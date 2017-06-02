import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import gc
from scipy.special import expit
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import fbeta_score
import numpy as np
from .model import loadModel, countParams, checkpointModel
from utils.constants import LABEL_LIST
import pandas as pd
import os


#TBD - feed in a single tensor into the get_label_strings_from_tensor, rather than doing it per batch
def predict(model, config, loader, dataset = ""):
    config.log("Predicting on {}".format(dataset))
    model.eval()
    print_every = 5
    
    num_examples = 0
    
    if dataset is 'train':
        num_examples = config.num_train
    elif dataset is 'val':
        num_examples = config.num_val
    elif dataset is 'test':
        num_examples = loader.dataset.num_examples
    
    preds_var = Variable(torch.FloatTensor(num_examples,17).type(config.dtype), volatile=True)
    subm = None
    image_list = []
    
    if dataset is not "test":
        subm = pd.DataFrame(columns=('image_name', 'tags', 'labels'))
        labels_var = Variable(torch.FloatTensor(num_examples,17).type(config.dtype), volatile=True)
        for t, (x, image_names, y) in enumerate(loader):
            if t%print_every == 0:
                print(t)
            x_var = Variable(x.type(config.dtype), volatile=True)
            scores = model(x_var)
            preds_var.data[t*loader.batch_size:t*loader.batch_size+x.size()[0]] = scores.data #tbd - verify this is good
            labels_var.data[t*loader.batch_size:t*loader.batch_size+x.size()[0]] = y
            image_list.extend(image_names)
        #hacky
        labels_var[labels_var>.5] = 1
        labels_var[labels_var<.5] = 0
        labels = get_label_strings_from_tensor(labels_var.data)
        
        subm['labels'] = labels
        
    else: #dataset is 'test'..
        subm = pd.DataFrame(columns=('image_name', 'tags'))
        for t, (x, image_names, _) in enumerate(loader):
            if t%print_every == 0:
                print(t)
            x_var = Variable(x.type(config.dtype), volatile=True)
            scores = model(x_var)
            preds_var.data[t*loader.batch_size:t*loader.batch_size+x.size()[0]] = scores.data #tbd - verify this is good
            image_list.extend(image_names)
            
    preds_var = nn.functional.sigmoid(preds_var)
    preds_var[preds_var>0.5] = 1
    preds_var[preds_var<=0.5] = 0
    preds = get_label_strings_from_tensor(preds_var.data)
    
    subm['image_name'] = image_list
    subm['tags'] = preds
    submission_name = os.path.join(config.log_dest, "submission_tt_v3_{}.csv".format(dataset))
    subm.to_csv(submission_name, index=False)
    config.log("Done. Made csv: {}".format(submission_name))

def get_label_strings_from_tensor(pred_labels_tensor):
    mlb = MultiLabelBinarizer(classes = LABEL_LIST)
    mlb = mlb.fit(None) #what hte fuck
    pred_labels_cpu = pred_labels_tensor.cpu().numpy()
    pred_labels_str = mlb.inverse_transform(pred_labels_cpu)
    pred_labels = [" ".join(pred_labels_str[i]) for i in range(pred_labels_cpu.shape[0])]
    return pred_labels

# Function: check_accuracy
# 
# Evaluates the model on a dataset
# Always takes a model in TRAIN and returns a model in TRAIN
# ----
# Args:
#   model: the model object
#   loader: DataLoader in pytorch
#  
def eval_performance(model, config, loader, f2 = True, recall = True, acc = True, label = ""):
    sum_f2 = 0.0
    num_samples_f2 = 0
    num_correct_recall = 0
    num_samples_recall = 0
    num_correct_acc = 0
    num_samples_acc = 0

    model.eval()
    for x, _, y in loader:
        y = y.type(torch.cuda.ByteTensor) if config.use_gpu else y.type(torch.ByteTensor)
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        #scores = expit(scores.data.cpu().numpy())
        scores = nn.functional.sigmoid(scores)
        preds = scores > 0.5
        if f2:
            sum_f2 += fbeta_score(preds.data.cpu().numpy(), y.cpu().numpy(), beta=2, average='samples')*y.size(0)
            num_samples_f2 += y.size(0)
        if recall:
            num_correct_recall += (preds.data == y).sum()
            num_samples_recall += preds.size(0)*17
        if acc:
            #num_correct_acc += np.sum([1 for i in range(preds.size(0)) if np.array_equal(preds[i].data.cpu().numpy(), y.cpu().numpy()[i])])
            num_correct_acc += (((preds.data == y).sum(1)) == 17).sum()
            num_samples_acc += preds.size(0)
    if f2:
        f2_score = float(sum_f2)/num_samples_f2
        config.log('F2 score {%s} : Got %.2f' % (label, 100.0 * f2_score))
    if recall:
        recall = float(num_correct_recall) / num_samples_recall
        config.log('Global recall {%s} : Got %d / %d correct (%.2f)' % (label, num_correct_recall, num_samples_recall, 100.0 * recall))
    if acc:
        acc = float(num_correct_acc) / num_samples_acc
        config.log('All or none acc {%s} : Got %d / %d correct (%.2f)' % (label, num_correct_acc, num_samples_acc, 100 * acc))
    model.train()
    return f2_score, recall, acc


def f2_score(model, config, loader, label=""):
    sum_f2 = 0.0 
    model.eval()
    num_samples = 0
    for x, _, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.cpu().numpy())
        # multiply by num examples to get sum, not average
        sum_f2 += fbeta_score(scores > 0.5, y.cpu().numpy(), beta=2, average='samples')*y.size(0)
        num_samples += y.cpu().numpy().shape[0]
    f2_score = float(sum_f2)/num_samples
    config.log('F2 score {%s} : Got %.2f' % (label, 100.0 * f2_score))
    model.train()
    return f2_score

def check_global_recall(model, config, loader, label = ""):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, _, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.cpu().numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += (preds == y.cpu().numpy()).sum()
        num_samples += preds.shape[0]*17
    acc = float(num_correct) / num_samples
    config.log('Global recall {%s} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100.0 * acc))
    model.train()
    return acc

def check_all_or_none_accuracy(model, config, loader, label = ""):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, _, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.cpu().numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += np.sum([1 for i in range(preds.shape[0]) if np.array_equal(preds[i], y.cpu().numpy()[i])])
        num_samples += preds.shape[0]
    acc = float(num_correct) / num_samples
    config.log('All or none acc {%s} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
    model.train()
    return acc

def check_per_class_accuracy(model, config, loader, label = ""):
    num_correct = np.zeros((17,))
    num_samples = 0
    model.eval()
    for x, _, y in loader:
        x_var = Variable(x.type(config.dtype), volatile=True)
        scores = model(x_var)
        scores = expit(scores.data.cpu().numpy())
        # sigmoid 

        preds = scores > 0.5
        num_correct += (preds == y.cpu().numpy()).sum(axis=0)
        num_samples += preds.shape[0]
    acc = num_correct / num_samples
    # TODO: Not printing this right now because it would print 17 scores at every step.
    #config.log('{} : Got %d / %d correct (%.2f)' % (label, num_correct, num_samples, 100 * acc))
    model.train()
    return acc

###############
###############
###############
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

    best_f2 = 0.0
    loss_history = [] # per iteration
    train_f2_history = []
    val_f2_history = []
    train_all_or_none_acc_history = [] # per epoch
    val_all_or_none_acc_history = [] # per epoch
    # train_per_class_acc_history = [] # TODO
    # val_per_class_acc_history = []  # TODO
    train_global_recall_history = []
    val_global_recall_history = []

    if config.checkpoint:
        loadModel(model, config, optimizer)
        if config.predict:
            config.log("Skipping training. Predicting only.")
            return None
        

    countParams(model, config)

    model.train()
    for epoch in range(config.epochs):
        config.log('\nStarting epoch %d / %d' % (epoch + 1, config.epochs))
        for t, (x, _, y) in enumerate(config.train_loader):
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
                    f2_score(model, config, config.train_loader, "train")
                    check_all_or_none_accuracy(model, config, config.train_loader, "train")
                    check_global_recall(model, config, config.train_loader, "train")
                if config.val_loader:
                    f2_score(model, config, config.val_loader, "val")
                    check_all_or_none_accuracy(model, config, config.val_loader, "val")
                    check_global_recall(model, config, config.val_loader, "val")
                             
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gc.collect()

        config.log("Finished Epoch {}/{}".format(epoch + 1, config.epochs))
        config.log("Evaluating...")
        if config.train_loader:
            f2, recall, acc = eval_performance(model, config, config.train_loader, label = "train")
            train_f2_history.append(f2)
            train_global_recall_history.append(recall)
            train_all_or_none_acc_history.append(acc)
            # train_f2_history.append(f2_score(model, config, config.train_loader, "train"))
            # train_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.train_loader, "train"))
            # train_global_recall_history.append(check_global_recall(model, config, config.train_loader, "train"))
        if config.val_loader:
            f2, recall, acc = eval_performance(model, config, config.val_loader, label = "val")
            val_f2_history.append(f2)
            val_global_recall_history.append(recall)
            val_all_or_none_acc_history.append(acc)
            # val_f2_history.append(f2_score(model, config, config.val_loader, "val"))
            # val_all_or_none_acc_history.append(check_all_or_none_accuracy(model, config, config.val_loader, "val"))
            # val_global_recall_history.append(check_global_recall(model, config, config.val_loader, "val"))

        is_best = False
        if float(val_f2_history[-1]) > float(best_f2):
            best_f2 = val_f2_history[-1]
            is_best = True

        if config.save_every>0 or is_best:
            stats = {
                'loss': loss_history[-1],
                'train_f2': train_f2_history[-1],
                'train_acc': train_all_or_none_acc_history[-1],
                'train_recall': train_global_recall_history[-1],
            }
            if config.val_loader:
                stats['val_f2'] = val_f2_history[-1]
                stats['val_acc'] = val_all_or_none_acc_history[-1]
                stats['val_recall'] = val_global_recall_history[-1]
            if not config.no_save and (epoch+1) % config.save_every==0 :
                checkpointModel(model, config, optimizer, epoch, stats, is_best)
        gc.collect()

    print("\nFinished training.")
   
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
