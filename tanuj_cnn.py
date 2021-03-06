from torch import nn
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

from utils import train, predict
from utils import NaiveDataset, splitIndices, UpsamplingWeights
from utils import Config, parseConfig
from utils.layers import Flatten, initialize_weights
from utils.constants import NUM_CLASSES, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_TRAIN, TEST_DATA_PATH, TEST_LABELS_PATH
from utils import visualize
from torchvision import transforms
from random import randint
import numpy as np

def createModel(config):
    model = nn.Sequential(
                      # Conv_Relu_BatchNorm --> 32 x 32
                      nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 2),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                      nn.MaxPool2d(4, stride=2),
      
                      # Conv_Relu_BatchNorm_Maxpool --> 32 x 14 x 14
                      nn.Conv2d(32, 32, kernel_size=7, stride=2, padding = 2),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                      nn.MaxPool2d(2, stride=2),
      
                      # Aggregation Layers
                      Flatten(), # see above for explanation
                      nn.Linear(1152, 512), # affine layer
                      nn.ReLU(inplace = False),
                      #nn.Dropout(p=0.45, inplace = False), #don't use dropout until I overfit..
                      nn.Linear(512, NUM_CLASSES), # affine layer
            )
    if config.use_gpu:
      model = model.cuda()
    return model

def randomTranspose(x):
  k = randint(0,4)
  x = np.rot90(x, k = k)
  return x

def main():
    # Get Config
    args = parseConfig("Tanuj's ConvNet Model")
    config = Config(args)
    config.log(config)


    # Transformations
    size = 112
    transformations = transforms.Compose([ 
                                  transforms.Scale(size+5),
                                  transforms.RandomCrop(size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.Lambda(lambda x: randomTranspose(np.array(x))),
                                  transforms.Lambda(lambda x: np.array(x)[:,:,:3]),                                      
                                  transforms.ToTensor(),
                              ])



    # Datasets
    train_dataset = NaiveDataset(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, num_examples = NUM_TRAIN, transforms = transformations)
    train_idx, val_idx = splitIndices(train_dataset, config, shuffle = True)


    weights = UpsamplingWeights(train_dataset)

    train_sampler = WeightedRandomSampler(weights = weights[train_idx], replacement = True, num_samples = config.num_train)
    val_sampler = SubsetRandomSampler(val_idx)


    # Loaders
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 4, sampler = train_sampler)
    val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 1, sampler = val_sampler)
    
    config.train_loader = train_loader
    config.val_loader = val_loader
    
    #get test data
    test_dataset = NaiveDataset(TEST_DATA_PATH, TEST_LABELS_PATH)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, num_workers = 2)
    
    # Create Model
    model = createModel(config)

    # Init Weights
    model.apply(initialize_weights)

    # Train and Eval Model
    results = train(model, config, lr_decay = 0.0001, weight_decay = 0.0005)
    visualize.plot_results(results, config)

    make_predictions = False
    if make_predictions:
      predict(model, config, test_loader, dataset = "test")
      #predict(model, config, train_loader, dataset = "train")
      #predict(model, config, val_loader, dataset = "val")

if __name__ == '__main__':
    main()



