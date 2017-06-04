from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from utils import train, predict
from utils import NaiveDataset, splitIndices, UpsamplingWeights
from utils import Config, parseConfig
from utils.layers import Flatten, Conv_BN_Relu, initialize_weights
from utils.constants import NUM_CLASSES, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_TRAIN, TEST_DATA_PATH, TEST_LABELS_PATH
from utils import visualize
from random import randint
import numpy as np


class BranchedCNN(nn.Module):
    def __init__(self, config):
      super(BranchedCNN, self).__init__()
      self.stem = nn.Sequential(
          Conv_BN_Relu(3, 32, kernel_size = 1, stride = 1, padding = 0),
          Conv_BN_Relu(32, 64, kernel_size = 3, stride = 1, padding = 0),
          Conv_BN_Relu(64, 64, kernel_size = 1, stride = 1, padding = 0)
        )

      self.conv1 = nn.Sequential(
          Conv_BN_Relu(64, 32, kernel_size = 1, stride = 1, padding = 0),
          Conv_BN_Relu(32, 32, kernel_size = 3, stride = 1, padding = 1),
          Conv_BN_Relu(32, 64, kernel_size = 1, stride = 1, padding = 0),
          nn.MaxPool2d(2, stride=2), 
          Conv_BN_Relu(64, 64, kernel_size = 5, stride = 1, padding = 1),
          nn.MaxPool2d(2, stride=2)
        )

      self.conv2 = nn.Sequential(
          nn.UpsamplingNearest2d(scale_factor = 2),
          Conv_BN_Relu(64, 128, kernel_size = 1, stride = 1, padding = 0),
          Conv_BN_Relu(128, 128, kernel_size = 3, stride = 1, padding = 1),
          Conv_BN_Relu(128, 256, kernel_size = 1, stride = 1, padding = 0),
          nn.UpsamplingNearest2d(scale_factor = 2),
        )

      self.feats_decoder = nn.Sequential(
        Conv_BN_Relu(64, 1, kernel_size = 1, stride = 1, padding = 0),
        nn.MaxPool2d(4, stride=2),
        Flatten(),
        nn.Linear(15876, NUM_CLASSES)
      )

      self.conv1_decoder = nn.Sequential(
        Conv_BN_Relu(64, 1, kernel_size = 1, stride = 1, padding = 0),
        Flatten(),
        nn.Linear(3844, NUM_CLASSES)
      )

      self.conv2_decoder = nn.Sequential(
        nn.MaxPool2d(2, stride=2),
        Conv_BN_Relu(256, 3, kernel_size = 1, stride = 1, padding = 0),
        Flatten(),
        nn.Linear(46128, NUM_CLASSES)
      )

      self.conv1_conv2_short = nn.Sequential(
          Conv_BN_Relu(64, 256, kernel_size = 1, stride = 1, padding = 0),
          nn.UpsamplingNearest2d(scale_factor = 2),
          Conv_BN_Relu(256, 256, kernel_size = 1, stride = 1, padding = 0),
          nn.UpsamplingNearest2d(scale_factor = 2),
        )

    def forward(self, x):
      feats = self.stem(x)
      conv1 = self.conv1(feats)
      conv2 = self.conv2(conv1)

      feats_activation = self.feats_decoder(feats)
      conv1_activation = self.conv1_decoder(conv1)
      conv1_short = self.conv1_conv2_short(conv1)
      conv2_activation = self.conv2_decoder(conv2 + conv1_short)

      activations = feats_activation + conv1_activation + conv2_activation
      return activations



def randomTranspose(x):
  k = randint(0,4)
  x = np.rot90(x, k = k)
  return x

def main():
    # Get Config
    args = parseConfig("Zach's Branched ConvNet Model")
    config = Config(args)
    config.log(config)

    # Transformations
    size = 256
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
    model = BranchedCNN(config)
    if config.use_gpu:
      model = model.cuda()

    # Init Weights
    model.apply(initialize_weights)


    # Train and Eval Model
    #results = train(model, config)
    #results = train(model, config, lr_decay = 0.00001)
    results = train(model, config, lr_decay = 0.00005, weight_decay = 0.0005)
    visualize.plot_results(results, config)

    make_predictions = False
    if make_predictions:
      predict(model, config, test_loader, dataset = "test")
      predict(model, config, train_loader, dataset = "train")
      predict(model, config, val_loader, dataset = "val")

if __name__ == '__main__':
    main()



