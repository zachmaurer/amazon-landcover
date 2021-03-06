from torch import nn
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler

from utils import train, predict
from utils import NaiveDataset, splitIndices
from utils import Config, parseConfig
from utils.layers import Flatten, initialize_weights
from utils.constants import NUM_CLASSES, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_TRAIN, TEST_DATA_PATH, TEST_LABELS_PATH
from utils import visualize

class BaselineCNN(nn.Module):
    def __init__(self, config):
      super(BaselineCNN, self).__init__()
      self.model = nn.Sequential(
                        # Conv_Relu_BatchNorm --> 32 x 32
                        nn.Conv2d(3, 32, kernel_size = 7, stride = 1, padding = 2),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                        nn.MaxPool2d(4, stride=2),
        
                        # Conv_Relu_BatchNorm_Maxpool --> 32 x 14 x 14
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding = 2),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                        nn.MaxPool2d(2, stride=2),
        
                        # Aggregation Layers
                        Flatten(), # see above for explanation
                        nn.Linear(131072, 2048), # affine layer
                        nn.ReLU(inplace = False),
                        nn.Dropout(p=0.55, inplace = False),
                        nn.Linear(2048, NUM_CLASSES), # affine layer
              )
      self.model = self.model.type(config.dtype)
    
    def forward(self, input):
      return self.model(input)


def main():
    # Get Config
    args = parseConfig("Zach's HW2 ConvNet Model")
    config = Config(args)
    config.log(config)

    train_dataset = NaiveDataset(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, num_examples = NUM_TRAIN)
    train_idx, val_idx = splitIndices(train_dataset, config, shuffle = True)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 4, sampler = train_sampler)
    val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 1, sampler = val_sampler)
    
    config.train_loader = train_loader
    config.val_loader = val_loader
    
    #get test data
    test_dataset = NaiveDataset(TEST_DATA_PATH, TEST_LABELS_PATH)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, num_workers = 2)
    
    # Create Model
    model = BaselineCNN(config)
    model.apply(initialize_weights)

    # Train and Eval Model
    results = train(model, config)
    visualize.plot_results(results, config)

    make_predictions = True
    if make_predictions:
      predict(model, config, train_loader, dataset = "train")
      predict(model, config, val_loader, dataset = "val")
      predict(model, config, test_loader, dataset = "test")

if __name__ == '__main__':
    main()



