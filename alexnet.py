import torch
from torch import nn
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import train, predict
from utils import NaiveDataset, splitIndices
from utils import Config, parseConfig
from utils.layers import Flatten
from utils.constants import NUM_CLASSES, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_TRAIN, TEST_DATA_PATH, TEST_LABELS_PATH
from utils import visualize


Ndim, Cdim, Hdim, Wdim = 0, 1, 2, 3

##########
## U-Net ##
##########

class AlexNet(nn.Module):
  def __init__(self):
    super().__init__()
    model_conv = torchvision.models.alexnet(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    self.features = model_conv.features
    num_ftrs = list(model_conv.classifier.children())[-1].in_features 
    mod = list(model_conv.classifier.children())
    mod.pop()
    mod.append(nn.Linear(num_ftrs, NUM_CLASSES))
    self.classifier = nn.Sequential(*mod)

  def forward(self, x):
    f = self.features(x)
    f = f.view(f.size(0), 256*6*6)
    y = self.classifier(f)
    return y
    
    


def main():
    # Get Config
    args = parseConfig("Alexnet")
    config = Config(args)
    config.log(config)
    
    transform = transforms.Compose([
            transforms.CenterCrop(224)
        ])
    train_dataset = NaiveDataset(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, num_examples = NUM_TRAIN, transform=transform)
    train_idx, val_idx = splitIndices(train_dataset, config, shuffle = True)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 3, sampler = train_sampler)
    val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 1, sampler = val_sampler)
    
    config.train_loader = train_loader
    config.val_loader = val_loader

    # Create Model
    model = AlexNet()
    if config.use_gpu:
      model = model.cuda()

    # Train and Eval Model
    print(list(model.classifier.children())[-1])
    optimizer_conv = optim.Adam(list(model.classifier.children())[-1].parameters(), config.lr)

    results = train(model, config, optimizer=optimizer_conv)
    visualize.plot_results(results, config)
  
    # Evaluate Results
    test_dataset = NaiveDataset(TEST_DATA_PATH, TEST_LABELS_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, num_workers = 3)

    make_predictions = True
    if make_predictions:
      predict(model, config, test_loader, dataset = "test")
      predict(model, config, train_loader, dataset = "train")
      predict(model, config, val_loader, dataset = "val")

if __name__ == '__main__':
    # model = UNet()
    # x = torch.autograd.Variable(torch.randn(1, 3, 256, 256))
    # model(x)
    main()



