import torch
from torch import nn
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import SubsetRandomSampler

from utils import train, predict
from utils import NaiveDataset, splitIndices
from utils import Config, parseConfig
from utils.layers import Flatten
from utils.constants import NUM_CLASSES, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, NUM_TRAIN
from utils import visualize


# Architecture from "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015) 
# Paper Link: https://arxiv.org/pdf/1505.04597.pdf
# Referenced: https://github.com/lopuhin/kaggle-dstl/blob/292840bf4faf49ecf7c74bed9b6d91982a139090/models.py#L211

Ndim, Cdim, Hdim, Wdim = 0, 1, 2, 3

################
## Conv Module ##
################

class UNetConvModule(nn.Module):
  def __init__(self, input_dim, output_dim, bn = True, upsample = False):
    super().__init__()
    self.conv1 = nn.Conv2d(input_dim, output_dim, 3, padding =1)
    self.conv2 = nn.Conv2d(output_dim, output_dim, 3, padding =1)
    self.activation = nn.ReLU()
    
    self.bn = bn
    if bn:
      self.bn1 = nn.BatchNorm2d(output_dim)
      self.bn2 = nn.BatchNorm2d(output_dim)

    self.up_sample = upsample
    if self.up_sample:
      self.upsample = nn.UpsamplingNearest2d(scale_factor = 2)

  def forward(self, input, branch_input = None):
    if branch_input:
      input = torch.cat((input, branch_input), Cdim) # Channel-wise concat, NCHW
    
    # Conv1
    x = self.conv1(input)
    x = self.activation(x)
    if self.bn:
      x = self.bn1(x)

   # Conv2
    x = self.conv2(x)
    x = self.activation(x)
    if self.bn:
      x = self.bn2(x)

    # Upsample
    if self.up_sample:
      x = self.upsample(x)

    return x
   
####################
## Pool-Crop Module ##
####################

class UNetPoolCropModule(nn.Module):
  def __init__(self, crop_size = None):
    super().__init__()
    self.pool = nn.MaxPool2d(2, 2)
    self.crop_size = crop_size

  def forward(self, input):
    pooled = self.pool(input)
    if self.crop_size:
      _, _, w, h = input.size()
      start = w - self.crop_size - 1
      end = start + self.crop_size
      indices = torch.autograd.Variable(torch.LongTensor(list(range(start, end))))
      cropped = torch.index_select(input, Hdim, indices)
      cropped = torch.index_select(cropped, Wdim, indices)
      #cropped = torch.Variable(cropped)
    else:
      cropped = input
    return pooled, cropped

##########
## U-Net ##
##########

class UNet(nn.Module):
  def __init__(self):
    super().__init__()

    # Seg Net
    # self.crop_seq = [256, 128, 32, 18]
    self.filter_seq = [128, 256, 512, 1024]

    self.conv1 = UNetConvModule(3, self.filter_seq[0])
    self.pc1 = UNetPoolCropModule(crop_size = None)

    self.conv2 = UNetConvModule(self.filter_seq[0], self.filter_seq[1])
    self.pc2 = UNetPoolCropModule(crop_size = None)

    self.conv3 = UNetConvModule(self.filter_seq[1], self.filter_seq[2])
    self.pc3 = UNetPoolCropModule(crop_size = None)
    
    # Aggregate Net
    self.agg1 = UNetConvModule(self.filter_seq[0]*2, self.filter_seq[0])
    self.agg2 = UNetConvModule(self.filter_seq[1]*2, self.filter_seq[0], upsample = True)
    self.agg3 = UNetConvModule(self.filter_seq[2], self.filter_seq[1], upsample = True)

    # Output Layer
    self.out_conv = nn.Conv2d(self.filter_seq[0], 1, 3, padding =1)
    self.flatten = Flatten()
    self.out_dense = nn.Linear(256**2, NUM_CLASSES)

  def forward(self, input):
    # Seg Net
    h1 = self.conv1(input)
    h1_pool, h1_crop = self.pc1(h1)

    h2 = self.conv2(h1_pool)
    h2_pool, h2_crop = self.pc2(h2)

    h3 = self.conv3(h2_pool)
    _, h3_crop  = self.pc3(h3)
    
    # Agg Net
    q3 = self.agg3(h3_crop)
    q2 = self.agg2(h2_crop, branch_input = q3)
    q1 = self.agg1(h1_crop, branch_input = q2)
    
    # Output
    out = self.out_conv(q1)
    out = self.flatten(out)
    out = self.out_dense(out)
    return out



def main():
    # Get Config
    args = parseConfig("U-Net Model ( https://arxiv.org/pdf/1505.04597.pdf )")
    config = Config(args)
    config.log(config)

    train_dataset = NaiveDataset(TRAIN_DATA_PATH, TRAIN_LABELS_PATH, num_examples = NUM_TRAIN)
    train_idx, val_idx = splitIndices(train_dataset, config, shuffle = True)

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 4, sampler = train_sampler)
    val_loader = DataLoader(train_dataset, batch_size = config.batch_size, num_workers = 4, sampler = val_sampler)
    
    config.train_loader = train_loader
    config.val_loader = val_loader

    # Create Model
    model = UNet().type(config.dtype)

    # Train and Eval Model
    results = train(model, config)
    visualize.plot_results(results, config)

    predict(model, config) # TODO, see train.py
    

if __name__ == '__main__':
    # model = UNet()
    # x = torch.autograd.Variable(torch.randn(1, 3, 256, 256))
    # model(x)
    main()



