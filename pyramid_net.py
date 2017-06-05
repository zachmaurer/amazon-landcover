from torchvision import transforms
from torch import nn
import torch.nn.functional as F
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
import torch

def make_linear_bn_prelu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.PReLU(out_channels),
    ]


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_max_flat(out):
    flat = F.adaptive_max_pool2d(out,output_size=1)  ##nn.AdaptiveMaxPool2d(1)(out)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_avg_flat(out):
    flat =  F.adaptive_avg_pool2d(out,output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_shortcut(out, modifier):
    if modifier is None:
        return out
    else:
        return modifier(out)

def make_flat(out):
    #flat =  F.adaptive_avg_pool2d(out,output_size=4)
    out  = F.avg_pool2d(out,kernel_size=4, stride=2, padding=0)
    out  = F.adaptive_max_pool2d(out,output_size=1)
    flat = out.view(out.size(0), -1)
    return flat


#############################################################################3


class PyNet_10(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(PyNet_10, self).__init__()
        in_channels, height, width = in_shape

        self.preprocess = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
        ) # 128

        self.conv1d = nn.Sequential(
            *make_conv_bn_relu(16,32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32,64, kernel_size=1, stride=1, padding=0 ),
        ) # 128
        self.shortld = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)


        self.conv2d = nn.Sequential(
            *make_conv_bn_relu(64,64,  kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64,64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64,128, kernel_size=1, stride=1, padding=0 ),
        ) # 64
        self.short2d = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3d = nn.Sequential(
            *make_conv_bn_relu(128,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(128,256, kernel_size=1, stride=1, padding=0 ),
        ) # 32
        self.short3d = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv4d = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 16
        self.short4d = None #nn.Identity()

        self.conv5d = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 8
        self.short5d = None #  nn.Identity()


        self.conv4u = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 16

        self.conv3u = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(128,128, kernel_size=1, stride=1, padding=0 ),
        ) # 32

        self.conv2u = nn.Sequential(
            *make_conv_bn_relu(128,64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 64,64, kernel_size=1, stride=1, padding=0 ),
        ) # 64

        self.conv1u = nn.Sequential(
            *make_conv_bn_relu(64,64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64,64, kernel_size=1, stride=1, padding=0 ),
        ) # 128


        self.cls2d = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls5d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )

        self.cls1u = nn.Sequential(
            *make_linear_bn_relu(64,  512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls2u = nn.Sequential(
            *make_linear_bn_relu( 64, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3u = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4u = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )



    def forward(self, x):

        out    = self.preprocess(x)                                       #128

        conv1d = self.conv1d(out)                                         #128
        out    = F.max_pool2d(conv1d, kernel_size=2, stride=2)  # 64

        conv2d = self.conv2d(out) + make_shortcut(out, self.short2d)      # 64
        out    = F.max_pool2d(conv2d, kernel_size=2, stride=2) # 32
        flat2d = make_max_flat(out)

        conv3d = self.conv3d(out) + make_shortcut(out, self.short3d)      # 32
        out    = F.max_pool2d(conv3d, kernel_size=2, stride=2) # 16
        flat3d = make_max_flat(out)

        conv4d = self.conv4d(out) + make_shortcut(out, self.short4d)      # 16
        out    = F.max_pool2d(conv4d, kernel_size=2, stride=2) #  8
        flat4d = make_max_flat(out)

        conv5d = self.conv5d(out) + make_shortcut(out, self.short5d)      #  8
        out    = conv5d                                        #  4
        flat5d = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 16
        out    = out + conv4d
        out    = self.conv4u(out)
        flat4u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 32
        out    = out + conv3d
        out    = self.conv3u(out)
        flat3u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 64
        out    = out + conv2d
        out    = self.conv2u(out)
        flat2u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      #128
        out    = out + conv1d
        out    = self.conv1u(out)
        flat1u = make_max_flat(out)



        logit2d = self.cls2d(flat2d).unsqueeze(2)
        logit3d = self.cls3d(flat3d).unsqueeze(2)
        logit4d = self.cls4d(flat4d).unsqueeze(2)
        logit5d = self.cls5d(flat5d).unsqueeze(2)

        logit1u = self.cls1u(flat1u).unsqueeze(2)
        logit2u = self.cls2u(flat2u).unsqueeze(2)
        logit3u = self.cls3u(flat3u).unsqueeze(2)
        logit4u = self.cls4u(flat4u).unsqueeze(2)


        logit = torch.cat((logit2d,logit3d,logit4d,logit5d,logit1u,logit2u,logit3u,logit4u) , 2)

        logit = F.dropout(logit, p=0.15,training=self.training)
        logit = logit.sum(2)
        logit = logit.view(logit.size(0),logit.size(1)) #unsqueeze(2)
        prob  = F.sigmoid(logit)

        return logit,prob


def randomTranspose(x):
  k = randint(0,4)
  x = np.rot90(x, k = k)
  return x

def main():
    # Get Config
    args = parseConfig("Heng Cher Keng's Branched ConvNet Model")
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


    #weights = UpsamplingWeights(train_dataset)

    #train_sampler = WeightedRandomSampler(weights = weights[train_idx], replacement = True, num_samples = config.num_train)
    train_sampler = SubsetRandomSampler(train_idx)
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
    model = PyNet_10((3, 256, 256), 17)
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