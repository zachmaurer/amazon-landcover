from torch import nn

from utils import train
#from utils import NaiveDataset
from utils import Config, parseConfig
from utils.layers import Flatten

def createModel(config):
    model = nn.Sequential(
                      # Conv_Relu_BatchNorm --> 32 x 32
                      nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 2),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
      
                      # Conv_Relu_BatchNorm_Maxpool --> 32 x 14 x 14
                      nn.Conv2d(32, 32, kernel_size=7, stride=1),
                      nn.ReLU(inplace=True),
                      nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
                      nn.MaxPool2d(2, stride=2),
      
                      # Aggregation Layers
                      Flatten(), # see above for explanation
                      nn.Linear(6272, 2048), # affine layer
                      nn.ReLU(inplace=True),
                      nn.BatchNorm1d(2048, affine = True),
                      nn.Dropout(p=0.55, inplace = True),
                      nn.Linear(2048, 512), # affine layer
                      nn.ReLU(inplace=True),
                      nn.BatchNorm1d(512, affine = True),
                      nn.Dropout(p=0.3, inplace = True),
                      nn.Linear(512, 10),
            )
    model = model.type(config.dtype)
    return model 


def main():
    # Get Config
    args = parseConfig("Zach's HW2 ConvNet Model")
    config = Config(args)
    config.log(config)

    # Create Model
    model = createModel(config)

    # Train Model
    train(model, config)

    # Final Eval
    

if __name__ == '__main__':
    main()



