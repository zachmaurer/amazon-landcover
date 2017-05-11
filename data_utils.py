#this library contains all the data utils, including data loading, data preprocessing, etc

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from PIL import Image

#some constant paths. Move this to a global config later
train_data_path = './input/train-jpg/'
test_data_path = './input/test-jpg/'
train_labels_path ='./input/train_v2.csv' 


#this is the naive implementation which pulls from file every time you get an item. no caching. Probably not useful anymore
class NaiveDataset(Dataset):
    """Dataset wrapping data and target tensors. Naive implementation does data preprocessing per 'get_item' call

    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    
    Arguments:
        data_path (str): path to image folder
        labels_path (str): path to csv containing labels per image
        num_examples (int): number of examples
    """
    def load_image(self, idx):
        image_name = self.labels_df['image_name'][idx]
        im = Image.open(self.data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]
        im = np.reshape(im,(im.shape[2],im.shape[0],im.shape[1]))
        return torch.from_numpy(im)
    
    def __init__(self, data_path=train_data_path, labels_path=train_labels_path,
                 num_examples=1000):
        self.labels_df = pd.read_csv(labels_path)
        assert num_examples <= self.labels_df.shape[0]
        self.num_examples = num_examples
        
        mlb = MultiLabelBinarizer()
        labels_words = [set([word for word in row.split()]) for row in self.labels_df['tags']]
        self.labels_tensor = torch.from_numpy(mlb.fit_transform(labels_words))
        
        self.data_path = data_path

    def __getitem__(self, idx):
        data_tensor = self.load_image(idx)
        target_tensor = self.labels_tensor[idx]
        return data_tensor,target_tensor

    def __len__(self):
        return self.num_examples

class DynamicDataset(Dataset):
    """Dataset wrapping data and target tensors with dynamic loading and buffering

    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    
    Precondition - buffer_size must be a multiple of num_examples (relax this later)

    Arguments:
        data_path (str): path to image folder
        labels_path (str): path to csv containing labels per image
        num_examples (int): number of examples
        buffer_size (int): size of precaching buffer
        rand_seed (None/int): if None, go sequentially. If <0, use system clock for seed. If >0, use seed value
    """

    def __init__(self, data_path=train_data_path, labels_path=train_labels_path,
                 num_examples=1000, buffer_size=1000, rand_seed = None):
        self.labels_df = pd.read_csv(labels_path)
        assert num_examples <= self.labels_df.shape[0]
        assert num_examples >= buffer_size
        assert num_examples % buffer_size == 0
        
        mlb = MultiLabelBinarizer()
        labels_words = [set([word for word in row.split()]) for row in self.labels_df['tags']]
        self.labels_tensor = torch.from_numpy(mlb.fit_transform(labels_words))
        
        self.num_examples = num_examples
        self.buffer_size = buffer_size
        self.rand_seed = 0
        self.buffer_index = 0
        
        if rand_seed is None:
            self.inds_array = np.arange(num_examples)
        elif rand_seed<=0:
            self.inds_array = np.random.permutation(num_examples)
        elif rand_seed>0:
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            self.inds_array = np.random.permutation(num_examples)

        self.data_path = data_path
        
        self.data_tensor = np.zeros((buffer_size,3,256,256))
        self.backup_buffer = np.zeros(self.data_tensor.shape)
        
        self.buffer_flag = -1
        
        for i in range(self.buffer_index*buffer_size,self.buffer_index*buffer_size+1):
            self.data_tensor[i,:,:,:] = self.load_image(i)
        self.data_tensor = torch.from_numpy(self.data_tensor)
        
    def load_image(self, idx):
        image_name = self.labels_df['image_name'][idx]
        im = Image.open(self.data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]
        im = np.reshape(im,(im.shape[2],im.shape[0],im.shape[1]))
        return im  
    
    def fill_buffer(self):
        if self.buffer_index%2 == 0:
            del self.backup_buffer
            backup_buffer_np = np.zeros((self.buffer_size,3,256,256)) #does this clear the GPU RAM properly? Monitor memory..
            for i in range(int(self.buffer_size)):
                backup_buffer_np[i,:,:,:] = self.load_image(i+self.buffer_index*self.buffer_size)
            self.backup_buffer = torch.from_numpy(backup_buffer_np) #should this happen halfway, or at the end?
            del backup_buffer_np
        else:
            del self.data_tensor
            data_tensor_np = np.zeros((self.buffer_size,3,256,256)) #does this clear the GPU RAM properly? Monitor memory..
            for i in range(int(self.buffer_size)):
                data_tensor_np[i,:,:,:] = self.load_image(i+self.buffer_index*self.buffer_size)
            self.data_tensor = torch.from_numpy(data_tensor_np) #should this happen halfway, or at the end?
            del data_tensor_np

    def __getitem__(self, index):
        if index>(1+self.buffer_index)*self.buffer_size/2:
            self.fill_buffer()
        #elif index>int(3*self.buffer_index*self.buffer_size/4):
        #    self.backup_buffer = torch.from_numpy(self.backup_buffer)
        if index>=self.buffer_index*self.buffer_size:
            self.buffer_index += 1
        if self.buffer_index%2 == 1:
            return self.data_tensor[index%self.buffer_size], self.labels_tensor[index%self.buffer_size]
        else:
            return self.backup_buffer[index%self.buffer_size], self.labels_tensor[index%self.buffer_size]
            #self.data_tensor = self.backup_buffer #does this do assignment properly w/o causing a GPU/CPU RAM memory leak?
        #return self.data_tensor[index%self.buffer_size], self.labels_tensor[index%self.buffer_size]

    def __len__(self):
        return self.num_examples

#Since dataloaders are created in conjunction with samplers, and because of our RAM constraint when loading data,
#We needed to create this helper function to produce a dataloader object with the appropraite sampler. Without this helper
#function, there is a risk that the Sampler would not be able to sample random pictures properly
def createFastLoaderWithSampler(data_path=train_data_path, labels_path=train_labels_path,
                 num_examples=1000, buffer_size=1000, rand_seed = None, batch_size=100, num_workers=8):
    dynamic_dataset = DynamicDataset(data_path, labels_path,
                 num_examples, buffer_size, rand_seed)
    print(len(dynamic_dataset))
    return torch.utils.data.DataLoader(dynamic_dataset, batch_size=100, 
                                       shuffle=(rand_seed is not None), sampler=SequentialSampler(dynamic_dataset),
                                       num_workers=num_workers)