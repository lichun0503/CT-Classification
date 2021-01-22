import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DatasetGenerator_train (Dataset):

    def __init__ (self, pathImageDirectory, transform,model_name):
        self.images_root = os.path.join(pathImageDirectory, 'train_images')
        self.filenames = os.listdir(self.images_root)
        self.filenames.sort()
        self.category=["COVID","NonCOVID"]
        self.transform=transform
        self.scale=[0.75,0.5,1,1.25]
        self.model_name=model_name
    def __getitem__(self, index):
        choice=int(np.random.randint(0,3,1))
        scale=self.scale[choice]
        filename = self.filenames[index]
        with open(os.path.join(self.images_root,filename), 'rb') as f:
            image = Image.open(f).convert('RGB')
            _,w,h=np.shape(image)
          #  image=image.resize((int(w*scale),int(h*scale)))
        image = self.transform(image)
        category= filename.split("__")[0]
        ind=self.category.index(category)
        label = torch.zeros(2)
        label[ind]=1
        return image,label


    def __len__(self):
        return len(self.filenames)


class DatasetGenerator_test(Dataset):

    def __init__(self, pathImageDirectory,  transform,model_name):

        self.images_root = os.path.join(pathImageDirectory, 'test_images')
        self.filenames = os.listdir(self.images_root)
        self.filenames.sort()
        self.category = ["COVID", "NonCOVID"]
        self.transform = transform
        self.model_name = model_name

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(os.path.join(self.images_root, filename), 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.transform(image)
        category = filename.split("__")[0]
        ind = self.category.index(category)
        label = torch.zeros(2)
        label[ind] = 1
        return image, label

    def __len__(self):
        return len(self.filenames)