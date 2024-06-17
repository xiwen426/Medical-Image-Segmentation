import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


import nibabel as nib

class Amos_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
  
        self.base_dir = '/data2/wyx/medical/amos22/'
        data_name_list = []
        cls_name_list = []
        # for item in os.listdir(self.base_dir + 'datatr/'):
        #     data_name_list.append(item)
        #     cls_name_list.append('tr')
        
        # for item in os.listdir(self.base_dir + 'datats/'):
        #     data_name_list.append(item)
        #     cls_name_list.append('ts')
            
        for item in os.listdir(self.base_dir + 'datava/224/'):
            data_name_list.append(item)
            cls_name_list.append('va')
        
        print(len(data_name_list), len(cls_name_list))
        self.sample_list = data_name_list
        self.cls_name_list = cls_name_list
        
    

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        vol_name = self.sample_list[idx]
        filepath = self.base_dir + 'data' + self.cls_name_list[idx] + '/224/' + vol_name
        # "/{}.npy.h5".format(vol_name)
        
        # data = h5py.File(filepath) # without resolution
        data = np.load(filepath)
        image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample







class AMOSDataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
    
        self.transform = transform
        self.split = split
        # 读取包含样本文件名的列表文件
        
        self.base_dir = '/data2/wyx/medical/amos22/'
        image_name_list = []
        label_name_list = []
        names = ['Tr', 'Ts', 'Va']
        cls_list = []
        for name in names:
            for item in os.listdir(self.base_dir + 'images' + name + '/'):
                
                image_name_list.append(item)
                cls_list.append(name) # 存放术语属于哪个目录
            for item in os.listdir(self.base_dir + 'labels' + name + '/'):
                print(self.base_dir + 'labels' + name + '/')
                label_name_list.append(item)
            print(len(image_name_list), len(label_name_list),len(cls_list))
        
        self.image_name_list = image_name_list
        self.label_name_list = label_name_list
        self.cls_list = cls_list


    def __len__(self):
       
        return len(self.image_name_list)

    def __getitem__(self, idx):

        case_name = self.image_name_list[idx]
        image_path = self.base_dir + 'images' + self.cls_list[idx] + '/' + case_name
        label_path = self.base_dir + 'labels' + self.cls_list[idx] + '/' + case_name
        img = nib.load(image_path)
        label_img = nib.load(label_path)
        
        data = img.get_fdata()
        label = label_img.get_fdata()
        
        # Convert to PyTorch tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        sample = {'image': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = case_name
        return sample