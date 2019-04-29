# -*- coding: utf-8 -*-
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import h5py


def get_cifar10(root, label_mode='original'):
    '''
    :param root:...
    :param label_mode:'original' or 'random' or 'partially-0.x'
    :return:
    '''
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    root = os.path.expanduser(root)

    # now load the picked numpy arrays
    data = []
    labels = []
    for fentry in train_list + test_list:
        f = fentry[0]
        file = os.path.join(root, base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        data.append(entry['data'])
        if 'labels' in entry:
            labels += entry['labels']
        else:
            labels += entry['fine_labels']
        fo.close()

    data = np.concatenate(data)
    data = data.reshape((60000, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))

    if label_mode == 'original':
        labels = labels
    elif label_mode == 'random':
        labels = np.random.choice(range(10), 50000)
    elif label_mode.startswith('partially'):
        p = float(label_mode.split('-')[1])
        labels = [y if np.random.uniform() > p else
                  np.random.randint(0, 10) for y in labels]

    # train_data = data[:50000]
    # train_label = labels[:50000]
    # test_data = data[50000:]
    # test_label = labels[50000:]
    train_data, test_data, train_label, test_label = train_test_split(data, labels, train_size=5 / 6)

    return train_data, train_label, test_data, test_label


def get_cifar100(root, label_mode='original'):
    '''
    :param root:...
    :param label_mode:'original' or 'random' or 'partially-0.x'
    :return:
    '''
    base_folder = 'cifar-100-python'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    root = os.path.expanduser(root)

    # now load the picked numpy arrays
    data = []
    labels = []
    for fentry in train_list + test_list:
        f = fentry[0]
        file = os.path.join(root, base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        data.append(entry['data'])
        if 'labels' in entry:
            labels += entry['labels']
        else:
            labels += entry['fine_labels']
        fo.close()

    data = np.concatenate(data)
    data = data.reshape((60000, 3, 32, 32))
    data = data.transpose((0, 2, 3, 1))

    if label_mode == 'original':
        labels = labels
    elif label_mode == 'random':
        labels = np.random.choice(range(100), 50000)
    elif label_mode.startswith('partially'):
        p = float(label_mode.split('-')[1])
        labels = [y if np.random.uniform() > p else
                  np.random.randint(0, 100) for y in labels]

    # train_data = data[:50000]
    # train_label = labels[:50000]
    # test_data = data[50000:]
    # test_label = labels[50000:]
    train_data, test_data, train_label, test_label = train_test_split(data, labels, train_size=5 / 6)

    return train_data, train_label, test_data, test_label


def get_miniImageNet(root, label_mode='original'):
    data = h5py.File(root, 'r')
    data = np.concatenate([np.array(data['data_train'],np.uint8).reshape(-1, 84, 84, 3),
                           np.array(data['data_val'],np.uint8).reshape(-1, 84, 84, 3),
                           np.array(data['data_test'],np.uint8).reshape(-1, 84, 84, 3)])
    labels = np.repeat(np.arange(100), 600, axis=0)


    if label_mode == 'original':
        labels = labels
    elif label_mode == 'random':
        labels = np.random.choice(range(100), 50000)
    elif label_mode.startswith('partially'):
        p = float(label_mode.split('-')[1])
        labels = [y if np.random.uniform() > p else
                  np.random.randint(0, 100) for y in labels]

    # train_data = data[:50000]
    # train_label = labels[:50000]
    # test_data = data[50000:]
    # test_label = labels[50000:]
    train_data, test_data, train_label, test_label = train_test_split(data, labels, train_size=5 / 6)

    return train_data, train_label, test_data, test_label


class Cifar10_new(Dataset):
    def __init__(self, data, labels, tranform=None):
        super(Cifar10_new, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = tranform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.labels)


def get_loader(root, batch_size=100, label_mode='original', transform=None, name='cifar10'):
    if name == 'cifar10':
        train_data, train_label, test_data, test_label = get_cifar10(root, label_mode)
    elif name == 'cifar100':
        train_data, train_label, test_data, test_label = get_cifar100(root, label_mode)
    elif name == 'miniImageNet':
        train_data, train_label, test_data, test_label = get_miniImageNet(root, label_mode)

    train_dataset = Cifar10_new(train_data, train_label, transform)
    test_dataset = Cifar10_new(test_data, test_label, transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return (train_dataset, test_dataset), \
           (train_loader, test_loader)
