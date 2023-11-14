import os
import json
import subprocess
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class Transient(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.root = os.path.abspath(root)
        self.phase = phase
        self.img_list = []
        self.transform = transform
        # download_coco2014(self.root, phase)
        self.get_anno()
        self.num_classes = len(self.cat2idx)
        self.inp_name = './word2vec/Transient_Attributes_word2vec_29.pkl'

        if self.inp_name is not None:
            with open(self.inp_name, 'rb') as f:
                self.inp = pickle.load(f)
        else:
            self.inp = np.identity(80)

        print('[dataset] COCO2014 classification phase={} number of classes={}  number of images={}'.format(phase, self.num_classes, len(self.img_list)))
        # print(self.inp)

    def get_anno(self):
        list_path = os.path.join(self.root, '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        filename = item['file_name']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, '{}'.format(self.phase), filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # target = np.zeros(self.num_classes, np.float32) - 1
        target = torch.zeros(self.num_classes,  dtype=torch.float32) - 1
        target[labels] = 1
        inp = torch.Tensor(np.array(self.inp))
        data = {'image':img, 'name': filename, 'target': target, 'inp': inp}
        return data
        # return image, target
        # return (img, filename), target
