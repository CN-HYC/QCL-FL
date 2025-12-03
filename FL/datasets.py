import torch
import random
from torch.utils import data


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class = None, target_class = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class  
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class = None, target_class = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class  
            
    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class 
        return x, y 

    def __len__(self):
        return len(self.dataset)


class RandomFlipDataset(data.Dataset):

    def __init__(self, dataset):
        super(RandomFlipDataset, self).__init__()
        self.dataset = dataset
        num_classes = 10
        classes = list(range(num_classes))
        mapping = classes.copy()

        while any(o == f for o, f in zip(classes, mapping)):
            random.shuffle(mapping)

        self.flip_map = dict(zip(classes, mapping))

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        new_label = self.flip_map[label]
        return image, new_label

    def __len__(self):
        return len(self.dataset)