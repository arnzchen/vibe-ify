import random
import os
import pickle
import math
from collections import defaultdict, namedtuple

import numpy as np
from tqdm import tqdm, trange
import torch

# TODO imports
from .util import suppress_stdout
from .constants import *


DatasetInfo = namedtuple('DatasetInfo', []) # TODO fields in list


def collate(batch):
    raise NotImplementedError


class Dataset:
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        # TODO walk data
        self.splits = {} # TODO
        self.splits['dev'] = None
        self.splits['test'] = None
        self.splits['train'] = None
        print('done loading data')
        print('split sizes:')
        for key in ['train', 'dev', 'test']:
            print(key, len(self.splits[key]))

        if args.dataset_info is not None:
            with open(args.dataset_info, 'rb') as rf:
                self.dataset_info = pickle.load(rf)
        else:
            self.dataset_info = DatasetInfo() # TODO fields


    def shuffle(self, split, seed=None):
        assert split in ['train', 'dev', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])


    def loader(self, split, batch_size=32, num_workers=20, indices=None):
        assert split in ['train', 'dev', 'test']
        data = self.splits[split] if indices is None else [self.splits[split][i] for i in indices]
        return torch.utils.data.DataLoader(SplitLoader(data), pin_memory=True, collate_fn=collate, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return self
    

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self.data):
                raise StopIteration
            example = None # TODO load self.data[self.pos]
            valid = True
            self.pos += increment
        return example