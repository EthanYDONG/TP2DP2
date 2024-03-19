from torch.utils.data import Dataset
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from copy import deepcopy

class data_preprocessor():
    def __init__(self, data_dir, dim, tmax, shuffle_flag = True, train_prop = 0.8, val_prop = 0.1):
        self.dim = dim
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.data_dir = data_dir
        self.shuffle_flag = shuffle_flag
        self.tmax = tmax
        self.size = None
        
        self.cluster_labels = None
        
        self.seq_times = None
        self.seq_types = None
        self.seq_lengths = None
        
        self.indexs_train = None
        self.indexs_val = None
        self.indexs_test = None
        
        np.random.seed(37)
        self.split_dataset()
    
    def split_dataset(self):
        f = open(self.data_dir, 'rb')
        data = pickle.load(f)
        
        events_list = data['event']
        self.cluster_labels = np.array(data['label'])
        event_times_list = [torch.from_numpy(seq[:, 0]) for seq in events_list]
        event_types_list = [torch.from_numpy(seq[:, 1]) for seq in events_list]
        seq_times = nn.utils.rnn.pad_sequence(event_times_list, batch_first=True, padding_value = self.tmax).float()
        self.seq_times = torch.cat((torch.zeros_like(seq_times[:, :1]), seq_times), dim = -1) 
        seq_types = nn.utils.rnn.pad_sequence(event_types_list, batch_first=True, padding_value = self.dim)
        
        self.seq_types = torch.cat((self.dim*torch.ones_like(seq_types[:, :1]), seq_types), dim = -1).long()
        self.seq_lengths = np.array(data['length']) + 1
        self.pad_masks = self.seq_times < self.tmax
        
        self.size = len(events_list)
        tmp = np.arange(0, self.size)
        if self.shuffle_flag:
            np.random.shuffle(tmp)
        self.indexs_train = tmp[:int(self.size * self.train_prop)]
        self.indexs_val = tmp[int(self.size * self.train_prop): int(self.size * (self.train_prop + self.val_prop))]
        self.indexs_test = tmp[int(self.size * (self.train_prop + self.val_prop)):]
    
    def get_data(self, flag = 'train'):
        if (flag == 'train'):
            indexs = self.indexs_train
        elif (flag == 'val'):
            indexs = self.indexs_val
        else:
            indexs = self.indexs_test
            
        return self.seq_times[indexs], self.seq_types[indexs], \
            self.seq_lengths[indexs], self.pad_masks[indexs], self.cluster_labels[indexs]
    def __getitem__(self, index):
        return (self.seq_times[index], self.seq_types[index], self.pad_masks[index])
 

class data_loader(Dataset):
    def __init__(self, times, types, lengths, pad_masks, labels, shuffle_flag = True):
        self.seq_times = times
        self.seq_types = types
        self.seq_lengths = lengths
        self.pad_masks = pad_masks
        self.cluster_labels = labels
        self.shuffle_flag = shuffle_flag
        self.cursor = 0
        self.indexs = np.arange(0, len(times))
        self.temp_label = None
        self.shuffle()
        
    def shuffle(self):
        if (self.shuffle_flag):

            np.random.shuffle(self.indexs)
        self.cursor = 0
    def next_batch(self, batch_size=10):
        end = False
        
        if batch_size > len(self.indexs):
            batch_size = len(self.indexs)
        end_cursor = self.cursor + batch_size
        if (end_cursor >= len(self.indexs)):
            end_cursor = len(self.indexs)
            end = True
        indexs = self.indexs[self.cursor: end_cursor]
        self.cursor = end_cursor
        
        batch_max_length = self.seq_lengths[indexs].max()
        self.temp_label = torch.tensor(self.cluster_labels[indexs])
        return self.seq_times[indexs, :batch_max_length], self.seq_types[indexs, :batch_max_length], \
                self.pad_masks[indexs, :batch_max_length], indexs, end, self.cluster_labels[indexs]
    
    def get_label(self):
        return self.temp_label
    def __getitem__(self, index):
        return (self.seq_times[index], self.seq_types[index], self.pad_masks[index])