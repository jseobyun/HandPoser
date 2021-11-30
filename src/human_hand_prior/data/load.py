import os
import warnings
import torch
import torch.utils.data as TD
import numpy as np

from data.freihand import FreiHand
from data.interhand import InterHand

def dataset_exists(dataset_name, dataset_dir, split_names=None):
    if dataset_name.lower() not in ['freihand', 'interhand']:
        return False
    if dataset_dir is None:
        return False
    if split_names is None:
        split_names = ['train', 'val', 'test']
    if not isinstance(split_names, list) and isinstance(split_names, str):
        split_names = [split_names]

    if dataset_name.lower() == 'interhand':
        annot_dir = os.path.join(dataset_dir, 'InterHand', 'annots')
        train_dir = os.path.join(annot_dir, 'train')
        test_dir = os.path.join(annot_dir, 'test')
        val_dir = os.path.join(annot_dir, 'val')
        train_json = os.path.join(train_dir, 'InterHand2.6M_train_MANO_NeuralAnnot.json')
        test_json = os.path.join(test_dir, 'InterHand2.6M_test_MANO_NeuralAnnot.json')
        val_json = os.path.join(val_dir, 'InterHand2.6M_val_MANO_NeuralAnnot.json')
        if not os.path.exists(train_json) and 'train' in split_names:
            return False
        if not os.path.exists(test_json) and 'test' in split_names:
            return False
        if not os.path.exists(val_json) and 'val' in split_names:
            return False
        return True

    elif dataset_name.lower() == 'freihand':
        train_json = os.path.join(dataset_dir, 'FreiHand', 'training_mano.json')
        if not os.path.exists(train_json) and 'train' in split_names:
            return False
        if 'test' in split_names:
            warnings.warn("FreiHand dataset does not have test dataset")
            return False
        if 'val' in split_names:
            warnings.warn("FreiHand dataset does not have validation dataset")
            return False
        return True
    ###
    else:
        return False

def dataset_load(dataset_name, dataset_dir, split_name):
    if dataset_name.lower() == 'interhand':
        dataset = InterHand(dataset_dir, split_name)
    elif dataset_name.lower() =='freihand':
        dataset = FreiHand(dataset_dir, split_name)
    else:
        None


def dataloader


class HandPoserDataset(TD.Dataset):
    def __init__(self, dataset_names, dataset_dir, split_name):
        assert split_name in ['train', 'test', 'valid']
        if not isinstance(dataset_names, list) and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        for dataset_name in dataset_names:
            assert dataset_exists(dataset_name, dataset_dir, split_name)

        self.dataset_names = dataset_names
        self.dataset_dir = dataset_dir
        self.split_name = split_name

        data =