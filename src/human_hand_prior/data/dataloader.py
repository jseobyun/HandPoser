import os
import warnings
import torch
import torch.utils.data as TD
import numpy as np

def dataset_exists(dataset_name, dataset_dir, split_names=None):
    if dataset_name.lower() not in ['freihand', 'interhand']:
        return False
    if dataset_dir is None:
        return False
    if split_names is None:
        split_names = ['train', 'valid', 'test']
    if not isinstance(split_names, list) and isinstance(split_names, str):
        split_names = [split_names]

    if dataset_name.lower() == 'interhand':
        annot_dir = os.path.join(dataset_dir, 'InterHand', 'annots')
        train_dir = os.path.join(annot_dir, 'train')
        test_dir = os.path.join(annot_dir, 'test')
        valid_dir = os.path.join(annot_dir, 'valid')
        train_json = os.path.join(train_dir, 'InterHand2.6M_train_MANO_NeuralAnnot.json')
        test_json = os.path.join(test_dir, 'InterHand2.6M_test_MANO_NeuralAnnot.json')
        valid_json = os.path.join(valid_dir, 'InterHand2.6M_val_MANO_NeuralAnnot.json')
        if not os.path.exists(train_json) and 'train' in split_names:
            return False
        if not os.path.exists(test_json) and 'test' in split_names:
            return False
        if not os.path.exists(valid_json) and 'valid' in split_names:
            return False
        return True

    elif dataset_name.lower() == 'freihand':
        train_json = os.path.join(dataset_dir, 'FreiHand', 'training_mano.json')
        if not os.path.exists(train_json) and 'train' in split_names:
            return False
        if 'test' in split_names:
            warnings.warn("FreiHand dataset does not have test dataset")
            return False
        if 'valid' in split_names:
            warnings.warn("FreiHand dataset does not have validation dataset")
            return False
        return True
    ###
    else:
        return False

class HandPoserDataset(TD.Dataset):
    def __init__(self, dataset_names, split_name):
        assert split_name in ['train', 'test', 'valid']
        if not isinstance(dataset_names, list) and isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        for dataset_name in datset_names:
            assert dataset_exists(dataset_name, )

        self.dataset_names = dataset_names
        self.split_name = split_name
