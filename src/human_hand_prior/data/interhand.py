import os
import cv2
import numpy as np
import torch
import torch.utils.data as TD

from src.human_hand_prior.utils.io_utils import load_json
from src.human_hand_prior.utils.vis_utils import vis_mano
class InterHand(TD.Dataset):
    def __init__(self, dataset_dir, split_name):
        annot_dir = os.path.join(dataset_dir, 'InterHand', 'annots')
        split_dir = os.path.join(annot_dir, split_name)
        json_path =  os.path.join(split_dir, 'InterHand2.6M_'+split_name+'_MANO_NeuralAnnot.json')
        data = []
        data_dict = load_json(json_path)
        for subject in list(data_dict.keys()):
            subj_data = data_dict[subject]
            for capture in list(subj_data.keys()):
                rhand = subj_data[capture]['right']
                lhand = subj_data[capture]['left']
                if rhand is not None:
                    data.append(rhand)
                if lhand is not None:
                    data.append(lhand)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        pose = np.array(sample['pose']).reshape(16, 3)
        root_pose = pose[0:1, :].reshape(-1)
        hand_pose = pose[1:, :].reshape(-1)
        shape = np.array(sample['shape']).reshape(10)

        data = {
            'root_pose': root_pose.astype(np.float32),
            'hand_pose': hand_pose.astype(np.float32),
            'hand_shape': shape.astype(np.float32),
        }
        return data



if __name__ == '__main__':
    dataset_dir ='/home/jseob/Desktop/yjs/data'
    split_name = 'train'
    db = InterHand(dataset_dir, split_name)

    for i, (root_pose, hand_pose, shape) in enumerate(db):
        print(i)
        canvas = vis_mano(hand_pose, shape)
        cv2.imshow('tmp', canvas)
        cv2.waitKey(10)