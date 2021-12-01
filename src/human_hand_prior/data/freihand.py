import os
import cv2
import numpy as np
import torch
import torch.utils.data as TD

from src.human_hand_prior.utils.io_utils import load_json
from src.human_hand_prior.utils.vis_utils import vis_mano

class FreiHand(TD.Dataset):
    def __init__(self, dataset_dir, split_name):
        assert split_name == 'train'
        json_path = os.path.join(dataset_dir, 'FreiHand', 'training_mano.json')

        self.data = load_json(json_path)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        param = self.data[index][0]
        pose = np.array(param[:48]).reshape(16, 3)
        root_pose = pose[0:1, :].reshape(-1)
        hand_pose = pose[1:, :].reshape(-1)
        shape = np.array(param[48:58]).reshape(10)
        data = {
            'root_pose': root_pose.astype(np.float32),
            'hand_pose': hand_pose.astype(np.float32),
            'hand_shape': shape.astype(np.float32),
        }
        return data


if __name__ == '__main__':
    dataset_dir = '/home/jseob/Desktop/yjs/data'
    split_name = 'train'
    db = FreiHand(dataset_dir, split_name)

    for i, (root_pose, hand_pose, shape) in enumerate(db):
        print(i)
        canvas = vis_mano(hand_pose, shape)
        cv2.imshow('tmp', canvas)
        cv2.waitKey(10)