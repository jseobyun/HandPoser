# HandPoser : Variational Hand Pose Prior

![hand_space](asset/hand_space.png)
## Description
This repository includes hand(mano) version of [VPoser](https://github.com/nghorbani/human_body_prior).

2021.12.02 first version updated. However, the code is not well trimmed yet. 

## Installation
**Requirements**
- Python 3.8
- Pytorch 1.11.0
- opencv-python
- pyrender
- chumpy
- pytorch-lightning
- dotmap

Clone this repo and run the following from the root folder:
```bash
python install -r requirements.txt
python setup.py develop
```

## Setting

Revise main/config.yaml file

***Essential***
- general.dataset_dir : the path of dataset directory that includes FreiHand and Interhand
- general.mano_dir : the path of dataset directory that includes mano pkl file

***Optional***
- train_params.*
- val_params.*
- model_params.*

  
## Data
```
$data_repository
        |-InterHand
        |   |-annots
        |       |-train
        |       |   |-InterHand2.6M_train_MANO_NeuralAnnot.json
        |       |   |- ...                
        |       |-test
        |       |   |-InterHand2.6M_test_MANO_NeuralAnnot.json
        |       |   |- ...
        |       |-val
        |           |-InterHand2.6M_val_MANO_NeuralAnnot.json
        |           |- ...             
        |-FreiHand   
            |-training_mano.json
            |- ...
```

## Usage
```python
import human_hand_prior as HPoser

hposer = HPoser.create()
latent_hand_pose = ... # torch [1, 32] tensor following gaussian distribution
hand_pose = hposer.decode(latent_hand_pose)['hand_pose']

```

## Result
![tsne_distr](asset/tsne_distr.png)
