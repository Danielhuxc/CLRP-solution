import random
import numpy as np
import os
import torch
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    
def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=num_splits)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

def generate_config(model_type,pretrained_path,save_path,lr_type,lr_setting):
    config = {'model_dir': './pretrained/roberta-large/',
              'n_folds': 5,
              'num_epoch': 3,
              'weight_decay': 0.01,
              'head_lr': 1e-4,
              'weight_lr': 5e-2,
              'base_lr': 7e-5,
              'min_lr': 2e-5,
              'low_lr': 1e-5,
              'warm_up_ratio': 0.06,
              'decline_1': 0.15,
              'decline_2': 0.6,
              'decline_3': 0.7,
              'decline_4': 0.75,
              'layerwise_decay_rate': 0.875**0.5,
              'seed_': 88888888,
              'reini_head':False,
              'only_val_in_radius': True,
              'save_center': 330,
              'save_radius': 5,
              'betas': (0.9, 0.999),
         }
    config['pretrained_path'] = pretrained_path
    config['save_path'] = save_path
    config['lr_type'] = lr_type
    if model_type == 'ro':
        config['model_dir'] = './pretrained/roberta-large/'
        config['batch_size'] = 16
        config['accumulation_steps'] = 1
        config['pseudo_save_name'] = 'roberta_large_single.pt'
    elif model_type == 'de':
        config['model_dir'] = './pretrained/deberta-large/'
        config['batch_size'] = 8
        config['accumulation_steps'] = 2
        config['save_center'] = 660
        config['save_radius'] = 10
        config['pseudo_save_name'] = 'deberta_large_single.pt'
    
    if lr_setting == '2':
        config['num_epoch'] = 2
        config['head_lr']= 1e-5
        config['weight_lr']= 5e-3
        config['base_lr']= 7e-6
        config['min_lr']= 2e-6
        config['low_lr']= 1e-6
    elif lr_setting == '3':
        config['head_lr']= 5e-5
        config['weight_lr']= 2e-3
        config['base_lr']= 3e-5
        config['min_lr']= 1e-5
        config['low_lr']= 5e-6
    
    return config
