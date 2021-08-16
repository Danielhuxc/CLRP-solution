#args:
#1. source file
#2. target file
#3. num of models
#4. model dir
#5. mode: 0=label data 1=5fold labels
#...

import sys
import numpy as np
from components.predict import get_single_model
import pandas as pd
import os

def main():
    ###
    # generate prediction for 1. inference 2. 5fold labels
    ###
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    num_of_models = int(sys.argv[3])
    model_dirs = [sys.argv[i+5] for i in range(num_of_models)]
    
    data = pd.read_csv(source_path)
    
    preds = []
    for i in range(num_of_models):
        preds.append(get_single_model(model_dirs[i],data))
        
    if sys.argv[4] == '1':
        #hard coded weight for when one of each of roberta and deberta is used to predict
        if num_of_models == 2 and 'roberta' in model_dirs[0] and 'deberta' in model_dirs[1]:
            preds_fold0 = [pred[0] for pred in preds]
            preds_fold1 = [pred[1] for pred in preds]
            preds_fold2 = [pred[2] for pred in preds]
            preds_fold3 = [pred[3] for pred in preds]
            preds_fold4 = [pred[4] for pred in preds]
            preds_0 = preds_fold0[0] * 0.33 + preds_fold0[1] * 0.67
            preds_1 = preds_fold1[0] * 0.33 + preds_fold1[1] * 0.67
            preds_2 = preds_fold2[0] * 0.33 + preds_fold2[1] * 0.67
            preds_3 = preds_fold3[0] * 0.33 + preds_fold3[1] * 0.67
            preds_4 = preds_fold4[0] * 0.33 + preds_fold4[1] * 0.67
        else:
            preds_0 = np.mean(np.concatenate([pred[0] for pred in preds],axis=1),axis=1)
            preds_1 = np.mean(np.concatenate([pred[1] for pred in preds],axis=1),axis=1)
            preds_2 = np.mean(np.concatenate([pred[2] for pred in preds],axis=1),axis=1)
            preds_3 = np.mean(np.concatenate([pred[3] for pred in preds],axis=1),axis=1)
            preds_4 = np.mean(np.concatenate([pred[4] for pred in preds],axis=1),axis=1)
        labeled_extra0 = data.copy()
        labeled_extra1 = data.copy()
        labeled_extra2 = data.copy()
        labeled_extra3 = data.copy()
        labeled_extra4 = data.copy()
        labeled_extra0['target'] = preds_0
        labeled_extra1['target'] = preds_1
        labeled_extra2['target'] = preds_2
        labeled_extra3['target'] = preds_3
        labeled_extra4['target'] = preds_4

        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        labeled_extra0.to_csv(target_path + 'labeled_extra_0.csv',index=False)
        labeled_extra1.to_csv(target_path + 'labeled_extra_1.csv',index=False)
        labeled_extra2.to_csv(target_path + 'labeled_extra_2.csv',index=False)
        labeled_extra3.to_csv(target_path + 'labeled_extra_3.csv',index=False)
        labeled_extra4.to_csv(target_path + 'labeled_extra_4.csv',index=False)
        
    else:
        preds = [np.expand_dims(np.mean(np.concatenate(pred,axis=1),axis=1),axis=1) for pred in preds]
        if num_of_models == 2 and 'roberta' in model_dirs[0] and 'deberta' in model_dirs[1]:
            pred = preds[0] * 0.33 + preds[1] * 0.67
        else:
            cat = np.concatenate(preds,axis=1)
            pred = np.mean(cat,axis=1)
        data['target'] = pred
        data.to_csv(target_path,index=False)
    
if __name__ == "__main__":
    main()
