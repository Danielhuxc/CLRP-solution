import pandas as pd
from transformers import AutoTokenizer, RobertaForMaskedLM, AutoConfig
from transformers import Trainer, TrainingArguments
from components.util import seed_everything
from components.dataset import MLMDataset
from components.optimizer import get_optimizer_robertaMLM, get_scheduler
import torch
import os
import sys

def main():
    ###
    # MLM pretrain with training data
    ###
    device = "cuda:0"
    model_dir = './pretrained/roberta-large/'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256, add_prefix_space=True)
    model = RobertaForMaskedLM.from_pretrained(model_dir, local_files_only=True).to(device)

    df = pd.read_csv('./data/train.csv')[['excerpt']]
    texts = df['excerpt'].tolist()
    df_val = pd.read_csv('./data/test.csv')[['excerpt']]
    test = df_val['excerpt'].tolist()
    texts = texts+test

    seed_everything(456982)

    train_dataset = MLMDataset(True,texts,tokenizer)
    val_dataset = MLMDataset(True,texts,tokenizer)

    config = {
        'lr_type':'custom',
        'base_lr':9e-5,
        'head_lr':1.2e-4,
        'min_lr':4e-5,
        'low_lr':2e-5,
        'n_epoch':5,
        'bs':16,
        'ga':1,
        'lr_scheduler_mul_factor':2,
        'weight_decay':0.01,
        'warm_up_ratio':0.2,
        'decline_1': 0.2,
        'decline_2': 0.7,
        'decline_3': 0.8,
        'decline_4': 0.9,
        'layerwise_decay_rate': 0.9**0.5,
        'betas': (0.9,0.993),
    }

    train_len = len(train_dataset)
    total_train_steps = int(train_len * config['n_epoch'] / config['ga'] / config['bs']) 
    optimizer = get_optimizer_robertaMLM(model,config)
    lr_scheduler = get_scheduler(optimizer, total_train_steps, config)

    training_args = TrainingArguments(
        output_dir='./',          # output directory
        num_train_epochs=config['n_epoch'],              # total number of training epochs
        overwrite_output_dir=True,
        per_device_train_batch_size=config['bs'],  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        weight_decay=0.01,               # strength of weight decay
        logging_strategy='no',
        gradient_accumulation_steps = config['ga'],
        save_strategy = "no",
        evaluation_strategy= 'epoch',
        prediction_loss_only=True,
        learning_rate = config['base_lr'],
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        optimizers = (optimizer, lr_scheduler)
    )

    trainer.train()
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    dict_ = model.state_dict()
    for key in list(dict_.keys()):
        dict_[key.replace('roberta.', 'base.')] = dict_.pop(key)
    torch.save(dict_, f'./models/roberta_large_pretrain.pt')
    
if __name__ == "__main__":
    main()
