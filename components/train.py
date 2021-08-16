from components.dataset import CLRPDataset_finetune, CLRPDataset_pseudo, CLRPDataset_pseudo_5fold
from components.util import seed_everything, create_folds, generate_config
from components.model import Custom_bert
from components.optimizer import get_optimizer, get_scheduler
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import os
import gc
gc.enable()

def run_fold_ft(fold,config,train_data,tokenizer,t_bar):
    device = "cuda:0"
    #prep train/val datasets
    train_dataset = CLRPDataset_finetune(True, fold,train_data,tokenizer)
    val_dataset = CLRPDataset_finetune(False, fold,train_data,tokenizer)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
    
    total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'])
    val_step = 1
    min_valid_loss = np.inf
    
    #load model
    model = Custom_bert(config['model_dir']).to(device)
    _ = model.eval()

    model.load_state_dict(torch.load(config['pretrained_path']), strict=False)

    #get optimizer and scheduler
    optimizer = get_optimizer(model,config)
    lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

    step = 0
    min_step = 0
    last_save_step = 0
    last_save_index = 0

    #seed_everything(seed=config['seed_'] + fold)

    optimizer.zero_grad()
    for epoch in range(config['num_epoch']):
        model.train()
        count = 0
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            outputs = model(input_ids, attention_mask)

            cls_loss = nn.MSELoss()(torch.squeeze(outputs,1),target)

            loss = cls_loss / config['accumulation_steps']

            total_loss+=torch.pow(nn.MSELoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']

            loss.backward()

            if (count+1) % config['accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                count = 0
                total_loss = 0
            else:
                count+=1

            #only save in radius of certain step
            if step >= (config['save_center']-config['save_radius']) and step <= (config['save_center']+config['save_radius']):
                val_step = 1
            do_val = True
            if config['only_val_in_radius']:
                if step < (config['save_center']-config['save_radius']) or step > (config['save_center']+config['save_radius']):
                    do_val = False

            if ((step+1) % val_step == 0 and count == 0) and do_val:
                model.eval()
                l_val = nn.MSELoss(reduction='sum')
                with torch.no_grad():
                    total_loss_val = 0
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        outputs = model(input_ids, attention_mask)

                        cls_loss_val = l_val(torch.squeeze(outputs),batch['target'].to(device))

                        val_loss = cls_loss_val

                        total_loss_val+=val_loss.item()
                    total_loss_val/=len(val_dataset)
                    total_loss_val = total_loss_val**0.5

                    if min_valid_loss > total_loss_val and step >= (config['save_center']-config['save_radius']) and step <= (config['save_center']+config['save_radius']):
                        #saves model with lower loss
                        min_step = step
                        min_valid_loss = total_loss_val
                        #print("min loss updated to ",min_valid_loss," at step ",min_step)
                        if not os.path.isdir('./models'):
                            os.mkdir('./models')
                        if not os.path.isdir(config['save_path']):
                            os.mkdir(config['save_path'])
                        if 'roberta' in config['model_dir']:
                            torch.save(model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt')
                        else:
                            torch.save(model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                model.train()
            step+=1
            t_bar.update(1)
    del model,train_dataset,train_loader,val_dataset,val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return min_valid_loss, min_step

def train_ft(config):
    seed_everything(config['seed_'])
    
    train_data = pd.read_csv("./data/train.csv")
    train_data = create_folds(train_data, num_splits=5)
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)
    
    t_bar = tqdm(total=((2834*0.8//config['batch_size'])+1)*config['num_epoch']*config['n_folds'])
    train_losses = []
    for i in range(config['n_folds']):
        loss, m_step = run_fold_ft(i,config,train_data,tokenizer,t_bar)
        train_losses.append(loss)
    return train_losses

def train_pseudo(config, label_path):
    device = "cuda:0"
    seed_everything(config['seed_'])
    train_data = pd.read_csv("./data/train.csv")
    train_data = create_folds(train_data, num_splits=5)
    
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)
    
    train_dataset = CLRPDataset_pseudo(True,label_path,train_data,tokenizer)
    t_bar = tqdm(total=((len(train_dataset)//config['batch_size'])+1)*config['num_epoch'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    val_dataset = CLRPDataset_pseudo(False,label_path,train_data,tokenizer)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'])
    val_step = 100*config['accumulation_steps']
    min_valid_loss = np.inf

    model = Custom_bert(config['model_dir']).to(device)
    _ = model.eval()

    if config['pretrained_path'] not in [None,'None']:
        print(model.load_state_dict(torch.load(config['pretrained_path']), strict=False))

    optimizer = get_optimizer(model,config)
    lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

    step = 0
    min_step = 0
    last_save_step = 0
    last_save_index = 0

    optimizer.zero_grad()
    for epoch in range(config['num_epoch']):
        model.train()
        count = 0
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            outputs = model(input_ids, attention_mask)

            cls_loss = nn.MSELoss()(torch.squeeze(outputs,1),target)

            loss = cls_loss / config['accumulation_steps']

            total_loss+=torch.pow(nn.MSELoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']
            loss.backward()

            if (count+1) % config['accumulation_steps'] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                count = 0
                total_loss = 0
            else:
                count+=1

            if ((step+1) % val_step == 0):
                l_val = nn.MSELoss(reduction='sum')
                with torch.no_grad():
                    model.eval()
                    total_loss_val = 0
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        outputs = model(input_ids, attention_mask)

                        cls_loss_val = l_val(torch.squeeze(outputs),batch['target'].to(device))

                        val_loss = cls_loss_val

                        total_loss_val+=val_loss.item()
                    total_loss_val/=len(val_dataset)
                    total_loss_val = total_loss_val**0.5

                if min_valid_loss > total_loss_val:
                    min_step = step
                    min_valid_loss = total_loss_val
                    #print("min loss updated to ",min_valid_loss," at step ",min_step)
                    # Saving State Dict
                    if not os.path.isdir(config['save_path']):
                        os.mkdir(config['save_path'])
                    torch.save(model.state_dict(), config['save_path'] + config['pseudo_save_name'])
                model.train()
            step+=1
            t_bar.update(1)
    del model,train_dataset,train_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return min_valid_loss

def train_pseudo_5fold(config, label_path):
    device = "cuda:0"
    seed_everything(config['seed_'])

    train_data = pd.read_csv("./data/train.csv")
    train_data = create_folds(train_data, num_splits=5)
    model_dir = config['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)

    min_val_losses = []
    for fold in range(config['n_folds']):
        train_dataset = CLRPDataset_pseudo_5fold(True,fold,train_data,tokenizer,label_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        
        val_dataset = CLRPDataset_pseudo_5fold(False,fold,train_data,tokenizer,label_path)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

        if fold == 0:
            t_bar = tqdm(total=((len(train_dataset)*5//config['batch_size'])+1)*config['num_epoch'])
            
        total_train_steps = int(len(train_loader) * config['num_epoch'] / config['accumulation_steps'])
        val_step = 100*config['accumulation_steps']
        min_valid_loss = np.inf

        model = Custom_bert(config['model_dir']).to(device)
        _ = model.eval()

        if config['pretrained_path'] not in [None,'None']:
            model.load_state_dict(torch.load(config['pretrained_path']), strict=False)

        optimizer = get_optimizer(model,config)
        lr_scheduler = get_scheduler(optimizer,total_train_steps,config)

        step = 0
        min_step = 0
        last_save_step = 0
        last_save_index = 0

        optimizer.zero_grad()
        for epoch in range(config['num_epoch']):
            model.train()
            count = 0
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target = batch['target'].to(device)

                outputs = model(input_ids, attention_mask)

                cls_loss = nn.MSELoss()(torch.squeeze(outputs,1),target)

                loss = cls_loss / config['accumulation_steps']

                total_loss+=torch.pow(nn.MSELoss()(torch.squeeze(outputs,1),target),0.5).item() / config['accumulation_steps']
                loss.backward()

                if (count+1) % config['accumulation_steps'] == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    count = 0
                    total_loss = 0
                else:
                    count+=1

                if ((step+1) % val_step == 0):
                    model.eval()
                    l_val = nn.MSELoss(reduction='sum')
                    with torch.no_grad():
                        total_loss_val = 0
                        for batch in val_loader:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            outputs = model(input_ids, attention_mask)

                            cls_loss_val = l_val(torch.squeeze(outputs),batch['target'].to(device))

                            val_loss = cls_loss_val

                            total_loss_val+=val_loss.item()
                        total_loss_val/=len(val_dataset)
                        total_loss_val = total_loss_val**0.5

                    if min_valid_loss > total_loss_val and epoch > 0:
                        min_step = step
                        min_valid_loss = total_loss_val
                        if not os.path.isdir('./models'):
                            os.mkdir('./models')
                        if not os.path.isdir(config['save_path']):
                            os.mkdir(config['save_path'])
                        if 'roberta' in config['model_dir']:
                            torch.save(model.state_dict(), config['save_path']+f'roberta_large_{fold}.pt')
                        else:
                            torch.save(model.state_dict(), config['save_path']+f'deberta_large_{fold}.pt')
                    model.train()
                step+=1
                t_bar.update(1)
        del model,train_dataset,train_loader,val_dataset,val_loader
        gc.collect()
        torch.cuda.empty_cache()
        min_val_losses.append(min_valid_loss)
    return min_val_losses
