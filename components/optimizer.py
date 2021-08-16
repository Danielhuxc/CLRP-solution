from transformers import AdamW
import torch

def get_optimizer(model,config):
    # divide encoder layers into 3 groups and assign different lr
    # head lr is set separately
    layers = len(model.base.encoder.layer)
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr_head = ["layer_weights"]
    ### not in high_lr_head
    params_lst = [{'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.base.encoder.layer.named_parameters())
                             and not any(nd in n for nd in no_decay)
                             and not any(nd in n for nd in high_lr_head)], 
                   'lr': config['head_lr'],
                   'weight_decay': config['weight_decay']
                  }]
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.base.encoder.layer.named_parameters())
                             and any(nd in n for nd in no_decay)
                             and not any(nd in n for nd in high_lr_head)], 
                       'lr': config['head_lr'],
                       'weight_decay': 0.0
                      })
    ###
    ### in high_lr_head
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.base.encoder.layer.named_parameters())
                             and not any(nd in n for nd in no_decay)
                             and any(lw in n for lw in high_lr_head)], 
                   'lr': config['weight_lr'],
                   'weight_decay': config['weight_decay']
                  })
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.base.encoder.layer.named_parameters())
                             and any(nd in n for nd in no_decay)
                             and any(lw in n for lw in high_lr_head)], 
                       'lr': config['weight_lr'],
                       'weight_decay': 0.0
                      })
    ###
    parts = 3
    for i,j in zip(range(layers-1,-1,-int(layers/parts)),range(0,layers,int(layers/parts))):
        for k in range(int(layers/parts)):
            param_dict1 = {'params': [p for n, p in model.base.encoder.layer[i-k].named_parameters()
                                      if not any(nd in n for nd in no_decay)],
                           'weight_decay': config['weight_decay'],
                           'lr':pow(config['layerwise_decay_rate'],j)*config['base_lr']
                          }
            param_dict2 = {'params': [p for n, p in model.base.encoder.layer[i-k].named_parameters()
                                      if any(nd in n for nd in no_decay)],
                           'weight_decay': 0.0,
                           'lr':pow(config['layerwise_decay_rate'],j)*config['base_lr']
                          }
            params_lst.append(param_dict1)
            params_lst.append(param_dict2)
            
    optimizer = AdamW(params_lst, betas = config['betas'])
    
    return optimizer

def get_optimizer_robertaMLM(model,config):
    layers = len(model.roberta.encoder.layer)
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr_head = ["layer_weights"]
    ### not in high_lr_head
    params_lst = [{'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.roberta.encoder.layer.named_parameters())
                             and not any(nd in n for nd in no_decay)
                             and not any(nd in n for nd in high_lr_head)], 
                   'lr': config['head_lr'],
                   'weight_decay': config['weight_decay']
                  }]
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.roberta.encoder.layer.named_parameters())
                             and any(nd in n for nd in no_decay)
                             and not any(nd in n for nd in high_lr_head)], 
                       'lr': config['head_lr'],
                       'weight_decay': 0.0
                      })
    ###
    ### in high_lr_head
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.roberta.encoder.layer.named_parameters())
                             and not any(nd in n for nd in no_decay)
                             and any(lw in n for lw in high_lr_head)], 
                   'lr': config['base_lr'],
                   'weight_decay': config['weight_decay']
                  })
    params_lst.append({'params':[p for n, p in model.named_parameters() 
                             if not any(en in n for en,ep in model.roberta.encoder.layer.named_parameters())
                             and any(nd in n for nd in no_decay)
                             and any(lw in n for lw in high_lr_head)], 
                       'lr': config['base_lr'],
                       'weight_decay': 0.0
                      })
    ###
    parts = 3
    for i,j in zip(range(layers-1,-1,-int(layers/parts)),range(0,layers,int(layers/parts))):
        for k in range(int(layers/parts)):
            param_dict1 = {'params': [p for n, p in model.roberta.encoder.layer[i-k].named_parameters()
                                      if not any(nd in n for nd in no_decay)],
                           'weight_decay': config['weight_decay'],
                           'lr':pow(config['layerwise_decay_rate'],j)*config['base_lr']
                          }
            param_dict2 = {'params': [p for n, p in model.roberta.encoder.layer[i-k].named_parameters()
                                      if any(nd in n for nd in no_decay)],
                           'weight_decay': 0.0,
                           'lr':pow(config['layerwise_decay_rate'],j)*config['base_lr']
                          }
            params_lst.append(param_dict1)
            params_lst.append(param_dict2)
            
    optimizer = AdamW(params_lst, betas = config['betas'])
    
    return optimizer

def get_scheduler(optimizer, total_train_steps, config):
    #two schedules:
    #1. custom is similar to linear decay with warmup
    #2. 3stage is simply halving every 1/3 steps
    def lr_lambda_1(step):
        total_steps = total_train_steps
        w = int(config['warm_up_ratio']*total_steps)
        d1 = int(config['decline_1']*total_steps)
        d2 = int(config['decline_2']*total_steps)
        d3 = int(config['decline_3']*total_steps)
        d4 = int(config['decline_4']*total_steps)
        min_vs_base_ratio = config['min_lr']/config['base_lr']
        low_vs_base_ratio = config['low_lr']/config['base_lr']
        if step <= w:
            return step/w
        elif step <= d1:
            return 1
        elif step <= d3:
            return max(min_vs_base_ratio,min_vs_base_ratio+(1-min_vs_base_ratio)*(d2-step)/(d2-d1))
        else:
            return max(low_vs_base_ratio,low_vs_base_ratio+(min_vs_base_ratio-low_vs_base_ratio)*(d4-step)/(d4-d3))
    def lr_lambda_2(step):
        if step <= total_train_steps * (1/3):
            return 1
        if step <= total_train_steps * (2/3):
            return 0.5
        if step <= total_train_steps * (3/3):
            return 0.25
    if config['lr_type'] == 'custom':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_1)
    elif config['lr_type'] == '3stage':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_2)
