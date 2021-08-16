from components.dataset import CLRPDataset_pred
from components.model import Custom_bert
import numpy as np
import torch
import gc
gc.enable()
from tqdm import tqdm
from transformers import AutoTokenizer
device = "cuda:0"

def run_fold(fold_num,model_path,data):
    if 'roberta' in model_path:
        model_dir = './pretrained/roberta-large/'
        model_name = f"roberta_large_{fold_num}.pt"
    elif 'deberta' in model_path:
        model_dir = './pretrained/deberta-large/'
        model_name = f"deberta_large_{fold_num}.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=256)    
    model = Custom_bert(model_dir).to(device)
    _ = model.eval()
    model.load_state_dict(torch.load(model_path+model_name))
    
    test_ds = CLRPDataset_pred(data,tokenizer)
    test_dl = torch.utils.data.DataLoader(test_ds,
                                          batch_size = 192,
                                          shuffle=False,
                                          pin_memory=True)
    
    pred = []
    with torch.no_grad():
        for batch in tqdm(test_dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            pred.extend(output.detach().cpu().numpy())
    
    del model, test_dl, test_ds
    gc.collect()
    torch.cuda.empty_cache()
            
    return np.array(pred)

def get_single_model(pth,data):
    pred0 = run_fold(0,pth,data)
    pred1 = run_fold(1,pth,data)
    pred2 = run_fold(2,pth,data)
    pred3 = run_fold(3,pth,data)
    pred4 = run_fold(4,pth,data)
    
    return [pred0,pred1,pred2,pred3,pred4]
