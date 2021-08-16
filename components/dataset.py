import torch
import pandas as pd

class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, texts, tokenizer):
        self.is_train = is_train
        self.tokenizer = tokenizer
        if self.is_train:
            self.data = texts
        else:
            self.data = texts
        ### only use portion of data
        length = int(len(self.data)/1)
        self.data = self.data[:length]
        ###

    def __getitem__(self, idx):
        item = self.tokenizer(self.data[idx], padding='max_length', is_split_into_words = False,truncation=True, return_tensors="pt")
        
        item['labels'] = item['input_ids'].clone()
        
        probability_matrix = torch.full(item['labels'].shape, 0.15)
        special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in item['labels'].tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        item['labels'][~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(item['labels'].shape, 0.8)).bool() & masked_indices
        item['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(item['labels'].shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), item['labels'].shape, dtype=torch.long)
        item['input_ids'][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        item['input_ids'] = item['input_ids'][0]
        item['attention_mask'] = item['attention_mask'][0]
        item['labels'] = item['labels'][0]
        return item

    def __len__(self):
        return len(self.data)
    
class CLRPDataset_finetune(torch.utils.data.Dataset):
    def __init__(self, is_train, fold, train_data, tokenizer):
        self.is_train = is_train
        self.tokenizer = tokenizer
        
        if is_train:
            df = train_data.query(f"kfold != {fold}")[['excerpt','target']]
        else:
            df = train_data.query(f"kfold == {fold}")[['excerpt','target']]
        self.excerpt = df['excerpt'].to_numpy()
        self.target = df['target'].to_numpy()

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.excerpt[idx],return_tensors='pt',
                              max_length=256,
                              padding='max_length',truncation=True)
        
        item = {}
        item['input_ids'] = tokenized['input_ids'][0]
        item['attention_mask'] = tokenized['attention_mask'][0]
        item['target'] = torch.tensor(self.target[idx]).type(torch.float32)
        
        return item

    def __len__(self):
        return len(self.target)
    
class CLRPDataset_pred(torch.utils.data.Dataset):
    def __init__(self,df,tokenizer):
        self.excerpt = df['excerpt'].to_numpy()
        self.tokenizer = tokenizer
    
    def __getitem__(self,idx):
        encode = self.tokenizer(self.excerpt[idx],return_tensors='pt',
                                max_length=256,
                                padding='max_length',truncation=True)
        encoded = {'input_ids':encode['input_ids'][0],
                   'attention_mask':encode['attention_mask'][0]
                  }
        
        return encoded
    
    def __len__(self):
        return len(self.excerpt)
    
class CLRPDataset_pseudo(torch.utils.data.Dataset):
    def __init__(self, is_train, label_path, train_data, tokenizer):
        self.tokenizer = tokenizer
        if is_train:
            df1 = pd.read_csv(label_path+"labeled_extra_0.csv")
            df2 = pd.read_csv(label_path+"labeled_extra_1.csv")
            df3 = pd.read_csv(label_path+"labeled_extra_2.csv")
            df4 = pd.read_csv(label_path+"labeled_extra_3.csv")
            df5 = pd.read_csv(label_path+"labeled_extra_4.csv")
            self.excerpt = df1['excerpt'].to_numpy()
            self.target = (df1['target'] + df2['target'] + df3['target'] + df4['target'] + df5['target']).to_numpy()/5
        else:
            self.excerpt = train_data['excerpt'].to_numpy()
            self.target = train_data['target'].to_numpy()

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.excerpt[idx],return_tensors='pt',
                              max_length=256,
                              padding='max_length',truncation=True)
        
        item = {}
        item['input_ids'] = tokenized['input_ids'][0]
        item['attention_mask'] = tokenized['attention_mask'][0]
        item['target'] = torch.tensor(self.target[idx]).type(torch.float32)
        
        return item

    def __len__(self):
        return len(self.target)

#reads 5 fold labeled data and mix 3x training data in
class CLRPDataset_pseudo_5fold(torch.utils.data.Dataset):
    def __init__(self, is_train, fold, train_data, tokenizer, label_path):
        self.tokenizer = tokenizer
        if is_train:
            df = pd.read_csv(label_path+f"labeled_extra_{fold}.csv")
            tr = train_data.query(f"kfold != {fold}")[['excerpt','target']]
            df = pd.concat([df,tr,tr,tr], ignore_index=True)
            df = df.sample(frac=1).reset_index(drop=True)
        else:
            df = train_data.query(f"kfold == {fold}")[['excerpt','target']]
        self.excerpt = df['excerpt'].to_numpy()
        self.target = df['target'].to_numpy()
        ###

    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.excerpt[idx],return_tensors='pt',
                              max_length=256,
                              padding='max_length',truncation=True)
        
        item = {}
        item['input_ids'] = tokenized['input_ids'][0]
        item['attention_mask'] = tokenized['attention_mask'][0]
        item['target'] = torch.tensor(self.target[idx]).type(torch.float32)
        
        return item

    def __len__(self):
        return len(self.target)
