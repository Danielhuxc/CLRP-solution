import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from components.util import init_params

class Custom_bert(nn.Module):
    def __init__(self,model_dir):
        super().__init__()

        #load base model
        config = AutoConfig.from_pretrained(model_dir)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.base = AutoModel.from_pretrained(model_dir, config=config)  
        
        dim = self.base.encoder.layer[0].output.dense.bias.shape[0]
        
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)
        
        #weights for weighted layer average
        n_weights = 24
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
        #attention head
        self.attention = nn.Sequential(
            nn.Linear(1024, 1024),            
            nn.Tanh(),
            nn.Linear(1024, 1),
            nn.Softmax(dim=1)
        ) 
        self.cls = nn.Sequential(
            nn.Linear(dim,1)
        )
        init_params([self.cls,self.attention])
        
    def reini_head(self):
        init_params([self.cls,self.attention])
        return 
        
    def forward(self, input_ids, attention_mask):
        base_output = self.base(input_ids=input_ids,
                                      attention_mask=attention_mask)

        #weighted average of all encoder outputs
        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in base_output['hidden_states'][-24:]], dim=0
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(0)
    
        #multisample dropout
        logits = torch.mean(
            torch.stack(
                [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        return self.cls(logits)
