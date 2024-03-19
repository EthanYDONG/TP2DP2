import torch
import torch.nn as nn
import torch.nn.functional as F
from rmtpp_embed import eventEmbedding


class RMTPP(nn.Module):
    
    def __init__(self, h_dim, type_dim, device):
        super(RMTPP, self).__init__()
        
        self.h_dim = h_dim
        self.type_dim = type_dim
        self.device = device
        
        self.encode_emb = eventEmbedding(h_dim, type_dim)
        self.encoder = nn.LSTM(input_size=self.h_dim,
                           hidden_size=self.h_dim,
                           num_layers=1,
                           batch_first=True)
        
        self.intensity_v = nn.Linear(h_dim, 1, bias = False)
        self.intensity_wb = nn.Linear(1, 1)
        self.type_class = nn.Linear(self.h_dim, self.type_dim + 1)
        self.type_loss = nn.CrossEntropyLoss(reduction = 'none', ignore_index = self.type_dim)
        self.forward_logger = None
        
    
    def encode(self, seq_t, seq_types):
        
        if len(seq_t.shape) != 1:
            batch_size, seq_length = seq_t.shape
        else:
            batch_size = 1
            seq_length = seq_t.shape[0]

        
        event_emb = self.encode_emb(seq_t, seq_types)
        event_encode = self.encoder(event_emb)
        
        return event_encode

    
    def forward(self, seq_t, seq_types, pad_masks):
        
        if len(seq_t.shape) != 1:
            batch_size, seq_length = seq_t.shape
        else:
            batch_size = 1
            seq_length = seq_t.shape[0]
        
        past_encoding, _ = self.encode(seq_t, seq_types)
         
        
        w = self.intensity_wb.weight
        b = self.intensity_wb.bias
       
        past_effect = self.intensity_v(past_encoding[:, :-1, :]).squeeze()  
        delta_t = seq_t[:, 1:] - seq_t[:, :-1]
        intens_term = past_effect + self.intensity_wb(delta_t[:, :, None]).squeeze() + torch.exp(past_effect + b)/w \
                    - torch.exp(past_effect + self.intensity_wb(delta_t[:, :, None]).squeeze())/w
        intens_term = (intens_term * pad_masks[:, 1:])
        
       
        raw_score = self.type_class(past_encoding[:, :-1, :]) 
        softmax_term = - self.type_loss(raw_score.reshape(-1, self.type_dim + 1), seq_types[:, 1:].reshape(-1)).reshape(batch_size, -1)
        softmax_term = (softmax_term * pad_masks[:, 1:])
       
        log_like =  (intens_term + softmax_term).sum(-1)
        
        pred_types = torch.argmax(raw_score, dim=-1)
        valid_mask = seq_types[:, 1:] != self.type_dim
        correct_preds = (pred_types == seq_types[:, 1:]) & valid_mask
        num_correct_per_sample = torch.sum(correct_preds.int(), dim=1)
        num_valid_per_sample = torch.sum(valid_mask.int(), dim=1)
     
        
        return - log_like, num_correct_per_sample, num_valid_per_sample