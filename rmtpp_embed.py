import torch
import torch.nn as nn
import math

class timestampEmbedding(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        self.Wt = nn.Linear(1, h_dim)

    def forward(self, timestamps):
        
        
        time_emb = self.Wt(timestamps.unsqueeze(-1))

        return time_emb

class typeEmbedding(nn.Module):
    def __init__(self, h_dim, type_dim):
        super().__init__()
        self.type_emb = nn.Embedding(num_embeddings = type_dim + 1, embedding_dim = h_dim, padding_idx = type_dim)
    
    def forward(self, seq_types):
        type_emb = self.type_emb(seq_types)
        
        return type_emb

class eventEmbedding(nn.Module):
    def __init__(self, h_dim, type_dim):
        super().__init__()
        self.time_embedding = timestampEmbedding(h_dim)
        self.type_embedding = typeEmbedding(h_dim, type_dim)
    
    def forward(self, seq_t, seq_types):
        embedded_time = self.time_embedding(seq_t)
        embedded_type = self.type_embedding(seq_types)
        
        return embedded_time + embedded_type