import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q: (B, h, T_q, d_k)
        K: (B, h, T_k, d_k)
        V: (B, h, T_k, d_v)
        mask: (B, 1, T_q, T_k) or broadcastable shape
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1)) # (B, h, T_q, T_k)

        # mask
        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V) # (B, h, T_q, d_v)

        return output, attn

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (B, T, d_model)
        mask: (B, 1, T_q, T_k) or broadcastable
        """
        B = Q.size(0)

        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        # split to multi heads (B, h, T, d_k)
        Q = Q.view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.h, self.d_v).transpose(1, 2)
        output, attn = self.attention(Q, K, V, mask) # (B, h, T_q, d_v)
        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.W_o(output) # (B, T_q, d_model)

        return out, attn

# Positional Embedding
class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term) 
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x (B, T, d_model)
        T = x.size(1)
        x = x + self.pe[:, :T, :]

        return x
    
# Feed Forward Network
class FFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x (B, T, d_model)
        x = self.fc2(F.relu(self.fc1(x)))
        return x
    
# Add && Norm
class AddNorm(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, out):
        return self.norm(x + self.dropout(out))

# Encorder
class EncorderLayer(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.multi_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)


    def forward(self, x, mask=None):
        attn_out, _ = self.multi_attn(x, x, x, mask)  # (B, T, d_model)
        x = self.add_norm1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)
        return x
    
 # Decorder
# masking
def padding_mask(x, pad_idx=0):
    # x (B, T)
    return (x != pad_idx).unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)

def future_mask(size, device):
    mask = torch.tril(torch.ones(size, size), device=device).bool() 
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

def decorder_mask(x):
    pad_mask = padding_mask(x)
    ft_mask = future_mask(x.size(1), device=x.device)
    return pad_mask & ft_mask

class DecorderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = AddNorm(d_model, dropout)
        self.ffn = FFN(d_model, d_ff)
        self.norm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x, self_attn_out)
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x, cross_attn_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x, ffn_out)

        return x
    
# Encorder Stack
class Encorder(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_heads=8, dropout=0.1, pad_idx=0, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pad_idx = pad_idx
        self.pos_encorder = PositionalEncoding(d_model=d_model)
        self.dropaut = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncorderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)])
    
    def forward(self, src, src_mask=None):
        # (B, T_src)
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encorder(x)
        x = self.dropaut(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

# Decorder Stack
class Decorder(nn.Module):
    def __init__(self, vocab_size, d_model=512, d_ff=2048, num_heads=8, dropout=0.1, pad_idx=0, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pad_idx = pad_idx
        self.pos_encorder = PositionalEncoding(d_model=d_model)
        self.dropaut = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecorderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        # tgt (B, T_tgt)
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encorder(x)
        x = self.dropaut(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        
        return x

# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        src_pad_idx=0,
        tgt_pad_idx=0,
        dropout=0.1
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.encorder = Encorder(src_vocab_size, d_model, d_ff, num_heads, dropout, src_pad_idx, num_layers)
        self.decorder = Decorder(tgt_vocab_size, d_model, d_ff, num_heads, dropout, tgt_pad_idx, num_layers)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def make_src_mask(self, src):
        # (B,1,1,T_src)
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T_tgt)
        seq_len = tgt.size(1)
        subsequent_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=tgt.device)
        ).bool().unsqueeze(0).unsqueeze(0)  # (1,1,T_tgt,T_tgt)
        return pad_mask & subsequent_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src) # (B, 1, 1, T_src)
        tgt_mask = self.make_tgt_mask(tgt) # (B, 1, T_tgt, T_tgt)
        enc_out = self.encorder(src, src_mask)
        dec_out = self.decorder(tgt, enc_out, tgt_mask, src_mask)
        out = self.fc_out(dec_out)  # (B, T_tgt, tgt_vocab_size)

        return out

# test
if __name__ == "__main__":
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    src_pad_idx = 0
    tgt_pad_idx = 0

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=2,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        dropout=0.1
    )

    src = torch.randint(1, src_vocab_size, (4, 10))  # (B=4, T_src=10)
    tgt = torch.randint(1, tgt_vocab_size, (4, 12))  # (B=4, T_tgt=12)

    out = model(src, tgt)
    print(out.shape)  # (4, 12, 1000)

    nn.MultiheadAttention()