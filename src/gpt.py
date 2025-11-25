import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        
        # Combined QKV projection
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'], config['n_ctx']))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y

class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = GPT2MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.wpe = nn.Embedding(config['n_ctx'], config['n_embd'])
        
        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config['n_layer'])])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config['n_embd'])
        
    def forward(self, idx):
        B, T = idx.size()
        
        # Get embeddings
        tok_emb = self.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.wpe(pos)
        
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = x @ self.wte.weight.T
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config['n_ctx'] else idx[:, -self.config['n_ctx']:]
            
            # Forward pass
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx