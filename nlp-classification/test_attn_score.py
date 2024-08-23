import torch

num_heads = 2
seq_len = 10
hidden_size = 128
head_dim = hidden_size // num_heads
batch_size = 3

# ---------- 1. asymmetrical case ----------
print('---------- 1. asymmetrical case ----------')

attn_score = torch.rand(batch_size, num_heads, seq_len, seq_len) * 100
print('Greater than 0')
print((attn_score > 0).sum().item())
print(attn_score.numel())

bool_mask = torch.zeros([batch_size, seq_len])
attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,num_heads,1,-1])
attn_score = attn_score.masked_fill(attention_mask_expanded==0, -1e9)

for idx in range(batch_size):
    l = torch.randint(seq_len, (1,)).item()
    bool_mask[idx][:l] = 1

a = 1

N_R = attn_score.sum(-1)  # normalize row
N_C = attn_score.sum(-2)

# method 1
K_tilde_1 = torch.diag_embed(N_R**(-a)) @ attn_score @ torch.diag_embed(N_C**(-a))

# method 2
K_tilde_2 = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_C**(-a)).unsqueeze(-2)

print('K_tilde equivalence')
print((K_tilde_1 == K_tilde_2).sum().item())
print(K_tilde_1.numel())

print('\n')

# ---------- 2. symmetrical case ----------
print('---------- 2. symmetrical case ----------')

asym = torch.rand(batch_size, num_heads, seq_len, seq_len) * 100
attn_score = asym + asym.transpose(2,3)
print('Greater than 0 and symmetry')
print((attn_score > 0).sum().item())
print((attn_score == attn_score.transpose(2,3)).sum().item())
print(attn_score.numel())

bool_mask = torch.zeros([batch_size, seq_len])
attention_mask_expanded = bool_mask.unsqueeze(1).unsqueeze(2).expand([-1,num_heads,1,-1])
#attn_score = attn_score.masked_fill(attention_mask_expanded==0, -1e9)
"""
Conventional masking only applied to the keys, thus it will cause the attn_score to be asymmetrical in general
"""

for idx in range(batch_size):
    l = torch.randint(seq_len, (1,)).item()
    bool_mask[idx][:l] = 1

a = 1

N_R = attn_score.sum(-1)  # normalize row
N_C = attn_score.sum(-2)

print('Row and col equivalence')
print((N_R==N_C).sum().item())
print(N_R.numel())

# method 1
K_tilde_1 = torch.diag_embed(N_R**(-a)) @ attn_score @ torch.diag_embed(N_R**(-a))

# method 2
K_tilde_2 = (N_R**(-a)).unsqueeze(-1) * attn_score * (N_R**(-a)).unsqueeze(-2)

print('K_tilde equivalence')
print((K_tilde_1 == K_tilde_2).sum().item())
print(K_tilde_1.numel())