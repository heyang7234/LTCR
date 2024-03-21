import torch
import torch.nn.functional as F

LARGE_NUM = 1e9

# def simclr_loss(hidden_features_transform_1, hidden_features_transform_2, temperature=1.0, normalize=True, weights=1.0):
#     batch_size = hidden_features_transform_1.size(0)

#     h1 = hidden_features_transform_1
#     h2 = hidden_features_transform_2

#     if normalize:
#         h1 = F.normalize(h1, dim=1)
#         h2 = F.normalize(h2, dim=1)

#     labels = torch.arange(batch_size)
#     masks = F.one_hot(torch.arange(batch_size), batch_size)

#     logits_aa = torch.matmul(h1, h1.transpose(1, 2)) / temperature
#     logits_aa = logits_aa - masks * LARGE_NUM
#     logits_bb = torch.matmul(h2, h2.transpose(1, 2)) / temperature
#     logits_bb = logits_bb - masks * LARGE_NUM
#     logits_ab = torch.matmul(h1, h2.transpose(1, 2)) / temperature
#     logits_ba = torch.matmul(h2, h1.transpose(1, 2)) / temperature

#     logits_a = torch.cat([logits_ab, logits_aa], dim=1)
#     logits_b = torch.cat([logits_ba, logits_bb], dim=1)

#     loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    
#     loss_a = loss_function(logits_a, labels)
#     loss_b = loss_function(logits_b, labels)
#     loss = loss_a + loss_b

#     return loss

# 
def simclr_loss(z1, z2): # 
    # 
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    # 
    z = torch.cat([z1, z2], dim=0)  # 2B x O x C
    z = z.transpose(0, 1)  # O x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # O x 2B x C * O  x C x 2B =   O x 2B x 2B
    
    # 
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # O x 2B x (2B-1), left-down side, remove last zero column
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # O x 2B x (2B-1), right-up side, remove first zero column
    
    # 
    logits = -F.log_softmax(logits, dim=-1)  # log softmax do dividing and log
    
    # i = 0,1,2,...,B-1
    i = torch.arange(B, device=z1.device)

    # origin
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss