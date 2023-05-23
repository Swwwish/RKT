import copy
import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import debug_print
torch.set_printoptions(precision=4, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype('bool') #差异，源码k=0
    return torch.from_numpy(future_mask)

def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

def compute_corr_dict(qseqs, corr_dict):
    batch, seqlen = qseqs.shape[0], qseqs.shape[1]
    qseqs_cpu = qseqs.detach().cpu().numpy()
    corr= np.zeros((seqlen, seqlen))
    for i in range(batch):
        corr_temp= np.zeros((seqlen, seqlen))
        for j in range(seqlen):
            for k in range(seqlen):
                if qseqs_cpu[i][j] in corr_dict.keys() and qseqs_cpu[i][k] in corr_dict[qseqs_cpu[i][j]].keys():
                    corr_temp[j][k] = corr_dict[qseqs_cpu[i][j]][qseqs_cpu[i][k]]
        corr = np.concatenate((corr, corr_temp), axis=0)
    corr = np.reshape(corr, (batch+1, seqlen, seqlen))[1:,:,:]
    
    return corr

def compute_corr_matrix(qseqs, corr_matrix):
    batch, seqlen = qseqs.shape[0], qseqs.shape[1]
    qseqs_cpu = qseqs.detach().cpu().numpy()
    corr= np.zeros((seqlen, seqlen))
    for i in range(batch):
        corr_temp = corr_matrix[ np.ix_(qseqs_cpu[i], qseqs_cpu[i]) ]
        corr = np.concatenate((corr, corr_temp), axis=0)
    corr = np.reshape(corr, (batch+1, seqlen, seqlen))[1:,:,:]
    return corr

def computeTime(time_seq, time_span):
    batch_size = time_seq.shape[0]
    size = time_seq.shape[1]

    time_matrix= torch.abs(torch.unsqueeze(time_seq, axis=1).repeat(1,size,1).reshape((batch_size, size*size,1)) - \
                 torch.unsqueeze(time_seq,axis=-1).repeat(1, 1, size,).reshape((batch_size, size*size,1)))

    # time_matrix[time_matrix>time_span] = time_span
    time_matrix = time_matrix.reshape((batch_size,size,size))
    return time_matrix

def attention(query, key, value, rel, l1, l2, timestamp, mask=None, dropout=None):
    """Compute scaled dot product attention.
    """
    rel = rel * mask.to(torch.float) # future masking of correlation matrix.
    rel_attn = rel.masked_fill(rel == 0, -1e5)
    rel_attn = nn.Softmax(dim=-1)(rel_attn)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e32)
        time_stamp= torch.exp(-torch.abs(timestamp.float()))
        time_stamp=time_stamp.masked_fill(mask, -1e5)


    prob_attn = F.softmax(scores, dim=-1)
    time_attn = F.softmax(time_stamp, dim=-1)
    
    prob_attn = (1-l2)*prob_attn + l2*time_attn
    # prob_attn = F.softmax(prob_attn + rel_attn, dim=-1)

    prob_attn = (1-l1)*prob_attn + l1*rel_attn
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, rel, l1, l2, pos_key_embeds, pos_value_embeds, mask, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings
    
    scores = torch.matmul(query, key.transpose(-2, -1))     # BS, head, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    
    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key.transpose(-2, -1))
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    pad_zero = torch.zeros(bs, head, 1, 1, seqlen).to(device) ## 加了zero_pad
    prob_attn = torch.cat([pad_zero, prob_attn[:, :, 1:, :, :]], dim=2) # 第一行score置0
    
    
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, rel, l1, l2, timestamp, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        rel = rel.unsqueeze(1).repeat(1,self.num_heads,1,1)
        timestamp = timestamp.unsqueeze(1).repeat(1,self.num_heads,1,1)
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(query, key, value, rel, l1, l2, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, rel, l1, l2, timestamp, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn


class RKT(nn.Module):
    def __init__(self, num_c, num_items, embed_size, num_attn_layers, num_heads, batch_size, dataset_name,
                  max_pos, encode_pos, grad_clip, theta, drop_prob=0.1, time_span=100000, emb_type="qid", emb_path=""):
        """Self-attentive knowledge tracing.
        Arguments:
            num_items (int): number of questions
            num_c (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(RKT, self).__init__()
        self.model_name = "rkt"
        self.dataset_name = dataset_name
        self.emb_type = emb_type
        self.embed_size = embed_size
        self.encode_pos = encode_pos
        self.time_span = time_span
        self.grad_clip = grad_clip
        self.theta = theta
        
        if dataset_name in ["statics2011", "poj"]:
            self.item_embeds = nn.Embedding(num_c + 1, embed_size , padding_idx=0)
        else:
            self.item_embeds = nn.Embedding(num_items + 1, embed_size , padding_idx=0)
        # self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin_out = nn.Linear(embed_size, 1)
        self.l1 = nn.Parameter(torch.rand(1))
        self.l2 = nn.Parameter(torch.rand(1))

    def get_inputs(self, item_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        # skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, item_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs  
        inputs[..., self.embed_size:] *= 1 - label_inputs  
        return inputs

    def get_query(self, item_ids):
        item_ids = self.item_embeds(item_ids)
        # skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids], dim=-1)
        return query

    def forward(self, dcur, rel_dict, train=True):
        q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]  
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        timestamp = torch.cat((t[:,0:1], tshft), dim=1)
        #print(f"questions:{pid_data}, \n concepts:{q_data} \n {q_data.shape}")
        
        # dataset only with question, no concept
        if self.dataset_name in ["statics2011", "poj"]:
            inputs = self.get_inputs(q_data, target)
            query = self.get_query(q_data)
            rel = compute_corr_matrix(q_data, rel_dict)
        elif self.dataset_name in ["algebra2005", "bridge2006"]:
            inputs = self.get_inputs(pid_data, target)
            query = self.get_query(pid_data)
            rel = compute_corr_dict(pid_data, rel_dict)
        else:
            inputs = self.get_inputs(pid_data, target)
            query = self.get_query(pid_data)
            rel = compute_corr_matrix(pid_data, rel_dict)

        mask = future_mask(inputs.size(-2))
        inputs = F.relu(self.lin_in(inputs))
        time = computeTime(timestamp, self.time_span) #时间计算
        #debug_print(text = f"after timestamp",fuc_name="train_model")

        rel = np.where(rel < self.theta, 0, rel) #自己构建关系矩阵
        rel = torch.Tensor(rel).cuda()
        #debug_print(text = f"after compute_corr",fuc_name="train_model")
        if inputs.is_cuda:
            mask = mask.cuda()
        outputs, attn  = self.attn_layers[0](query, inputs, inputs, rel, self.l1, self.l2, time, self.encode_pos,
                                                   self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        
        for l in self.attn_layers[1:]:
            residual, attn = l(query, outputs, outputs, rel, self.l1, self.l2, time, 
                               self.encode_pos, self.pos_key_embeds,self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))
        out = self.lin_out(outputs).squeeze(-1)
        m = nn.Sigmoid()
        pred = m(out)
        
        return pred, attn
