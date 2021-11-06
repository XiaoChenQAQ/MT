import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter

class EmdAndPos(nn.Module):
    '''
    处理好的句子序列,并给他加上position编码
    参数(emb_size=d_model, seq_len, dict_number, padding_idx)
    输入:(batch_size, seq_len)
    输出:(batch_size, seq_len, emb_size)
    test:
    input1 = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    net = nn.Sequential(EmdAndPos(16,4,10))
    print(net(input1))
    '''
    def __init__(self,emb_size,seq_len,dict_number):
        super(EmdAndPos, self).__init__()
        self.emb = nn.Embedding(num_embeddings=dict_number, embedding_dim=emb_size, padding_idx=0)
        self.pos = self._position(emb_size,seq_len)

    def _position(self,emb_size,seq_len):
        '''
        pos只与位置有关,没有学习过程,句子中的每一个单词产生一个描述位置的与词嵌入等长的向量,整个句子产生一个(seq_len,emb_size)的矩阵
        输入:处理好的句子序列
        输出:输出(seq_len , emb_size)的矩阵
        '''
        PE = np.zeros((seq_len,emb_size))
        def func(pos,i,emb_size):
            if i%2 ==0:
                return math.sin(pos / (10000**(1.0*i/emb_size)))
            else:
                return math.cos(pos / (10000**(1.0*(i-1)/emb_size)))
        for xy,val in np.ndenumerate(PE):
            PE[xy]= func(xy[0], xy[1], emb_size)
        pos_matrix = torch.from_numpy(PE)
        return pos_matrix

    def forward(self, inputs):
        X = self.emb(inputs)
        # print(self.pos)
        # print("*"*80)
        # print(X)
        return X+self.pos

class MultiHeadAttention(nn.Module):
    ''' 
    基于点乘的多头注意力层;
    Q的维度(L,d_k),V的维度(L,d_k),V的维度(L,d_v);d_k,d_v分别表示key和value的大小,通常设置d_k=d_v=d_model
    输入:(batch_size, seq_len, d_model)
    输出:
    '''
    def __init__(self, seq_len,heads, d_model, d_k=None, d_v=None, dropout=0.1,decode=False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model if not d_k else d_k
        self.d_v = d_model if not d_v else d_v
        self.heads = heads
        self.head_dim = d_model // heads
        assert self.head_dim * heads == d_model ,"heads必须能整除d_model"
        self.Q = nn.Linear(d_model,d_k, bias=False)
        self.K = nn.Linear(d_model,d_k, bias=False)
        self.V = nn.Linear(d_model,d_v, bias=False)
        self.W_Q = nn.Parameter(data=torch.tensor(heads, d_model, d_k//heads),requires_grad=True)
        self.W_K = nn.Parameter(data=torch.tensor(heads, d_model, d_k//heads),requires_grad=True)
        self.W_V = nn.Parameter(data=torch.tensor(heads, d_model, d_v//heads),requires_grad=True)
        self.register_parameter('multihead_proj_weight', None)
        self.outputlinear = nn.Linear(d_k,d_model)
        #self.W_O = nn.Parameter(data=torch.tensor())
        if not decode:
            self.mask = None
        else:
            self.mask = torch.mask_fill(self._make_mask(seq_len), value=float("-inf"))
        self.softmax = nn.Softmax(dim=1)
            

    def _make_mask(self, dim):
        matirx = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matirx))
        return mask==1
    
    def _dotmulAtt(self, q, k, v):
        '''
        q,k,v向量点乘注意力
        q,k,v输入维度(batch_size,seq_len,head_dim)
        返回维度:(batch_size,seq_len, head_dim)
        '''
        return torch.bmm(self.softmax(torch.bmm(q,k.permute(0,2,1)) / math.sqrt(self.d_k) +self.mask),v)

    
    def forward(self, source, target, ):
        # 将XY仿射变换成QKV
        Q = self.Q(source)
        K = self.K(target)
        V = self.V(target)
        #多头点积注意力
        headlist = []
        for i in range(self.heads):
            head_i = self._dotmulAtt(Q @ self.W_Q[i], K @ self.W_K[i], V @ self.W_V[i])
            headlist.append(head_i)
        #连接后矩阵大小是(batch_size,seq_len,head_dim*heads)
        output = self.outputlinear(torch.cat(headlist))
        #经过线性层输出大小(batch_size,seq_len, d_model)
        return output

# class dotProdAttention(nn.Module):

#残差网络和层标准化
class AddNorm(nn.Module):
    def __init__(self, dropout=0.1, pre_norm = True):
        super(AddNorm, self).__init__()
        #self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
        

    def forward(self,x, sub_layer,**kwargs):
        if self.pre_norm:
            layer_norm = nn.LayerNorm(x.size()[1:])
            out = layer_norm(x)
            sub_output = sub_layer(out,**kwargs)
            out = self.dropout(sub_output)
            return out + x
        else:
            sub_output = sub_layer(x,**kwargs)
            x = self.dropout(x + sub_output)
            layer_norm = nn.LayerNorm(x.size()[1:])
            out = layer_norm(x)
            return out

#前馈神经网络
class Feed_Forward(nn.Module):
    def __init__(self,d_model,hidden_dim=2048):
        super(Feed_Forward, self).__init__()
        self.L1 = nn.Linear(d_model,hidden_dim)
        self.L2 = nn.Linear(hidden_dim,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.relu(self.L1(x))
        output = self.L2(output)
        return output

#编码器结构
class Encoder(nn.Module):
    '''
    Encoder:编码器
    输入:batch_size的句子list
    输出:(batch_size, seq_len, d_model)
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        


    def forward(self,x): 

        

        return output