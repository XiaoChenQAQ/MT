import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.emb_size = emb_size
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
        #新的tensor要注意device
        pos_matrix = torch.from_numpy(PE).to(device)
        return pos_matrix

    def forward(self, inputs):
        X = self.emb(inputs)
        # print("emb")
        # print(X.dtype,self.pos.dtype)
        # print((X+self.pos.float()).dtype)
        # print("*"*80)
        X = X * math.sqrt(self.emb_size)
        return X+self.pos.float()


class MultiHeadAttention(nn.Module):
    ''' 
    基于点乘的多头注意力层;
    Q的维度(L,d_k),V的维度(L,d_k),V的维度(L,d_v);d_k,d_v分别表示key和value的大小,通常设置d_k=d_v=d_model
    输入:(batch_size, seq_len, d_model)
    输出:(batch_size, seq_len, d_model)
    问题:dropout不知道在哪加
    '''
    def __init__(self, seq_len,heads, d_model, d_k=None, d_v=None, dropout=0.1, decode=False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model if not d_k else d_k
        self.d_v = d_model if not d_v else d_v
        self.heads = heads
        self.head_dim = d_model // heads
        self.seq_len = seq_len
        assert self.head_dim * heads == d_model ,"heads必须能整除d_model"
        self.Q = nn.Linear(d_model,d_k, bias=False)
        self.K = nn.Linear(d_model,d_k, bias=False)
        self.V = nn.Linear(d_model,d_v, bias=False)
        # 这样写的权重好像不参与训练,改成mxnet里的实现了
        # self.W_Q = nn.Parameter(data=torch.tensor(heads, d_model, d_k//heads),requires_grad=True)
        # self.W_K = nn.Parameter(data=torch.tensor(heads, d_model, d_k//heads),requires_grad=True)
        # self.W_V = nn.Parameter(data=torch.tensor(heads, d_model, d_v//heads),requires_grad=True)
        # self.register_parameter('multihead_proj_weight', None)
        self.dropout = nn.Dropout(dropout)
        self.outputlinear = nn.Linear(d_k,d_model)
        self.decode = decode
        #解码器需要future-mask
        if not decode:
            self.mask = None
        else:
            self.mask = self._make_mask(seq_len).to(device)
        self.softmax = nn.Softmax(dim=-1)
            

    def _make_mask(self, dim):
        matirx = np.ones((dim, dim))
        mask = torch.Tensor(np.tril(matirx))
        return mask==0

    def _dotmulAtt(self, q, k, v, mask):
        '''
        q,k,v向量点乘注意力
        q,k,v输入维度(batch_size * heads,seq_len,head_dim)
        返回维度:(batch_size * heads ,seq_len, head_dim)
        '''
        d = q.shape[-1]

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        # print("dot_att_scores")
        # print(scores[0])
        # print("*"*80)
        self.attention_weights = self.softmax(scores.masked_fill(mask.unsqueeze(1).expand((-1, self.heads, -1, -1)).reshape(-1, self.seq_len, self.seq_len), value=float("-inf")))
        
        # print("dot_att_weights")
        # #print(scores.masked_fill(mask.unsqueeze(1).expand((-1, self.heads, -1, -1)).reshape(-1, self.seq_len, self.seq_len), value=float("-inf"))[0])
        # print(self.attention_weights[0])
        # print("*"*80)
        return torch.bmm(self.dropout(self.attention_weights), v)

    def _transpose_qkv(self, X, num_heads): 
        '''qkv变换分片,以引用多头注意力机制'''
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)      
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])
    def _transpose_output(self, X, num_heads):
        '''逆变换,使output和输入shape相同'''
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    
    def get_attention_weights(self):
        return self.attention_weights
    
    def forward(self, q, k, v, mask):
        # print("att_in")
        # print(q)
        # print("*"*80)
        # 将XY仿射变换成QKV
        # t=self.Q(q.float())
        # print(t.dtype,t.shape)
        # print(len(t))
        # print(t[0].type, t[1].type)
        # print("*"*80)
        Q = self._transpose_qkv(self.Q(q), self.heads)
        K = self._transpose_qkv(self.K(k), self.heads)
        V = self._transpose_qkv(self.V(v), self.heads)
        # print("att_in")
        # print(Q)
        # print("*"*80)
        #点积注意力,mask好像有bug
        if self.decode:
            #解码器,有future_mask和padding_mask
            output = self._dotmulAtt(Q, K, V, mask|self.mask)
        else:
            #编码器,只有padding_mask
            output = self._dotmulAtt(Q, K, V, mask)
        #concat
        output_concat = self._transpose_output(output, self.heads)
        # print("att_out")
        # print(output_concat[0])
        # print("*"*80)
        return self.outputlinear(output_concat)


#残差网络和层标准化
# class AddNorm(nn.Module):
#     def __init__(self, dropout=0.1, pre_norm = True):
#         super(AddNorm, self).__init__()
#         #self.dropout = nn.Dropout(dropout)
#         self.pre_norm = pre_norm
#     def forward(self,x, sub_layer,**kwargs):
#         if self.pre_norm:
#             layer_norm = nn.LayerNorm(x.size()[1:])
#             out = layer_norm(x)
#             sub_output = sub_layer(out,**kwargs)
#             out = self.dropout(sub_output)
#             return out + x
#         else:
#             sub_output = sub_layer(x,**kwargs)
#             x = self.dropout(x + sub_output)
#             layer_norm = nn.LayerNorm(x.size()[1:])
#             out = layer_norm(x)
#             return out
#网上的一个更好的代码实现
#不用将模块作为参数输入AddNorm,输入tensor,通过两次调用AddNorm实现块中的残差运算
class AddNorm(nn.Module):
    '''
    normalized_shape应当等于(seq_len, d_model)
    '''
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # print("AddNorm")
        # print(X.shape,Y.shape)
        # print(X.dtype,Y.dtype)
        # print("*"*80)
        return self.ln(self.dropout(Y) + X)



#前馈神经网络,不过好像可以直接放在Encoder里
class Feed_Forward(nn.Module):
    def __init__(self,d_model,hidden_dim=1024):
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
    Encoder:编码器模块,调用前面写的所有类,tensorshape不变
    输入:batch_size的句子list
    输出:(batch_size, seq_len, d_model)
    '''
    def __init__(self, seq_len,heads, d_model, norm_shape, d_k=None, d_v=None, dropout=0.1,decode=False, hidden_dim=1024, **kwargs):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(seq_len,heads, d_model, d_k, d_v, dropout,decode)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.feedforward = Feed_Forward(d_model, hidden_dim)
        self.addnorm2 = AddNorm(norm_shape, dropout)        


    def forward(self, x, mask): 
        y = self.addnorm1(x, self.attention(x, x, x, mask))
        output = self.addnorm2(y, self.feedforward(y))
        return output

#组成一个TransformerEncoder
class TransformerEncoder(nn.Module):
    '''
    Transformer编码器
    '''
    def __init__(self, n_layers, dict_number, seq_len, heads, d_model, norm_shape, dropout=0.1,decode=False, hidden_dim=1024):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embpos = EmdAndPos(d_model, seq_len, dict_number)
        self.blocks = nn.Sequential()
        for i in range(n_layers):
            self.blocks.add_module(
                "block" + str(i),
                Encoder(seq_len=seq_len,heads=heads, d_model=d_model, norm_shape=norm_shape, dropout=dropout,decode=decode, d_k=hidden_dim,d_v=hidden_dim))
    
    def forward(self, X, mask,*args):
        X = self.embpos(X)
        # print(X.dtype,X.shape)
        # print("*"*88)
        self.attention_weights = [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            X = blk(X, mask)
            self.attention_weights[i] = blk.attention.attention_weights
        return X


#解码器
class Decoder(nn.Module):
    '''解码器和编码器类似'''
    def __init__(self, seq_len,heads, d_model, norm_shape, dropout=0.1,decode=True, hidden_dim=1024, **kwargs):
        super(Decoder, self).__init__()
        self.attention1 = MultiHeadAttention(seq_len,heads, d_model, d_k=hidden_dim, d_v=hidden_dim, dropout=dropout,decode=decode)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(seq_len,heads, d_model, d_k=hidden_dim, d_v=hidden_dim, dropout=dropout, decode=decode)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.feedforward = Feed_Forward(d_model, hidden_dim)
        self.addnorm3 = AddNorm(norm_shape, dropout) 

    def forward(self, X, enc_outputs, x_padding_mask, enc_padding_mask):
        '''
        Encoder-Decoder部分中需要输入encoder的tensor,而且不同的block对应不同,训练阶段和预测阶段也不同
        '''
        Y = self.addnorm1(X, self.attention1(X, X, X, x_padding_mask))
        # “编码器－解码器”注意力。
        # `enc_outputs` 的开头: (`batch_size`, `num_steps`, `d_model`)
        if enc_outputs is not None:
            Y2 = self.attention2(Y, enc_outputs, enc_outputs,x_padding_mask|enc_padding_mask)
            Y = self.addnorm2(Y, Y2)

        return self.addnorm3(Y, self.feedforward(Y))

    
#类似的,再写一个解码器
class TransformerDecoder(nn.Module):
    '''Transformer解码器'''
    def __init__(self, n_layers, dict_number, seq_len, heads, d_model, norm_shape, d_k=None, d_v=None, dropout=0.1,decode=True, hidden_dim=1024, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_layers = n_layers
        self.embpos = EmdAndPos(d_model, seq_len, dict_number)
        self.blocks = nn.Sequential()
        for i in range(n_layers):
            self.blocks.add_module(
                "block" + str(i),
                Decoder(seq_len=seq_len,heads=heads, d_model=d_model, norm_shape=norm_shape, dropout=dropout,decode=decode, hidden_dim=hidden_dim))
        self.dense = nn.Linear(d_model, dict_number)

    
    def forward(self, X, enc_outputs, x_padding_mask, enc_padding_mask):
        X = self.embpos(X)
        #print(X)
        self._attention_weights = [[None] * len(self.blocks) for _ in range(2)]
        for i, blk in enumerate(self.blocks):
            X = blk(X, enc_outputs, x_padding_mask, enc_padding_mask)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention_weights
        # print("decoder")
        # print(self.dense(X))
        # print("*"*80)
        return self.dense(X)

    @property
    def attention_weights(self):
        return self._attention_weights



class Transformer(nn.Module):
    '''好像就是一个encoder-decoder
    1.增加padding_mask
    
    '''
    
    def __init__(self,n_layers, dict_number, seq_len, heads, d_model, norm_shape,padding_idx=0, dropout=0.1,decode=True, hidden_dim=1024, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.encoder = TransformerEncoder(n_layers=n_layers, dict_number=dict_number, seq_len=seq_len, heads=heads, d_model=d_model, norm_shape=norm_shape, dropout=dropout,decode=False, hidden_dim=hidden_dim, **kwargs)
        self.decoder = TransformerDecoder(n_layers=n_layers, dict_number=dict_number, seq_len=seq_len, heads=heads, d_model=d_model, norm_shape=norm_shape, dropout=dropout,decode=False, hidden_dim=hidden_dim, **kwargs)
        self._model_init()
    
    def _make_padding_mask(self, seq, seq_len, pad=0):
        '''把idx的做成padding_mask'''
        mask = (seq==pad)
        mask.bool()
        mask = mask.unsqueeze(1).expand((-1, seq_len, -1))
        return mask
    def _model_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, enc_seq, dec_seq, *args):
        enc_mask = self._make_padding_mask(enc_seq, self.seq_len)
        enc_mask = enc_mask.to(device)
        enc_outputs = self.encoder(enc_seq,enc_mask, *args)
        enc_mask = enc_mask.to(device)
        dec_mask = self._make_padding_mask(dec_seq, self.seq_len)
        return self.decoder(dec_seq, enc_outputs, dec_mask, enc_mask)

if __name__ == '__main__':
    import numpy as np

    model = Transformer(6, 6, 6,4,512,[6,512])
    model.to(device)
    a = np.array([[1, 2, 3, 4, 5, 0], [2, 3, 4, 2, 1, 0]])
    a = torch.IntTensor(a).to(device)

    b = np.array([[2, 2, 2, 2, 0, 0], [3, 3, 3, 3, 3, 0]])
    b = torch.IntTensor(b).to(device)
    #print(model)
    out = model(a, b)
    print(out)
    print(out.shape)