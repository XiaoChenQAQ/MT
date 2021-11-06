# 机器翻译作业
作者:尹张森,2101839
本项目使用pytorch框架实现[transformer](https://arxiv.org/abs/1706.03762)

## 数据集

使用德语-英语2014数据集,使用双字节编码分词

```

```

## Transformer总体结构
![Transformer结构图](./src/Transformer.jpeg)
主要包括下列层
- 嵌入层和位置编码
- 自注意力表示层(多头注意力机制)
- 前馈神经网络
- 残差连接和层标准化

## 嵌入层和位置编码层
嵌入层和位置编码层将编码器输入和解码器输入序列变成向量表示.
###嵌入层
使用pytorch的Embedding()函数实现嵌入
###位置编码
位置编码是一个固定的矩阵大小是(序列长度seq_len,嵌入向量长度emb_size)
Transformer中使用的是不同频率的三角函数
$$ PE(pos,2i)=sin(\frac{pos}{10000^{2i/emb\_size}})$$
$$ PE(pos,2i+1)=cos(\frac{pos}{10000^{2i/emb\_size}})$$

## 基于点乘的多头注意力机制
多头注意力机制就是在原来点乘注意力机制的基础上,把原来d_model长度的向量切分成heads份,运算后在连接起来.它的好处是允许模型在不同的表示子空间里学习.在很多实验里发现,不同的表示空间的头捕获的信息是不同的.
对于上一层输入的X(batch_size, seq_len, d_model),使用线性变换(没有激活函数)分别映射成QKV(batch_size, seq_len, d_model).需要注意在解码器中QKV的来源不用,Q来源于源语言,KV是目标语言的仿射变换.
之后对Q,K,V进行切分,切分的参数矩阵维度分别是(heads, d_model, d_k ),(heads, d_model, d_k )(heads, d_model, d_v );(d_k=d_v=d_model/heads),这样切分后的qkv向量进行运算后在连接起来可以获得一个(1,heads*d_k)的向量并且(heads*d_k=d_model),对输出向量右乘一个输出矩阵W(d_model,d_model)获得最终多头注意力机制的score.
最终输出的矩阵O大小应为(batch_size, seq_len, d_model),并且为了方便显示注意力机制也可以输出一个注意力权重矩阵.
**Remark**:在pytorch源码中qkv的切分参数矩阵只有一个
###可能存在的问题: 
- 目标语言和源语言输入seq_len不一定一致

