import numpy as np

class Tokenizer:
    '''
    需要完成的方法:
    1.读取bpe表生成word2idx_dict,idx2word_dict
    2.句子编码
    3.句子解码,用上面的字典idx2word
    4.padding,padding后输入进
    5.depadding
    4.debpe过程
    '''
    def __init__(self, bpevocab_path) -> None:
        self.token2index , self.index2token = self.make_dict(bpevocab_path)
        

    def make_dict(bpevocab_path):
        '''
        用bpe分词生成token2index字典
        输入:bpe分词文件路径
        输出:两个字典token2index,index2token典
        '''
        list_token = []
        with open(bpevocab_path,'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                list_token.append(line.split()[0])
        token2index = {token:i+4 for i,token in enumerate(list_token)}
        index2token = {i+4:token for i,token in enumerate(list_token)}
        token2index["<pad>"] = 0
        index2token[0] = "<pad>"
        token2index["<unk>"] = 1
        index2token[1] = "<unk>"
        token2index["<sos>"] = 2
        index2token[2] = "<sos>"
        token2index["<eos>"] = 3
        index2token[3] = "<eos>"
        print("from token list(len({0}),make a dict(len({1})))".format(len(list_token),len(token2index)))
        return token2index,index2token
        
    def encoder_sentence(self, sentences):
        seq = []
        for sentence in sentences:
            encoder = []
            for word in sentence.rstrip().split(" "):
                    encoder.append(self.token2index[word])
            encoder.append(self.token2index["<eos>"])
            encoder.insert(0, self.token2index["<sos>"])
            seq.append(encoder)
        return seq
    
    def decoder_sentence(self, seq):
        '''
        list格式的句子解码,用于debpe
        '''
        sentences= []
        for text in seq:
            sentence = []
            for index in text:
                sentence.append(self.index2token[index])
            sentences.append(sentence)

        return sentences
    
    def debpe(self, seq):
        '''
        将输出的seq变成可读的句子
        '''
        sentences_list = self.decoder_sentence(seq)
        ...

    def padding(self, sentence, pad_num=0, max_len=512):
        '''
        padding,统一input长度
        '''
        mask = np.zeros((len(sentence), max_len), dtype="int32")
        for i, seq in enumerate(sentence):
            seq_len = len(seq)
            if seq_len > max_len:
                mask[i, :] = seq[:max_len]
            else:
                mask[i, :seq_len] = seq
        return mask

    def depadding(self, sentence, pad_num=0, max_len=512):
        '''
        depadding,
        '''
        mask = np.zeros((len(sentence), max_len), dtype="int32")        
        for i, seq in enumerate(sentence):
            seq_len = len(seq)
            if seq_len > max_len:
                mask[i, :] = seq[:max_len]
            else:
                mask[i, :seq_len] = seq
        return mask

