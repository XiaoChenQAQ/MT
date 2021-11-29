import numpy as np
from torch.utils import data as Data
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
    def __init__(self, bpevocab_path="./data/bpevocab.txt") -> None:
        self.token2index , self.index2token = self.make_dict(bpevocab_path)
        

    def make_dict(self, bpevocab_path):
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
        
    def encoder_sentence(self, sentences, max_len=32):
        '''把句子编码并pad'''
        seq = []
        for sentence in sentences:
            encoder = []
            for word in sentence.rstrip().split(" "):
                if word not in self.token2index:
                    encoder.append(self.token2index["<unk>"])
                else:
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

    def padding(self, sentence, pad_idx=0, max_len=32):
        '''
        padding,统一input长度
        '''
        pad = np.zeros((len(sentence), max_len), dtype="int32")
        for i, seq in enumerate(sentence):
            seq_len = len(seq)
            if seq_len > max_len:
                pad[i, :] = seq[:max_len]
            else:
                pad[i, :seq_len] = seq
        return pad

    def depadding(self, sentence, pad_idx=0, max_len=32):
        '''
        depadding,
        '''
        ...
    
# 构建数据集
# 张译有很好的,但我看不懂
class MyData(Data.Dataset):
    def __init__(self, de_train_data, en_train_data, bpevocab):
        self.de_train_data = self._read_file(de_train_data)
        self.en_train_data = self._read_file(en_train_data)
        self.token = Tokenizer(bpevocab)

    def __getitem__(self, index):
        de_sentence = self.de_train_data[index]
        en_sentence = self.en_train_data[index]
        de_sentence = de_sentence.rstrip()
        en_sentence = en_sentence.rstrip()
        de_sentence = self.token.encoder_sentence([de_sentence])
        de_sentence = self.token.padding(de_sentence)
        en_sentence = self.token.encoder_sentence([en_sentence])
        #头截断和尾截断
        en_sentence1 = [en_sentence[0][:-1]]
        en_sentence2 = [en_sentence[0][1:]]
        en_sentence1 = self.token.padding(en_sentence1)
        en_sentence2 = self.token.padding(en_sentence2)

        return np.int32(de_sentence[0, :]), np.int32(en_sentence1[0, :]), np.int64(en_sentence2[0, :])

    def __len__(self):
        return len(self.de_train_data)
    def _read_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            rf = f.readlines()
        return rf

if __name__ == '__main__':
    print('hello')
    token = Tokenizer()
    data = token.encoder_sentence(["und molecul@@ fi@@ cht natürlich die wichtigste voraus@@ setzung von allen an , dass geschäft geschäft ist , und phil@@ anth@@ rop@@ ie ist das instrument der menschen , die die welt verändern wollen ."])
    data2 = token.encoder_sentence(["molecul@@ molecul@@ molecul@@ molecul@@ molecul@@ molecul@@ molecul@@"])
    print(data)
    print(data2)
    data = token.padding(data)
    
    print(data)