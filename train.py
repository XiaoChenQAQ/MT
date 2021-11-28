import numpy as np
import torch
import os
from transformer import transformer

data_path = "./data/"
bpevocab_path = data_path+"bpevocab.txt"
train_de = data_path+"train_de.txt"
train_en = data_path+"train_en.txt"
valid_de = data_path+"valid_de.txt"
valid_en = data_path+"valid_en.txt"
print("file path:"+valid_en)

# make token to index 
list_token = []
with open(bpevocab_path,'r', encoding='utf-8') as f:
    for i,line in enumerate(f):
        list_token.append(line.split()[0])
    print(len(list_token))
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


# 构建自己的数据集
class Data(Data.Dataset):
    def __init__(self, de_train_data, en_train_data):
        self.de_train_data = de_train_data
        self.en_train_data = en_train_data

    def __getitem__(self, index):
        de_sentence = self.de_train_data[index]
        en_sentence = self.en_train_data[index]
        de_sentence = de_sentence.rstrip()
        en_sentence = en_sentence.rstrip()
        de_sentence = token.encoder_sentence([de_sentence])
        de_sentence = token.de_padding(de_sentence)
        en_sentence = token.encoder_sentence([en_sentence])
        en_sentence1 = [en_sentence[0][:-1]]
        en_sentence2 = [en_sentence[0][1:]]
        en_sentence1 = token.en_padding(en_sentence1)
        en_sentence2 = token.en_padding(en_sentence2)

        return np.int32(de_sentence[0, :]), np.int32(en_sentence1[0, :]), np.int64(en_sentence2[0, :])

    def __len__(self):
        return len(self.de_train_data)

def train():
    total_loss = 0
    total_sample = 0
    model.train()
    optim.zero_grad()
    ppl = 0
    training_epoch = 0
    for ind, samples in enumerate(tqdm(train_data)):  # Training
        samples = samples.to(device).get_batch()
        ind = ind + 1
        loss, logging_info = criteration(model, **samples)
        sample_size = logging_info["valid tokens num"]
        ppl += logging_info["ppl"]
        training_epoch += 1
        loss.backward()
        if ind % update_freq == 0:
            optim.step()
            scheduler.step()
            optim.zero_grad()
        total_loss += float(loss)
        total_sample += int(sample_size)

        if (ind // update_freq) % 100 == 0 and ind % update_freq == 0:
            print(
                f"Epoch: {epoch} Training loss: {float(total_loss) / total_sample} ppl: {ppl/training_epoch} lr: {float(optim.param_groups[0]['lr'])}"
            )
            total_loss = 0
            total_sample = 0
            ppl = 0
            training_epoch = 0

    with torch.no_grad():  # Validating
        total_loss = 0
        total_sample = 0
        model.eval()
        for samples in tqdm(valid_data):
            samples = samples.to(device).get_batch()
            loss, logging_info = criteration(model, **samples)
            sample_size = logging_info["valid tokens num"]
            ppl += logging_info["ppl"]
            training_epoch += 1
            total_loss += loss
            total_sample += sample_size
        print(
            f"Epoch: {epoch} Valid loss: {float(total_loss / total_sample)} ppl: {ppl/training_epoch}"
        )

    with open(os.path.join(save_dir, f"epoch{epoch}.pt"), "wb") as fl:
        torch.save(model, fl)



    