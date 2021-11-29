import numpy as np
import torch
import os
from torch.utils import data as Data
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from dataprocess import MyData,Tokenizer
from transformer import Transformer
from tqdm import tqdm
import math
import sys



#GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#数据路径
data_path = "./data/"
bpevocab_path = data_path+"bpevocab.txt"
train_de = data_path+"train_de.txt"
train_en = data_path+"train_en.txt"
valid_de = data_path+"valid_de.txt"
valid_en = data_path+"valid_en.txt"
print("file path:"+valid_en)

#结果保存
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = sys.stdout
        self.log = open(filename,'a')

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def train():
    sys.stdout = Logger("training.log")
    testloss = 0
    batch_size = 512
    lr = 0.0001
    dict_number = 10148
    epochs = 15
    n_layers = 6
    seq_len = 32
    heads =4
    d_model = 512
    norm_shape = [seq_len, d_model]
    #创建数据
    train_data = MyData(train_de, train_en, bpevocab_path)
    train_data = Data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True)
    valid_data = MyData(valid_de, valid_en, bpevocab_path)
    valid_data = Data.DataLoader(valid_data, shuffle=False, batch_size=batch_size, num_workers=0, pin_memory=True)

    # 创建网络
    model = Transformer(n_layers, dict_number, seq_len, heads, d_model, norm_shape)
    # model.load_state_dict(torch.load("de_to_en_2.pkl", map_location=torch.device(device)))

    #训练阶段
    model.train()
    print("*"*80)
    print("training")
    model.to(device)
    torch.cuda.empty_cache()
    # 初始化参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss(ignore_index=0)
    nb = len(train_data)
    # old_loss_time = None

    all_accuracy = 0
    all_step = 0
    for epoch in range(1, epochs):
        old_loss_time = None
        pbar = tqdm(train_data, total=nb)
        optimizer.zero_grad()
        for step, (x, x_y, y) in enumerate(pbar):
            all_step += 1
            x = x.long()
            y = y.long()
            x_y = x_y.long()
            x = x.to(device)
            x_y = x_y.to(device)
            y = y.to(device)
            y = y.view(-1)
            logits = model(x, x_y)
            adjust_learning_rate(512, all_step, optimizer)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            logits = logits.view(-1, logits.shape[-1])
            all_accuracy += calculate_accuracy(logits, y)
            loss = loss_fc(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if old_loss_time is None:
                loss_time = loss
                old_loss_time = loss
            else:
                old_loss_time += loss
                loss_time = old_loss_time / (step + 1)
            
            s = (
                "train ===> epoch: {} ---- step: {} ---- loss: {:.4f} ---- loss_time: {:.4f} ---- train_accuracy: {:.4f}% ---- lr: {:.6f}".format(
                    epoch, step, loss,
                    loss_time, all_accuracy / (step + 1), lr))
            pbar.set_description(s)

            if ((step + 1) % 5 == 0):
                torch.save(model.state_dict(), "./weights/de2en_5epoch.pkl")

        #eval,算验证集上的效果
        model.eval()
        validstep = 0
        total_loss = 0
        count_loss = 0
        valid_accuracy = 0
        valid_result = []
        with torch.no_grad():
            for x, x_y, y in valid_data:
                validstep += 1
                x = x.long()
                y = y.long()
                x_y = x_y.long()
                x = x.to(device)
                x_y = x_y.to(device)
                y = y.to(device)
                y = y.view(-1)
                valid_output = model(x, x_y)
                valid_output = valid_output.view(-1, valid_output.shape[-1])
                valid_accuracy += calculate_accuracy(valid_output, y)
                valid_loss = loss_fc(valid_output, y)
                total_loss += valid_loss.item()
                count_loss += 1
            if testloss > float(total_loss / count_loss):
                testloss = float(total_loss / count_loss)
                torch.save(model.state_dict(), "./weights/de_to_en_test.pkl")
            print(f'\nValidating at epoch', '%04d:' % (epoch + 1), 'loss:',
                  '{:.6f},'.format(total_loss / count_loss),
                  'ppl:', '{:.6}'.format(math.exp(total_loss / count_loss)), 'valid_accuracy:',
                  '{:.4f}%,'.format(valid_accuracy / validstep))
            print("step=", validstep)

            loss1 = total_loss / count_loss
            ppl = math.exp(total_loss / count_loss)
            v_acc = valid_accuracy / validstep
            valid_result.append([loss1,ppl,v_acc])
        
        torch.save(model.state_dict(), "./weights/de2en_" + str(epoch) + ".pkl")


def calculate_accuracy(model_predict, target, ignore_index=0):
    # target中 非0的地方填充为1
    non_pad_mask = target.ne(ignore_index)
    # target中 unk的地方也不算
    # 得到target中1的数量，即有效的词的数量
    word_num = non_pad_mask.sum().item()
    # 得到预测正确的数量
    predict_correct_num = model_predict.max(dim=-1).indices.eq(target).masked_select(non_pad_mask).sum().item()
    return predict_correct_num / word_num * 100
def adjust_learning_rate(d_model, step, optimizer, warmup_steps=1000 * 8):
    lr = 1 / math.sqrt(d_model) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    for param_group in optimizer.param_groups:
        # 在每次更新参数前迭代更改学习率
        param_group["lr"] = lr

#复制的别人的,我看不懂
# def bleu(pred_seq, label_seq, k):
#     """Compute the BLEU."""
#     pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
#     len_pred, len_label = len(pred_tokens), len(label_tokens)
#     score = math.exp(min(0, 1 - len_label / len_pred))
#     for n in range(1, k + 1):
#         num_matches, label_subs = 0, collections.defaultdict(int)
#         for i in range(len_label - n + 1):
#             label_subs[''.join(label_tokens[i:i + n])] += 1
#         for i in range(len_pred - n + 1):
#             if label_subs[''.join(pred_tokens[i:i + n])] > 0:
#                 num_matches += 1
#                 label_subs[''.join(pred_tokens[i:i + n])] -= 1
#         score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
#     return score

if __name__ == "__main__":
    train()
    