import random

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Seq2Seq(nn.Module):
    """
    要点：
    1.该网络包含一个encoder和一个decoder，使用的RNN的结构相同，最后使用全连接接预测结果
    2.RNN网络结构要熟知
    3.seq2seq的精髓：encoder层生成的参数作为decoder层的输入
    """
    def __init__(self, input_vocab_size, output_vocab_size, n_hidden):
        super().__init__()
        # 此处的input_size是每一个节点可接纳的状态，hidden_size是隐藏节点的维度
        self.enc = nn.RNN(input_size=input_vocab_size, hidden_size=n_hidden, dropout=0.5).to(device)
        self.dec = nn.RNN(input_size=output_vocab_size, hidden_size=n_hidden, dropout=0.5).to(device)
        self.fc = nn.Linear(n_hidden, output_vocab_size).to(device)
        self.enc.cuda()
        self.dec.cuda()
        self.fc.cuda()

    def forward(self, enc_input, enc_hidden, dec_input):
        # RNN要求输入：(seq_len, batch_size, n_class)，这里需要转置一下
        enc_input = enc_input.transpose(0,1)
        dec_input = dec_input.transpose(0,1)
        _, enc_states = self.enc(enc_input, enc_hidden)
        outputs, _ = self.dec(dec_input, enc_states)
        pred = self.fc(outputs)

        return pred
