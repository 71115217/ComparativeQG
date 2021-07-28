import nltk
import torch
import numpy as np
# Transformer Parameters
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torch import nn, optim
import torch.utils.data as Data
from SparseSQL2Seq import get_source_sentence, get_target_sentence
from Seq2Seq_Attention import normalizeString
from rouge import Rouge
#构建词典
def construct_word_dictionary(data):
    word2id = {}
    word2id["pad"] =0 #结束标志
    word2id["sos"] = 1#未知词
    word2id["eos"] = 2 #起始标志
    word2id["unk"] = 3 #占位符
    id_index = 4
    for sentence in data:
        sentence_words = sentence.split()
        for word in sentence_words:
            word = word.lower()
            word = word.strip('?')
            word = word.strip(')')
            word = word.strip('(')
            if word not in word2id.keys():
                word2id[word] = id_index
                id_index = id_index + 1
    word2id['?'] = id_index
    id_index = id_index + 1
    id2word = {i:k for i, k in enumerate(word2id)}

    return word2id,id2word

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

# source_test_sentences = get_source_sentence("source_test.txt")
# target_test_sentences = get_target_sentence("target_test.txt")
source_train_sentences = get_source_sentence("source_all_spurious_SQL_data.txt")
target_train_sentences = get_target_sentence("target_all_questions_data.txt")

source_corpus = source_train_sentences
target_corpus = target_train_sentences

sourceWord2Id, sourceId2Word = construct_word_dictionary(source_corpus)
targetWord2Id, targetId2Word = construct_word_dictionary(target_corpus)


train_sentences = []
for s_index in range(len(source_corpus)):
    one_batch = []
    one_batch.append(normalizeString(source_corpus[s_index].lower())+" pad")
    one_batch.append("sos " + normalizeString(target_corpus[s_index].lower()))
    one_batch.append(normalizeString(target_corpus[s_index].lower()) + " eos")
    train_sentences.append(one_batch)

# sentences = [
#     # enc_input                dec_input            dec_output
#     ['select comparee which more comparee japan standard china parameter medals comparee_column country <PAD>',
#      '<SOS> which country got more medals japan or china',
#      'which country got more medals japan or china <EOS>'],
#     ['select comparee whch most parameter awards comparee_column player <PAD>',
#      '<SOS> which player got the most awards',
#      'which player got the most awards <EOS>']
# ]

# Padding Should be Zero
# src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(sourceWord2Id)

# tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
idx2word = {i: w for i, w in enumerate(targetWord2Id)}
tgt_vocab_size = len(targetWord2Id)

src_len = 64  # enc_input max sequence length
tgt_len = 65 # dec_input(=dec_output) max sequence length

#
# def make_input_batch(data, seq_len, sparseSQLWord2Id, batch_size):
#     input_batch = []
#     for sequence in data:
#         one_batch = []
#         sequence_word = sequence.split()
#         for word in sequence_word:
#             if word in sparseSQLWord2Id.keys():
#                 one_batch.append(sparseSQLWord2Id[word])
#             else:
#                 one_batch.append(sparseSQLWord2Id['<UNK>'])
#         for i in range(seq_len - len(sequence_word)):
#             one_batch.append(sparseSQLWord2Id['<PAD>'])
#         input_batch.append(np.eye(len(sparseSQLWord2Id))[one_batch])
#     for i in range(batch_size - len(data)):
#         for i in range(seq_len):
#             one_batch.append(sparseSQLWord2Id['<PAD>'])
#         input_batch.append(np.eye(len(sparseSQLWord2Id))[one_batch])
#     return torch.tensor(input_batch, dtype=torch.float, device=device)
#
# def make_output_batch(data, seq_len, targetWord2Id, batch_size):
#     output_batch = []
#     for sequence in data:
#         one_batch = []
#         sequence_word = sequence.split()
#         one_batch.append(targetWord2Id['<SOS>'])
#         for word in sequence_word:
#             if word in targetWord2Id.keys():
#                 one_batch.append(targetWord2Id[word])
#             else:
#                 one_batch.append(targetWord2Id['<UNK>'])
#         for i in range(seq_len - len(sequence_word)):
#             one_batch.append(targetWord2Id['<PAD>'])
#         output_batch.append(np.eye(len(targetWord2Id))[one_batch])
#     for i in range(batch_size - len(data)):
#         for i in range(seq_len):
#             one_batch.append(targetWord2Id['<PAD>'])
#         output_batch.append(np.eye(len(targetWord2Id))[one_batch])
#     return torch.tensor(output_batch, dtype=torch.float, device=device)

#将句子转化为词向量表示，词向量每一维代表单词索引
def make_data(train_sentences, seq_len):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(train_sentences)):
        input_words = []
        for n in normalizeString(train_sentences[i][0]).split():
            if n in sourceWord2Id.keys():
                input_words.append(sourceWord2Id[n] )
            else:
                input_words.append(sourceWord2Id['unk'])
        for t in range(seq_len - len(input_words)):
            input_words.append(sourceWord2Id['pad'])
          # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        enc_input = [input_words]

        dec_input_words = []
        for n in normalizeString(train_sentences[i][1]).split():
            if n in targetWord2Id.keys():
                dec_input_words.append(targetWord2Id[n])
            else:
                dec_input_words.append(targetWord2Id['unk'])

        for t in range(seq_len - len(dec_input_words)+1):
            dec_input_words.append(targetWord2Id['pad'])
        dec_input = [dec_input_words]  # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]

        dec_output_words = []
        for n in normalizeString(train_sentences[i][2]).split():
            if n in targetWord2Id.keys():
                dec_output_words.append(targetWord2Id[n])
            else:
                dec_output_words.append(targetWord2Id['unk'])
        for t in range(seq_len - len(dec_output_words)+1):
            dec_output_words.append(targetWord2Id['pad'])

        dec_output = [dec_output_words]  # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(train_sentences, src_len)

# train_set = []
# test_set = []
# for i in range(len(enc_inputs_org)):
#     if i % 10 == 1 or i % 10 ==2 or i % 10 == 3:
#         test_set.append([enc_inputs_org[i],dec_inputs_org[i], dec_outputs_org[i]])
#     else:
#         train_set.append([enc_inputs_org[i],dec_inputs_org[i],dec_outputs_org[i]])



#数据集加载器
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]





loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
d_model = 128  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 128  # dimension of K(=Q), V
n_layers = 2  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention

#注意力机制mask，对无效区域采取屏蔽策略。
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

#位置编码向量
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

#解码器的掩码生成
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

#Q，K，V计算，获得注意力向量
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        #residual残差
        return nn.LayerNorm(d_model)(output + residual) # [batch_size, seq_len, d_model]

#编码层，输出注意力和输出
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

#编码层，
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_vocab_size, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        word_emb = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        pos_emb = self.pos_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = word_emb + pos_emb
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

custom_dataset = loader.dataset

# for i in range(len(custom_dataset)):
#     print(custom_dataset[i])

train_size = int(len(custom_dataset) * 0.8)
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])
train_loader = Data.DataLoader(train_dataset, 2, True)
test_loader = Data.DataLoader(test_dataset, 1, True)

# for tuple_test_input in test_dataset:
#     print(tuple_test_input[2])
# print("#" * len(custom_dataset))
#
# for i in range(len(train_dataset)):
#     print(train_dataset[i])
#
# print("#" * len(train_dataset))
#
# for i in range(len(test_dataset)):
#     print(test_dataset[i])

for epoch in range(10):
    for enc_inputs,dec_inputs, dec_outputs in train_loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      # enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      loss = criterion(outputs, dec_outputs.view(-1))


      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input
rouge = Rouge()
bleu1 = 0
bleu2 = 0
bleu3 = 0
bleu4 = 0
rouge_avg_1 = 0
rouge_avg_2 = 0
rouge_avg_L = 0
meteor = 0
# Test
test_size = 0
# for tuple_test_input in test_loader:
for enc_inputs, dec_inputs, dec_outputs in test_loader:
    greedy_dec_input= greedy_decoder(model, enc_inputs[0].view(1, -1), start_symbol=targetWord2Id["sos"])
    pre_sentence = [idx2word[n.item()] for n in greedy_dec_input.squeeze()]
    if 'eos' in pre_sentence:
        end = pre_sentence.index('eos')
    else:
        end = len(pre_sentence) - 1
    start = pre_sentence.index('sos')
    pre_sentence = pre_sentence[start+1:end]
    predict_sentence = ' '.join(pre_sentence)

    src_sentence = [idx2word[n.item()] for n in dec_outputs[0].squeeze()]
    src_end = src_sentence.index('eos')
    src_sentence = src_sentence[:src_end]
    source_sentence = ' '.join(src_sentence)
    print(predict_sentence)
    print(source_sentence)
    bleu1 += sentence_bleu([src_sentence], pre_sentence, (1, 0, 0, 0))
    bleu2 += sentence_bleu([src_sentence], pre_sentence, (0.5, 0.5, 0, 0))
    bleu3 += sentence_bleu([src_sentence], pre_sentence, (0.3, 0.3, 0.4, 0))
    bleu4 += sentence_bleu([src_sentence], pre_sentence, (0.25, 0.25, 0.25, 0.25))

    rouge_score = rouge.get_scores(predict_sentence, source_sentence)
    rouge_1 = rouge_score[0]["rouge-1"]['f']
    rouge_2 = rouge_score[0]["rouge-2"]['f']
    rouge_L = rouge_score[0]["rouge-l"]['f']
    rouge_avg_1 = rouge_avg_1 + rouge_1
    rouge_avg_2 = rouge_avg_2 + rouge_2
    rouge_avg_L = rouge_avg_L + rouge_L
    meteor += round(meteor_score([source_sentence], predict_sentence), 4)
    test_size = test_size + 1


print("BLEU-1:%f", float(bleu1/test_size))
print("BLEU-2:%f",float(bleu2 / test_size))
print("BLEU-3:%f",float(bleu3 / test_size))
print("BLEU-4:%f",float(bleu4 / test_size))
print("METEOR:%f",float(meteor / test_size))
ROUGE1 = float(rouge_avg_1/ test_size)
ROUGE2 = float(rouge_avg_2 / test_size)
ROUGEL = float(rouge_avg_L / test_size)
print("ROUGE-1:%s",str(ROUGE1))
print("ROUGE-2:%s",str(ROUGE2))
print("ROUGE-L:%s",str(ROUGEL))

# for enc_inputs, dec_inputs, dec_outputs in test_loader:
#     decoder_input = torch.tensor([[sourceWord2Id['pad']]*(src_len+1)])
#     # enc_inputs, dec_inputs, _ = next(iter(test_loader))
#     # print(decoder_input)
#     # print(dec_inputs[0])
#     # model.eval()
#     predict, _, _, _ = model(enc_inputs[0].view(1, -1), dec_inputs[0].view(1, -1)) # model(enc_inputs[0].view(1, -1), greedy_dec_input)
#     predict = predict.data.max(1, keepdim=True)[1]
#     pre_sentence = []
#     tar_sentence = []
#     for n in predict.squeeze():
#         pre_sentence.append(idx2word[n.item()])
#     for t in dec_outputs.squeeze():
#         tar_sentence.append(idx2word[t.item()])
#
#     print(pre_sentence)
#     print(tar_sentence)
#     print(enc_inputs[0], '->', [idx2word[n.item()] for n in predict.squeeze()])