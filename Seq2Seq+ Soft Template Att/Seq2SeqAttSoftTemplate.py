from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn import model_selection
from torch import optim
import torch.nn.functional as F
from rouge import Rouge
import BLEU
# from ROUGE import Rouge_1, Rouge_2, Rouge_L
from SparseSQL2Seq import get_source_sentence, get_target_sentence
#可执行，加了软模板之后
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            word = word.strip('?')
            word = word.strip('(')
            word = word.strip(')')
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#convert unicode TO Ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# def readLangs(lang1, lang2, reverse=False):
#     print("Reading lines...")
#
#     # Read the file and split into lines
#     lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
#         read().strip().split('\n')
#
#     # Split every line into pairs and normalize
#     for line in lines:
#         for s in line.split("#"):
#             print(s)
#             print("\n")
#     pairs = [[normalizeString(s) for s in l.split('#')] for l in lines]
#
#     # Reverse pairs, make Lang instances
#     if reverse:
#         pairs = [list(reversed(p)) for p in pairs]
#         input_lang = Lang(lang2)
#         output_lang = Lang(lang1)
#     else:
#         input_lang = Lang(lang1)
#         output_lang = Lang(lang2)
#
#     return input_lang, output_lang, pairs
#
# def prepareData(lang1, lang2, reverse=False):
#     input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
#     print("Read %s sentence pairs" % len(pairs))
#     print("Counting words...")
#     for pair in pairs:
#         input_lang.addSentence(pair[0])
#         output_lang.addSentence(pair[1])
#     print("Counted words:")
#     print(input_lang.name, input_lang.n_words)
#     print(output_lang.name, output_lang.n_words)
#     return input_lang, output_lang, pairs

def prepareDataFromPairs(pairs, lang1, lang2, reverse = False):
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0].lower())
        output_lang.addSentence(pair[1].lower())
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang

# print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    # words = sentence.split(' ')
    # words = []
    res = []
    for word in normalizeString(sentence).split(' '):
        word = word.strip('?')
        word = word.strip('(')
        word = word.strip(')')
        if word in lang.word2index.keys():
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    return res


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


teacher_forcing_ratio = 0.5
MAX_LENGTH = 64

def train(input_tensor, target_tensor, encoder, template_vector, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #初始化隐藏层变量
    encoder_hidden = encoder.initHidden()
    #梯度设置为0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #输入句子的长度
    input_length = input_tensor.size(0)
    #目标句子的长度
    target_length = target_tensor.size(0)

    #编码层的输出 max_length*hidden_size，用于解码层attention的计算
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #编码层的初始输入向量
    decoder_input = torch.tensor([[SOS_token]], device=device)
    encoder_hidden.detach()
    encoder_hidden = encoder_hidden.detach()
    #decoder_hidden = torch.add(encoder_hidden, template_vector)
    decoder_hidden = 0.9*encoder_hidden+0.1*template_vector
    decoder_hidden = decoder_hidden.detach()
    #decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            #教师模式下的输入来源于目标句
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIters(training_pairs, epoch, encoder, decoder, n_iters, template_vector, print_every=100, plot_every=100, learning_rate=0.001):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    for s in range(epoch):
        print(s)
        for iter in range(len(training_pairs)):
            # print(str(iter)+"/"+str(n_iters))
            training_pair = training_pairs[iter]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder, template_vector,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                if iter == 0:
                    continue
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                if iter == 0:
                    continue
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            #用于计算注意力的输出层向量
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #每一个输出字符的注意力机制，每个字符对应的注意力大小为64*1,总共长度为64。
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        #所以注意力的map图应该是encoder_outputs 和 decoder_attention的乘积
        return decoded_words, decoder_attentions[:di + 1], encoder_outputs[:input_length+1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def getSoftTemplateVector(encoder, training_pairs, learning_rate, max_length):
    encoder_optimizer2 = optim.Adam(encoder.parameters(), lr=learning_rate)
    template_vector = torch.zeros(1, 1, encoder.hidden_size, device=device)
    for iter in range(len(training_pairs)):
        training_pair = training_pairs[iter]
        input_tensor = training_pair[0]
        # 初始化隐藏层变量
        encoder_hidden = encoder.initHidden()
        # 梯度设置为0
        encoder_optimizer2.zero_grad()
        # 输入句子的长度
        input_length = input_tensor.size(0)

        # 编码层的输出 max_length*hidden_size，用于解码层attention的计算
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # # 编码层的初始输入向量
        # decoder_input = torch.tensor([[SOS_token]], device=device)
        template_vector = template_vector + encoder_hidden
        # print(encoder_hidden)
    template_vector = template_vector/len(training_pairs)
    return template_vector


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

if __name__ == "__main__":
    hidden_size = 64

    # source_sentences = get_source_sentence("source_all_spurious_SQL_data.txt")
    # target_sentences = get_target_sentence("target_all_questions_data.txt")

    source_sentences = get_source_sentence("source_gradable_spurious_SQL_data.txt")
    target_sentences = get_target_sentence("target_gradable_questions_data.txt")
    examples = [[source_sentences[i],target_sentences[i]] for i in range(len(source_sentences))]

    train_examples, test_examples = model_selection.train_test_split(examples, test_size=0.3)

    source_corpus = source_sentences
    target_corpus = target_sentences
    source_test_sentences = [ test_pair[0] for test_pair in test_examples ]
    target_test_sentences = [ test_pair[1] for test_pair in test_examples ]
    source_train_sentences = [ train_pair[0] for train_pair in train_examples ]
    target_train_sentences = [ train_pair[1] for train_pair in train_examples ]

    #
    # source_test_sentences = get_source_sentence("source_test.txt")
    # target_test_sentences = get_target_sentence("target_test.txt")
    # source_train_sentences = get_source_sentence("superlative_source_train.txt")
    # target_train_sentences = get_target_sentence("superlative_target_train.txt")
    #
    # source_corpus = source_test_sentences + source_train_sentences
    # target_corpus = target_test_sentences+target_train_sentences

    pairs = [[source_corpus[i].strip('\n'),target_corpus[i].strip('\n')]for i in range(len(source_corpus))]
    input_pairs = [[source_train_sentences[i].strip('\n'),target_train_sentences[i].strip('\n')]for i in range(len(target_train_sentences))]
    output_pairs = [[source_test_sentences[i].strip('\n'),target_test_sentences[i].strip('\n')]for i in range(len(source_test_sentences))]
    input_lang, output_lang =prepareDataFromPairs(pairs,"sparseSQL","nl", False)
    # input_lang, output_lang, pairs = prepareData('sparseSQL', 'nl', False)

    encoder_template = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    training_pairs = [tensorsFromPair(pair,input_lang, output_lang)
                      for pair in input_pairs]
    template_vector = getSoftTemplateVector(encoder_template, training_pairs, learning_rate=0.001, max_length=MAX_LENGTH)

    trainIters(training_pairs, 50, encoder1, attn_decoder1, 20,template_vector, print_every=500)
    # bleu1 = 0
    # bleu2 = 0
    # bleu3 = 0
    # bleu4 = 0
    # for pair in output_pairs:
    #     decoded_words,_ = evaluate(encoder1, attn_decoder1, pair[0],max_length=MAX_LENGTH)
    #     bleu1 += BLEU.compute_bleu(pair[1].split(), decoded_words, 1)[0]
    #     bleu2 += BLEU.compute_bleu(pair[1].split(), decoded_words, 2)[0]
    #     bleu3 += BLEU.compute_bleu(pair[1].split(), decoded_words, 3)[0]
    #     bleu4 += BLEU.compute_bleu(pair[1].split(), decoded_words, 4)[0]
    #     print(decoded_words)
    #     print(pair[1])
    # print("BLEU-1:%f", float(bleu1/len(output_pairs)))
    # print("BLEU-2:%f",float(bleu2 / len(output_pairs)))
    # print("BLEU-3:%f",float(bleu3 / len(output_pairs)))
    # print("BLEU-4:%f",float(bleu4 / len(output_pairs)))
    rouge = Rouge()
    meteor = 0
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    for pair in output_pairs:
        decoded_words, decode_attentions, encoder_outputs = evaluate(encoder1, attn_decoder1, pair[0],max_length=MAX_LENGTH)
        # decode_attentions.to('cuda')
        # encoder_outputs.to('cuda')
        # print(decode_attentions.size())
        # print(encoder_outputs.size())
        a = decode_attentions
        b = encoder_outputs.cpu()
        # print(a)
        # print(b)
        attention_res = torch.mm(a, b.t())
        print(attention_res)
        print(decoded_words)
        bleu1 += sentence_bleu([pair[1].split()], decoded_words, (1, 0, 0, 0))
        bleu2 += sentence_bleu([pair[1].split()], decoded_words, (0.5, 0.5, 0, 0))
        bleu3 += sentence_bleu([pair[1].split()], decoded_words, (0.3, 0.3, 0.4, 0))
        bleu4 += sentence_bleu([pair[1].split()], decoded_words, (0.25, 0.25, 0.25, 0.25))
        print(decoded_words)
        print(pair[1])
    print("BLEU-1:%f", float(bleu1/len(output_pairs)))
    print("BLEU-2:%f",float(bleu2 / len(output_pairs)))
    print("BLEU-3:%f",float(bleu3 / len(output_pairs)))
    print("BLEU-4:%f",float(bleu4 / len(output_pairs)))
    rouge_avg_1 = 0
    rouge_avg_2 = 0
    rouge_avg_L = 0
    for index in range(len(output_pairs)):
        rouge_score = rouge.get_scores(" ".join(decoded_words), pair[1])
        meteor += round(meteor_score([" ".join(decoded_words)], pair[1]), 4)
        rouge_1 = rouge_score[0]["rouge-1"]['f']
        rouge_2 = rouge_score[0]["rouge-2"]['f']
        rouge_L = rouge_score[0]["rouge-l"]['f']
        rouge_avg_1 = rouge_avg_1 + rouge_1
        rouge_avg_2 = rouge_avg_2 + rouge_2
        rouge_avg_L = rouge_avg_L + rouge_L
    ROUGE1 = float(rouge_avg_1/ len(output_pairs))
    ROUGE2 = float(rouge_avg_2 / len(output_pairs))
    ROUGEL = float(rouge_avg_L / len(output_pairs))
    METEOR = float(meteor / len(output_pairs))
    print("ROUGE-1:%s",str(ROUGE1))
    print("ROUGE-2:%s",str(ROUGE2))
    print("ROUGE-L:%s",str(ROUGEL))
    print("METEOR:%s", str(METEOR))