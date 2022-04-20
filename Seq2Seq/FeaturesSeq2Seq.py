#coding=utf-8
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sklearn import model_selection

# from Metrics.BLEU import BLEU
from torch.autograd import Variable

# from ROUGE import Rouge, Rouge_1, Rouge_2, Rouge_L

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#获取原语句
from EncoderDecoder.Seq2Seq import Seq2Seq

#获取特征
def get_source_sentence(source_path):
    source_read_sentences = []
    with open(source_path) as r_source:
        source_sentences = r_source.readlines()
        for sentence in source_sentences:
            source_read_sentences.append(sentence)
        r_source.close()
    return source_read_sentences

#获取目标句
def get_target_sentence(target_path):
    target_read_sentence = []
    with open(target_path) as r_target:
        target_sentences = r_target.readlines()
        for sentence in target_sentences:
            target_read_sentence.append(sentence)
        r_target.close()
    return target_read_sentence

# seq_data = ['select comparee which MORE comparee Japan standard China parameter medals comparee_column total',
#             'select comparee what LESS comparee UK standard Austria parameter gold medals comparee_column total']
#
# target_DATA = ['which country got more medals Japan or China',
#                'what country got less gold medals UK or Austria']

#构建词典
def construct_word_dictionary(data):
    word2id = {}
    word2id["<EOS>"] =0 #结束标志
    word2id["<UNK>"] = 1#未知词
    word2id["<SOS>"] = 2 #起始标志
    word2id["<PAD>"] = 3 #占位符
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

    id2word = {i:k for i, k in enumerate(word2id)}

    return word2id,id2word


def batch_iter(source_sentences, target_sentences, batch_size, num_epochs, shuffle=True):
    data_size = len(source_sentences)

    num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了
    source_sentences = np.array(source_sentences)
    target_sentences = np.array(target_sentences)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))

            shuffled_source_sentences = source_sentences[shuffle_indices]
            shuffled_target_sentences = target_sentences[shuffle_indices]
        else:
            shuffled_source_sentences = source_sentences
            shuffled_target_sentences = target_sentences

        for batch_num in range(num_batches_per_epoch):  # batch_num取值0到num_batches_per_epoch-1
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield (shuffled_source_sentences[start_index:end_index], shuffled_target_sentences[start_index:end_index])

def train(n_hidden, seq_len, batch_size, sparseSQLWord2Id, sparseId2Word, targetWord2Id, targetId2Word,
                  input_vocab_size, output_vocab_size, source_sentences, target_sentences, epoch):
    model = Seq2Seq(input_vocab_size, output_vocab_size, n_hidden)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_index = 0
    # pbar = tqdm(range(100))
    for (input_sentence,output_sentence) in batch_iter(source_sentences, target_sentences, batch_size, epoch, True):
        input_batch = make_input_batch(input_sentence, seq_len, sparseSQLWord2Id, batch_size)
        output_batch = make_output_batch(output_sentence, seq_len, targetWord2Id, batch_size)
        target_batch = make_target_batch(output_sentence, seq_len, targetWord2Id, batch_size)
        hidden = torch.zeros(1, batch_size, n_hidden,dtype=torch.float, device=device)
        optimizer.zero_grad()
        pred = model(input_batch, hidden, output_batch)
        pred = pred.transpose(0, 1)
        loss = 0
        for i in range(batch_size):
            temp = pred[i]
            tar = target_batch[i]
            loss +=  loss_fun(pred[i], target_batch[i].long())
        # for i in pbar:
        #     time.sleep(.01)
        #     pbar.set_description("Processing %s" % i)
            epoch_index=epoch_index + 1
        if epoch_index%len(target_sentences) == 0:
            print(epoch_index)
        # print("\rEpoch: {:d} batch: {:d} loss: {:.4f} ".format(float(epoch_index/len(source_sentences)), float(epoch_index%len(source_sentences)), loss), end='')
        # print("\rEpoch: {:d}/{:d} epoch_loss: {:.4f} ".format(epoch_index, i, loss, end='\n'))
        # if (epoch + 1) % 1000 == 0:
        #     print('Epoch: %d   Cost: %f' % (epoch + 1, loss))
        loss.backward()
        optimizer.step()
    print("epoch:")
    print(epoch_index / len(target_sentences))
    return model

# def preprocess(data):
#     for sentence in data:
#         sentence_words = sentence.split()
#         num_dict = {n: i for i, n in enumerate(sentence_words)}
#         words = [num_dict[n] for n in sentence_words]

def make_input_batch(data, seq_len, sparseSQLWord2Id, batch_size):
    input_batch = []
    for sequence in data:
        one_batch = []
        sequence_word = sequence.split()
        for word in sequence_word:
            if word in sparseSQLWord2Id.keys():
                one_batch.append(sparseSQLWord2Id[word])
            else:
                one_batch.append(sparseSQLWord2Id['<UNK>'])
        for i in range(seq_len - len(sequence_word)):
            one_batch.append(sparseSQLWord2Id['<PAD>'])
        input_batch.append(np.eye(len(sparseSQLWord2Id))[one_batch])
    for i in range(batch_size - len(data)):
        for i in range(seq_len):
            one_batch.append(sparseSQLWord2Id['<PAD>'])
        input_batch.append(np.eye(len(sparseSQLWord2Id))[one_batch])
    return torch.tensor(input_batch, dtype=torch.float, device=device)

def make_output_batch(data, seq_len, targetWord2Id, batch_size):
    output_batch = []
    for sequence in data:
        one_batch = []
        sequence_word = sequence.split()
        one_batch.append(targetWord2Id['<SOS>'])
        for word in sequence_word:
            if word in targetWord2Id.keys():
                one_batch.append(targetWord2Id[word])
            else:
                one_batch.append(targetWord2Id['<UNK>'])
        for i in range(seq_len - len(sequence_word)):
            one_batch.append(targetWord2Id['<PAD>'])
        output_batch.append(np.eye(len(targetWord2Id))[one_batch])
    for i in range(batch_size - len(data)):
        for i in range(seq_len):
            one_batch.append(targetWord2Id['<PAD>'])
        output_batch.append(np.eye(len(targetWord2Id))[one_batch])
    return torch.tensor(output_batch, dtype=torch.float, device=device)

def make_target_batch(data, seq_len, targetWord2Id, batch_size):
    target_batch = []
    for sequence in data:
        one_batch = []
        sequence_word = sequence.split()
        for word in sequence_word:
            if word in targetWord2Id.keys():
                one_batch.append(targetWord2Id[word])
            else:
                one_batch.append(targetWord2Id['<UNK>'])
        for i in range(seq_len - len(sequence_word)):
            one_batch.append(targetWord2Id["<PAD>"])
        one_batch.append(targetWord2Id["<EOS>"])
        target_batch.append(one_batch)

    for i in range(batch_size - len(data)):
        for i in range(seq_len):
            one_batch.append(targetWord2Id['<PAD>'])
        target_batch.append(np.eye(len(targetWord2Id))[one_batch])
    return torch.tensor(target_batch, dtype=torch.float, device=device)

def translate(input_sentence, model, batch_size, sourceWord2Id, targetId2Word):
    # hidden  (1, 1, n_class)
    pre_sentences = []
    for i in range(0, len(input_sentence), 2):
        batch_input_sentence = input_sentence[i:i + 2]
        if len(batch_input_sentence)< batch_size:
            continue
        input_batch = make_input_batch(batch_input_sentence,seq_len, sourceWord2Id, batch_size)
        output_sentence = ["which got more medals China or Japan"] * batch_size
        output_batch = make_output_batch(output_sentence,seq_len, targetWord2Id, len(output_sentence))
        # hidden 形状 (1, 1, n_class)
        hidden = torch.zeros(1, len(input_batch), n_hidden, device=device)
        # output 形状（seq_len，1， 17)
        output = model(input_batch, hidden, output_batch)
        predict = output.data.max(2, keepdim=True)[1]
        # decoded = [targetId2Word[i] for i in predict]
        predict = predict.reshape(-1, 2)
        predict = predict.transpose(0, 1)
        decoded = []
        for sentence_index in predict:
            one_sentence = []
            for j in range(len(sentence_index)):
                word = targetId2Word[int(sentence_index[j])]
                one_sentence.append(word)
            end = one_sentence.index('<PAD>')
            translated = ' '.join(one_sentence[:end])
            decoded.append(translated)
        pre_sentences = pre_sentences + decoded

    return pre_sentences

#预测语句、目标语句、bleu的ngram大小
def bleu_metric(pre_sentences, target_sentences, n):


    # references_corpus = [["which", "one", "got", "less", "medals", "Japan"],
    #                      ["which", "one", "got", "less", "most", "awards", "losses"]]
    # # target_corpus = [["which","country", "got", "more", "medals", "Brazil", "or", "China"],["which","country","got","the","most", "medals"]]
    # target_corpus = [["which", "one", "got", "less", "medals", "Japan"],
    #                  ["which", "one", "got", "less", "most", "awards", "losses"]]
    # b = ["which", "country", "got", "more", "medals", "Brazil", "or", "China"]
    bleu_avg = 0
    for index in range(len(pre_sentences)):
        target_sentence = [target_sentences[index].split()]
        pre_sentence = pre_sentences[index].split()
        if n == 1:
            bleu = sentence_bleu(target_sentence, pre_sentence, weights=(1, 0, 0, 0))
        elif n == 2:
            bleu = sentence_bleu(target_sentence, pre_sentence, weights=(0.5, 0.5, 0, 0))
        elif n == 3:
            bleu = sentence_bleu(target_sentence, pre_sentence, weights=(0.4, 0.3, 0.3, 0))
        else:
            bleu = sentence_bleu(target_sentence, pre_sentence, weights=(0.25, 0.25, 0.25, 0.25))
        # bleu = BLEU.compute_bleu(target_sentence, pre_sentence, n)
        print(bleu)
        bleu_avg = bleu_avg+bleu
    # print(bleu_avg)
    print(float(bleu_avg/len(pre_sentences)))
    return bleu_avg

def rouge(pre_sentences, target_sentences):
    rouge_avg_1 = 0
    rouge_avg_2 = 0
    rouge_avg_L = 0
    meteor = 0
    rouge = Rouge()
    for index in range(len(pre_sentences)):
        rouge_score = rouge.get_scores(pre_sentences[index], target_sentences[index])
        rouge_1 = rouge_score[0]["rouge-1"]['f']
        rouge_2 = rouge_score[0]["rouge-2"]['f']
        rouge_L = rouge_score[0]["rouge-l"]['f']
        meteor += round(meteor_score([pre_sentences[index]], target_sentences[index]), 4)
        # rouge_1 = Rouge_1(pre_sentences[index], target_sentences[index])
        # rouge_2 = Rouge_2(pre_sentences[index], target_sentences[index])
        # rouge_L = Rouge_L(pre_sentences[index], target_sentences[index])
        rouge_avg_1 = rouge_avg_1 + rouge_1
        rouge_avg_2 = rouge_avg_2 + rouge_2
        rouge_avg_L = rouge_avg_L + rouge_L
    ROUGE1 = float(rouge_avg_1 / len(pre_sentences))
    ROUGE2 = float(rouge_avg_2 / len(pre_sentences))
    ROUGEL = float(rouge_avg_L / len(pre_sentences))
    METEOR = float(meteor / len(pre_sentences))
    return ROUGE1, ROUGE2, ROUGEL,METEOR

if __name__=="__main__":
    source_sentences = get_source_sentence("source_all_spurious_SQL_data.txt")
    target_sentences = get_target_sentence("target_all_questions_data.txt")
    examples = [[source_sentences[i],target_sentences[i]] for i in range(len(source_sentences))]

    train_examples, test_examples = model_selection.train_test_split(examples, test_size=0.3)

    source_corpus = source_sentences
    target_corpus = target_sentences
    source_test_sentences = [ test_pair[0] for test_pair in test_examples ]
    target_test_sentences = [ test_pair[1] for test_pair in test_examples ]
    source_train_sentences = [ train_pair[0] for train_pair in train_examples ]
    target_train_sentences = [ train_pair[1] for train_pair in train_examples ]

    # input_sentence = ["select comparee which MORE comparee Brazil standard China parameter medals comparee_column medals",
    #                   "select comparee which MOST parameter awards comparee_column country"]
    # source_test_sentences = get_source_sentence("source_test.txt")
    # target_test_sentences = get_target_sentence("target_test.txt")
    # source_train_sentences = get_source_sentence("superlative_source_train.txt")
    # target_train_sentences = get_target_sentence("superlative_target_train.txt")

    # source_corpus = source_test_sentences + source_train_sentences
    # target_corpus = target_test_sentences + target_train_sentences

    sourceWord2Id, sourceId2Word = construct_word_dictionary(source_corpus)
    targetWord2Id, targetId2Word = construct_word_dictionary(target_corpus)
    input_vocab_size = len(sourceWord2Id)
    output_vocab_size = len(targetWord2Id)
    n_hidden = 64
    seq_len = 64
    batch_size = 8
    epoch = 2

    model = train(n_hidden, seq_len, batch_size, sourceWord2Id, sourceId2Word, targetWord2Id, targetId2Word,
                  input_vocab_size, output_vocab_size, source_train_sentences, target_train_sentences, epoch)
    pre_sentences = translate(source_test_sentences, model, batch_size, sourceWord2Id, targetId2Word)

    bleu_1 = bleu_metric(pre_sentences, target_test_sentences, 1)
    bleu_2 = bleu_metric(pre_sentences, target_test_sentences, 2)
    bleu_3 = bleu_metric(pre_sentences, target_test_sentences, 3)
    bleu_4 = bleu_metric(pre_sentences, target_test_sentences, 4)
    print("BLEU-1:%s",str(bleu_1))
    print("BLEU-2:%s", str(bleu_2))
    print("BLEU-3:%s", str(bleu_3))
    print("BLEU-4:%s", str(bleu_4))

    rouge1, rouge2, rougeL, METEOR = rouge(pre_sentences, target_test_sentences)
    print("ROUGE-1:%s",str(rouge1))
    print("ROUGE-2:%s",str(rouge2))
    print("ROUGE-L:%s",str(rougeL))
    print("METEOR:%s", str(METEOR))
    print(pre_sentences)
