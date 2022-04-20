import torch.nn as nn
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from torch.autograd import Variable
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftTemplateEncoder(nn.Module):
    def __init__(self, lang, embedding_size, hidden_size):
        super(SoftTemplateEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.embedding = nn.Embedding(len(lang.tok_to_idx), embedding_size)
        init_embedding_weight = self.get_glove_embedding_weight()
        weight = torch.squeeze(init_embedding_weight, 1)
        # init_embedding_weight = init_embedding_weight.squeeze(1)
        self.embedding.from_pretrained(weight)
        # self.embed.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, iput, lengths):
        # iput batch must be sorted by sequence length
        hidden = self.init_hidden(1)
        iput = iput.masked_fill(iput > self.embedding.num_embeddings, 5)  # replace OOV words with <UNK> before embedding
        embedded = self.embedding(iput)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

    def get_glove_embedding_weight(self):
        '''转换向量过程'''
        weight = []
        # 已有的glove词向量
        # glove_file = datapath('glove.6B.100d.txt')
        # # 指定转化为word2vec格式后文件的位置
        # tmp_file = get_tmpfile("word2vec.6B.100d.txt")
        # glove2word2vec(glove_file, tmp_file)
        #
        # # ‘’‘’导入向量‘’‘’
        # # 加载转化后的文件
        # wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
        # with open('./pickled_model_300d', 'wb') as f:
        #     pickle.dump(wvmodel, f)
        f = open('./pickled_model_300d', 'rb')
        wvmodel = pickle.load(f)
        for i in range(len(self.lang.tok_to_idx)):
            try:
                token = self.lang.idx_to_tok[i]
                if token in wvmodel.key_to_index.keys():
                    w=torch.tensor(torch.from_numpy(wvmodel.get_vector(token)))
                    weight.append(w.unsqueeze(dim=0))
                else:
                    weight.append(torch.zeros(1, wvmodel.vector_size))
            except Exception as e:
                weight.append(torch.zeros(1, wvmodel.vector_size))
                continue

        return  torch.tensor([item.cpu().detach().numpy() for item in weight]).to(device)