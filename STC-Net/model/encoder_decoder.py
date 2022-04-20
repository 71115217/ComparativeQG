from torch import nn
from CopyNet.copynet.utils import tokens_to_seq, seq_to_string
from .attention_decoder import AttentionDecoder
from .copynet_decoder import CopyNetDecoder
from spacy.lang.en import English
from .encoder import EncoderRNN
from torch.autograd import Variable
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import pickle
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_length, hidden_size, embedding_size, decoder_type, comparative_classified_soft_template):
        super(EncoderDecoder, self).__init__()

        self.lang = lang
        init_embedding_weight = self.get_glove_embedding_weight()
        self.encoder = EncoderRNN(len(self.lang.tok_to_idx),
                                  hidden_size,
                                  embedding_size, init_embedding_weight)
        self.decoder_type = decoder_type
        self.comparative_classified_soft_template = comparative_classified_soft_template
        decoder_hidden_size = 2*self.encoder.hidden_size
        if self.decoder_type == 'attn':
            self.decoder = AttentionDecoder(decoder_hidden_size,
                                            embedding_size,
                                            lang,
                                            max_length)
        elif self.decoder_type == 'copy':
            self.decoder = CopyNetDecoder(decoder_hidden_size,
                                          embedding_size,
                                          lang,
                                          max_length)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

    def get_glove_embedding_weight(self):
        '''转换向量过程'''
        weight = []
        # 已有的glove词向量
        # glove_file = datapath('glove.6B.100d.txt')
        # # 指定转化为word2vec格式后文件的位置
        # tmp_file = get_tmpfile("word2vec.6B.100d.txt")
        # glove2word2vec(glove_file, tmp_file)
        # # ‘’‘’导入向量‘’‘’
        # # 加载转化后的文件
        # wvmodel = KeyedVectors.load_word2vec_format(tmp_file)

        f = open('./pickled_model_300d', 'rb')
        wvmodel = pickle.load(f)
        for i in range(len(self.lang.tok_to_idx)):
            try:
                token = self.lang.idx_to_tok[i]
                if token in wvmodel.key_to_index.keys():
                    w = torch.from_numpy(wvmodel.get_vector(token))
                    weight.append(w)
                else:
                    weight.append(torch.zeros(wvmodel.vector_size))
            except Exception as e:
                weight.append(torch.zeros( wvmodel.vector_size))
                continue

        return torch.tensor([item.cpu().detach().numpy() for item in weight]).to(device)
    def comparative_tensor_selector(self, comparative_types):
        comparative_tensors = []
        for i in comparative_types:
            comparative_tensors.append(self.comparative_classified_soft_template[str(int(i))])
        comparative_tensors = torch.tensor([item.cpu().detach().numpy() for item in comparative_tensors]).to(device)
        comparative_tensors  = comparative_tensors.squeeze(1)
        return comparative_tensors

    def forward(self, inputs, lengths, comparative_type, targets=None, keep_prob=1.0, teacher_forcing=0.0):

        batch_size = inputs.data.shape[0]
        hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, hidden = self.encoder(inputs, hidden, lengths)
        comparative_tensor = self.comparative_tensor_selector(comparative_type)
        decoder_outputs, sampled_idxs, attention_scores = self.decoder(encoder_outputs,
                                                     inputs,
                                                     hidden,
                                                     comparative_tensor,
                                                     targets=targets,
                                                     teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs, attention_scores

    def get_response(self, input_string):
        use_extended_vocab = isinstance(self.decoder, CopyNetDecoder)

        if not hasattr(self, 'parser_'):
            self.parser_ = English()

        idx_to_tok = self.lang.idx_to_tok
        tok_to_idx = self.lang.tok_to_idx

        input_tokens = self.parser_(' '.join(input_string.split()))
        input_tokens = ['<SOS>'] + [token.orth_.lower() for token in input_tokens] + ['<EOS>']
        input_seq = tokens_to_seq(input_tokens, tok_to_idx, len(input_tokens), use_extended_vocab)
        input_variable = Variable(input_seq).view(1, -1)

        if next(self.parameters()).is_cuda:
            input_variable = input_variable.to(device)

        outputs, idxs = self.forward(input_variable, [len(input_seq)])
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)

        return output_string

    def interactive(self, unsmear):
        while True:
            input_string = input("\nType a message to Amy:\n")
            output_string = self.get_response(input_string)

            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')

            print('\nAmy:\n', output_string)
