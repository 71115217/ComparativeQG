import random

import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2Seq(nn.Module):
    def __init__(self, Encoder_LSTM, Decoder_LSTM):
        super(Seq2Seq, self).__init__()
        self.Encoder_LSTM = Encoder_LSTM
        self.Decoder_LSTM = Decoder_LSTM

    def forward(self, source, target, question_vocab_size, tfr=0.5):
        # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
        batch_size = source.shape[1]

        # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
        target_len = target.shape[0]
        target_vocab_size = question_vocab_size

        # Shape --> outputs (14, 32, 5766)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
        hidden_state, cell_state = self.Encoder_LSTM(source)

        # Shape of x (32 elements)
        x = target[0]  # Trigger token <SOS>

        for i in range(1, target_len):
            # Shape --> output (32, 5766)
            output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
            outputs[i] = output
            best_guess = output.argmax(1)  # 0th dimension is batch size, 1st dimension is word embedding
            x = target[
                i] if random.random() < tfr else best_guess  # Either pass the next word correctly from the dataset or use the earlier predicted word

        # Shape --> outputs (14, 32, 5766)
        return outputs