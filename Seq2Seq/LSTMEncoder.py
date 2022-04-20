import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(EncoderLSTM, self).__init__()

        # Size of the one hot vectors that will be the input to the encoder
        # self.input_size = input_size

        # Output size of the word embedding NN
        # self.embedding_size = embedding_size

        # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
        self.hidden_size = hidden_size

        # Number of layers in the lstm
        self.num_layers = num_layers

        # Regularization parameter
        self.dropout = nn.Dropout(p)
        self.tag = True

        # Shape --------------------> (5376, 300) [input size, embedding dims]
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    # Shape of x (26, 32) [Sequence_length, batch_size]
    def forward(self, x):
        # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
        embedding = self.dropout(self.embedding(x))

        # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
        # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
        outputs, (hidden_state, cell_state) = self.LSTM(embedding)

        return hidden_state, cell_state


# input_size_encoder = len(german.vocab)
# encoder_embedding_size = 300
# hidden_size = 1024
# num_layers = 2
# encoder_dropout = 0.5
#
# encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
#                            hidden_size, num_layers, encoder_dropout).to(device)
# print(encoder_lstm)