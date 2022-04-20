import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.metrics import bleu_score
from dataclasses import Field
import torch.nn as nn
# from torch.utils.data import DataLoader
from torchtext.data import Field,Example,Dataset, BucketIterator
from torchtext import vocab
from torchtext.data import TabularDataset
import torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

from CopyNet.copynet.evaluate import corpus_rouge
from EncoderDecoder.LSTMDecoder import DecoderLSTM
from EncoderDecoder.LSTMEncoder import EncoderLSTM
from EncoderDecoder.LSTMSeq2Seq import Seq2Seq
import pandas as pd

from Metrics.Metric import MetricCalculation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device= torch.device('cpu')
def features_data_preprocess(data_path):
    features = []
    questions = []
    comparative_data = pd.read_csv(data_path)
    print(comparative_data.columns)
    for row in comparative_data.iterrows():
        feature = list(row[-1][1:-1])
        question = row[-1][-1].split()
        feature_seq = ""
        for i in feature:
            if not i:
                feature_seq = feature_seq + "#" + " "
            else:
                feature_seq = feature_seq + str(i) + " "

        features.append(feature_seq.strip())
        questions.append(" ".join(question))

    return features, questions

def construct_dataset(features, questions):
    Features = Field(lower=True, fix_length=50, tokenize=str.split, init_token="<sos>",
                     eos_token="<eos>")
    Questions = Field(sequential=True, fix_length=50, tokenize=str.split, init_token="<sos>",
                      eos_token="<eos>")

    fields = [("comparative_feature", Features), ("question", Questions)]
    # 3.将数据转换为Example对象的列表
    examples = []
    for feature, question in zip(features, questions):
        # print(feature)
        # print(question)
        example = Example.fromlist([feature, question], fields=fields)
        examples.append(example)

    Features.build_vocab([example.comparative_feature for example in examples], max_size=10000, min_freq=3)
    Questions.build_vocab([example.question for example in examples], max_size=10000, min_freq=3)
    # print(Questions.vocab.__dict__.keys())
    # print(list(Questions.vocab.__dict__.values()))
    e = list(Questions.vocab.__dict__.values())

    word_2_idx = dict(e[3])
    idx_2_word = {}
    for k, v in word_2_idx.items():
        idx_2_word[v] = k
    # print(word_2_idx)
    # print(idx_2_word)

    dataset = Dataset(examples, fields)
    feature_vocab = Features.vocab
    question_vocab = Questions.vocab
    return dataset, feature_vocab, question_vocab

def translate_sentence(model, sentence, source_vocab, target_vocab, device, max_length=50):
    # spacy_ger = spacy.load("de")

    if type(sentence) == str:
        tokens = [token.lower() for token in sentence.split()]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, "<sos>")
    tokens.append("<eos>")
    text_to_indices = [source_vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [target_vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == target_vocab.stoi["<eos>"]:
            break

    translated_sentence = [target_vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

def bleu(data, model, feature_vocab, question_vocab, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, feature_vocab, question_vocab, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss):
    print('saving')
    print()
    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}
    torch.save(state, 'Seq2Seq-checkpoints/checkpoint-LSTMSeq2Seq')
    torch.save(model.state_dict(),'Seq2Seq-checkpoints/checkpoint-LSTMSeq2Seq-SD')


def train(train_iterator, feature_vocab, question_vocab):
    test_results = []
    learning_rate = 0.0001
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0
    epoch_loss = 0.0
    num_epochs = 100
    best_loss = 999999
    best_epoch = -1
    sentence1 = "what most # comparee"
    input_size_encoder = len(feature_vocab)
    encoder_embedding_size = 300
    hidden_size = 100
    num_layers = 2
    encoder_dropout = 0.5

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                               hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = len(question_vocab)
    decoder_embedding_size = 300
    hidden_size = 100
    num_layers = 2
    decoder_dropout = 0.5
    output_size = len(question_vocab)

    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                               hidden_size, num_layers, decoder_dropout, output_size).to(device)

    model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = question_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    ts1 = []

    for epoch in range(num_epochs):
        print("Epoch - {} / {}".format(epoch + 1, num_epochs))
        model.eval()
        # translated_sentence1 = translate_sentence(model, sentence1, feature_vocab, question_vocab, device, max_length=50)
        # print(f"Translated example sentence 1: \n {translated_sentence1}")
        # ts1.append(translated_sentence1)

        model.train(True)
        for batch_idx, batch in enumerate(train_iterator):
            input = batch.comparative_feature.to(device)
            target = batch.question.to(device)

            # Pass the input and target for model's forward method
            output = model(input, target, len(question_vocab))
            output = output[1:].reshape(-1, output.shape[2]).to(device)
            target = target[1:].reshape(-1).to(device)

            # Clear the accumulating gradients
            optimizer.zero_grad()

            # Calculate the loss value for every epoch
            loss = criterion(output, target)

            # Calculate the gradients for weights & biases using back-propagation
            loss.backward()

            # Clip the gradient value is it exceeds > 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Update the weights values using the gradients we calculated using bp
            optimizer.step()
            step += 1
            epoch_loss += loss.item()
            writer.add_scalar("Training loss", loss, global_step=step)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_loss)
            if ((epoch - best_epoch) >= 10):
                print("no improvement in 10 epochs, break")
                break
        print("Epoch_Loss - {}".format(loss.item()))
        # test
        result = evaluation(test_iterator, model, feature_vocab, question_vocab, device)
        test_results.append(result)

    with open("result.csv", "w") as w:
        for res in test_results:
            w.writeline(" ".join(res))
        w.close()
    print(epoch_loss / len(train_iterator))
    return model

def evaluation(test_iterator, model, feature_vocab, question_vocab, device):
    predict_questions = []
    target_questions = []
    for batch_index, batch in enumerate(test_iterator):
        input = batch.comparative_feature.to(device)
        target = batch.question.to(device)
        for i in range(len(batch)):
            features = [feature_vocab.itos[idx] for idx in input.t()[i]]
            features.remove("<sos>")
            features.remove("<eos>")
            features = " ".join(features)
            features = features.replace("<pad>", "").rstrip()
            # print(features)
            model.eval()
            predict_question = translate_sentence(model, features, feature_vocab, question_vocab, device, max_length=50)
            target_question = [question_vocab.itos[idx] for idx in target.t()[i]]
            target_question.remove("<sos>")
            target_question.remove("<eos>")
            target_question = " ".join(target_question)
            target_question = target_question.replace("<pad>", "").rstrip()
            #predict_question 是一个列表
            predict_questions.append(predict_question)
            #target_question 是一个字符串
            target_questions.append(target_question)


    score = bleu_score(predict_questions, target_questions)
    print(f"Bleu score {score * 100:.2f}")

    metric = MetricCalculation()
    bleu_1 = metric.bleu_metric(predict_questions, target_questions, 1)
    bleu_2 = metric.bleu_metric(predict_questions, target_questions, 2)
    bleu_3 = metric.bleu_metric(predict_questions, target_questions, 3)
    bleu_4 = metric.bleu_metric(predict_questions, target_questions, 4)
    print("BLEU-1:%s",str(bleu_1))
    print("BLEU-2:%s", str(bleu_2))
    print("BLEU-3:%s", str(bleu_3))
    print("BLEU-4:%s", str(bleu_4))

    rouge1, rouge2, rougeL, METEOR = metric.rouge(predict_questions, target_questions)
    print("ROUGE-1:%s",str(rouge1))
    print("ROUGE-2:%s",str(rouge2))
    print("ROUGE-L:%s",str(rougeL))
    print("METEOR:%s", str(METEOR))
    result = []
    result.append([bleu_1, bleu_2, bleu_3, bleu_4, rouge1, rouge2, rougeL, METEOR])

    bleu_1_score = corpus_bleu([i.split() for i in target_questions], predict_questions, weights=(1, 0, 0, 0),
                               smoothing_function=SmoothingFunction().method1)
    bleu_2_score = corpus_bleu([i.split() for i in target_questions], predict_questions, weights=(0.5, 0.5, 0, 0),
                               smoothing_function=SmoothingFunction().method1)
    bleu_3_score = corpus_bleu([i.split() for i in target_questions], predict_questions, weights=(0.4, 0.3, 0.3, 0),
                               smoothing_function=SmoothingFunction().method1)
    bleu_4_score = corpus_bleu([i.split() for i in target_questions], predict_questions, weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=SmoothingFunction().method1)
    ROUGE = corpus_rouge([i.split() for i in target_questions], predict_questions)


    return result


def inference(model):
    progress = []

    # for i, sen in enumerate(ts1):
    #     progress.append(TreebankWordDetokenizer().detokenize(sen))
    # print(progress)

    progress_df = pd.DataFrame(data=progress, columns=['Predicted Sentence'])
    progress_df.index.name = "Epochs"
    progress_df.to_csv('/content/predicted_sentence.csv')
    progress_df.head()

    model.eval()
    test_sentences = ["Zwei Männer gehen die Straße entlang", "Kinder spielen im Park.",
                      "Diese Stadt verdient eine bessere Klasse von Verbrechern. Der Spaßvogel"]
    actual_sentences = ["Two men are walking down the street", "Children play in the park",
                        "This city deserves a better class of criminals. The joker"]

    for idx, i in enumerate(test_sentences):
        model.eval()
        translated_sentence = translate_sentence(model, i, feature_vocab, question_vocab, device, max_length=50)
        progress.append(TreebankWordDetokenizer().detokenize(translated_sentence))
        print("German : {}".format(i))
        print("Actual Sentence in English : {}".format(actual_sentences[idx]))
        print("Predicted Sentence in English : {}".format(progress[-1]))
        print()

if __name__ == '__main__':
    #data preprocess
    features, questions = features_data_preprocess("../data/comparative_table_and_features_data.csv")
    dataset, feature_vocab, question_vocab = construct_dataset(features, questions)
    train_dataset, test_dataset = dataset.split(split_ratio=[0.7, 0.3])
    BATCH_SIZE = 16
    train_iterator, test_iterator = BucketIterator.splits((train_dataset, test_dataset),
                                                                        batch_size=BATCH_SIZE,
                                                                        sort_within_batch=True,
                                                                        sort_key=lambda x: len(x.comparative_feature),
                                                                        device=device)
    #train
    model = train(train_iterator, feature_vocab, question_vocab)



    # metric = Metric()
    # metric.rouge()
    # metric.bleu_metric()
    # bleu_1 = bleu_metric(pre_sentences, target_test_sentences, 1)
    # bleu_2 = bleu_metric(pre_sentences, target_test_sentences, 2)
    # bleu_3 = bleu_metric(pre_sentences, target_test_sentences, 3)
    # bleu_4 = bleu_metric(pre_sentences, target_test_sentences, 4)
    # print("BLEU-1:%s",str(bleu_1))
    # print("BLEU-2:%s", str(bleu_2))
    # print("BLEU-3:%s", str(bleu_3))
    # print("BLEU-4:%s", str(bleu_4))
    #
    # rouge1, rouge2, rougeL, METEOR = rouge(pre_sentences, target_test_sentences)
    # print("ROUGE-1:%s",str(rouge1))
    # print("ROUGE-2:%s",str(rouge2))
    # print("ROUGE-L:%s",str(rougeL))
    # print("METEOR:%s", str(METEOR))
    # print(pre_sentences)




