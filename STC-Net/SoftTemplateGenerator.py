import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from sklearn import model_selection
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining ,BertAdam
from rouge import Rouge
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import time
from SoftTemplate_CopyNet.soft_template_copynet.utils import tokens_to_seq
from utils import seq_to_string, to_np, trim_seqs
from torch.autograd import Variable
from pythonrouge.pythonrouge import Pythonrouge
from torch.utils.data import DataLoader
import pandas as pd
import torch
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from smart_open import smart_open, logger

from SoftTemplate_CopyNet.soft_template_copynet.SoftTemplateEncoder import SoftTemplateEncoder
from SoftTemplate_CopyNet.soft_template_copynet.dataset import SequencePairDataset
from SoftTemplate_CopyNet.soft_template_copynet.model.encoder_decoder import EncoderDecoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn import model_selection
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining ,BertAdam

superlative_candidates = ["most", "least", "first", "top", "last", "largest", "latest", "highest", "smallest",
                          "tallest", "longest", "winningest",
                          "fewest", "earliest", "lowest", "farthest", "biggest", "youngest", "oldest", "greatest",
                          "worst", "best", "fastest", "first",
                          "shortest", "closest", "heaviest", "deepest", "furthest", "slowest", "qucikest"]
gradable_candidates = ["more", "less", "same", "taller", "higher", "smaller", "longer", "equal", "older", "younger",
                       "heavier", "newer",
                       "earlier", "greater", "better", "worse", "farther", "shorter", "bigger", "closer", "larger",
                       "faster", "slower",
                       "deeper"]

def generateTemplate(feature, question):
    question_template = []
    for question_word in question:
        if question_word in [feature[4],feature[-2], feature[-3]]:
            question_template.append("<UNK>")
        else:
            question_word = question_word.replace(",","")
            question_template.append(question_word)
    return question_template

def features_data_preprocess(data_path):
    features = []
    questions = []
    comparative_question_types = []
    comparative_classified_data = {'0':[],
                                   '1':[],
                                   '2':[],
                                   '3':[],
                                   '4':[],
                                   '5':[]}

    comparative_data = pd.read_csv(data_path)
    print(comparative_data.columns)
    for row in comparative_data.iterrows():
        feature = list(row[-1][1:-1])
        question = row[-1][-1].split()
        if feature[0] in gradable_candidates:
            if feature[3] == "comparee":
                comparative_question_types.append(0)
                comparative_classified_data['0'].append(generateTemplate(feature, question))
            elif feature[3] == "index":
                comparative_question_types.append(1)
                comparative_classified_data['1'].append(generateTemplate(feature, question))
            elif feature[3] == "difference":
                comparative_question_types.append(2)
                comparative_classified_data['2'].append(generateTemplate(feature, question))
            else:
                comparative_question_types.append(5)
                comparative_classified_data['5'].append(generateTemplate(feature, question))
        elif feature[0] in superlative_candidates:
            if feature[3] == "comparee":
                comparative_question_types.append(3)
                comparative_classified_data['3'].append(generateTemplate(feature, question))
            elif feature[3] == "index":
                comparative_question_types.append(4)
                comparative_classified_data['4'].append(generateTemplate(feature, question))
            else:
                comparative_question_types.append(5)
                comparative_classified_data['5'].append(generateTemplate(feature, question))
        else:
            comparative_question_types.append(5)
            comparative_classified_data['5'].append(generateTemplate(feature, question))
        feature_seq = ""
        for i in feature:
            if not i:
                feature_seq = feature_seq + "#" + " "
            else:
                feature_seq = feature_seq + str(i) + " "

        features.append(feature_seq.strip())
        questions.append(" ".join(question))

    return features, questions, comparative_question_types, comparative_classified_data

def corpus_rouge(all_target_seqs, all_output_seqs):
    pair_target = []
    pair_output = []
    for target_seq in all_target_seqs:
        str_target = [str(i) for i in target_seq[0]]
        pair_target.append(" ".join(str_target))
    for output_seq in all_output_seqs:
        str_output = [str(i) for i in output_seq]
        pair_output.append(" ".join(str_output))
    meteor = 0
    for index in range(len(pair_target)):
        meteor += round(meteor_score([pair_target[index]], pair_output[index]), 4)
    predict_questions = []
    references = []
    for target_sentence in pair_target:
        references.append([[target_sentence]])
    for predict_question in pair_output:
        predict_questions.append([predict_question])

    rouge = Pythonrouge(summary_file_exist=False,
                        summary=predict_questions, reference=references,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()

    ROUGE1 = float(score['ROUGE-1'])
    ROUGE2 = float(score['ROUGE-2'])
    ROUGEL = float(score['ROUGE-L'])
    METEOR = float(meteor / len(pair_target))
    return [ROUGE1, ROUGE2, ROUGEL, METEOR]

def evaluate(encoder_decoder: EncoderDecoder, data_loader):

    loss_function = torch.nn.NLLLoss(ignore_index=0, reduce=False) # what does this return for ignored idxs? same length output?

    losses = []
    all_output_seqs = []
    all_target_seqs = []

    for batch_idx, (input_idxs, target_idxs, _, _, comparative_types) in enumerate(tqdm(data_loader)):
        input_lengths = (input_idxs != 0).long().sum(dim=1)
        sorted_lengths, order = torch.sort(input_lengths, descending=True)
        input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)], volatile=True)
        target_variable = Variable(target_idxs[order, :], volatile=True)
        batch_size = input_variable.shape[0]

        output_log_probs, output_seqs, attn_scores = encoder_decoder(input_variable, list(sorted_lengths),comparative_types)

        target_sentence = [[list(seq[seq > 0])] for seq in to_np(target_variable)]
        # print(target_sentence)
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

        flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
        batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
        losses.extend(list(to_np(batch_losses)))

    mean_loss = len(losses) / sum(losses)

    bleu_1_score = corpus_bleu(all_target_seqs, all_output_seqs, weights=(1,0,0,0), smoothing_function=SmoothingFunction().method1)
    bleu_2_score = corpus_bleu(all_target_seqs, all_output_seqs, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu_3_score = corpus_bleu(all_target_seqs, all_output_seqs, weights=(0.4, 0.3, 0.3, 0), smoothing_function=SmoothingFunction().method1)
    bleu_4_score = corpus_bleu(all_target_seqs, all_output_seqs, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    ROUGE = corpus_rouge(all_target_seqs, all_output_seqs)
    return mean_loss, [bleu_1_score,bleu_2_score,bleu_3_score,bleu_4_score],ROUGE

def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = './model/' + model_name + '/'
    result=[]
    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch %i' % epoch, flush=True)
        losses = 0.0
        index = 0
        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens, comparative_type) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length
            index = index + 1
            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)

            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])
            optimizer.zero_grad()
            output_log_probs, output_seqs, attn_scores = encoder_decoder(input_variable,
                                                            list(sorted_lengths),
                                                            comparative_type=comparative_type,
                                                            targets=target_variable,
                                                            keep_prob=keep_prob,
                                                            teacher_forcing=teacher_forcing
                                                            )

            batch_size = input_variable.shape[0]

            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))
            # batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
            # losses.extend(list(to_np(batch_losses)))
            float(batch_loss)
            losses= losses + batch_loss
            batch_loss.backward()
            optimizer.step()

            global_step += 1
        mean_loss = losses / index
        print(mean_loss)
        val_loss, val_bleu_score, val_rouge_val = evaluate(encoder_decoder, val_data_loader)
        print("val_loss:")
        print(val_loss)
        print("bleu:")
        print(val_bleu_score)
        print("rouge:")
        print(val_rouge_val)
        result.append([str(epoch), str(mean_loss), str(val_bleu_score), str(val_rouge_val)])
        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))

        print('-' * 100, flush=True)
    print(result)
    df = pd.DateFrame(result)
    df.to_csv("copynet_result.csv", index=False)

def initSoftTemplateEmbedding(soft_template_encoder, hidden_size, lang):
    features, questions, comparative_question_types, comparative_classified_data = features_data_preprocess(
        "../../data/comparative_table_and_features_data.csv")
    comparative_classified_soft_template = {}
    for key in comparative_classified_data:
        temp_comparative_type_tensor = torch.zeros(1, 1, 2*hidden_size)
        for masked_question in comparative_classified_data[key]:
            input_ids = tokens_to_seq(masked_question, lang.tok_to_idx, len(masked_question), True)
            input_ids = input_ids.to(device)
            input = torch.unsqueeze(input_ids,0)
            try:
                output, hidden = soft_template_encoder(input, [torch.tensor(len(masked_question))])
                b = torch.mean(hidden, dim=0)
                # b = hidden.detach().numpy()
                b = b.unsqueeze(0)
                b = b.detach().numpy()
                temp_comparative_type_tensor = temp_comparative_type_tensor + b
            except:
                continue

        comparative_type_tensor = temp_comparative_type_tensor/len(comparative_classified_data[key])
        comparative_classified_soft_template[key] = comparative_type_tensor
    return comparative_classified_soft_template


def main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed=42):

    model_path = './model/' + model_name + '/'

    # TODO: Change logging to reflect loaded parameters

    print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(model_path + model_name + '.pt')

        print("creating training and validation datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(maxlen = max_length,
                                            lang=encoder_decoder.lang,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,

                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

        val_dataset = SequencePairDataset(maxlen=max_length,
                                          lang=encoder_decoder.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          use_extended_vocab=(encoder_decoder.decoder_type=='copy'))

    else:
        os.mkdir(model_path)

        print("creating training and validation datasets", flush=True)
        train_dataset = SequencePairDataset(vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            is_val=False,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=(decoder_type=='copy'))

        val_dataset = SequencePairDataset(lang=train_dataset.lang,
                                          use_cuda=use_cuda,
                                          is_val=True,
                                          val_size=val_size,
                                          seed=seed,
                                          use_extended_vocab=(decoder_type=='copy'))

        soft_template_encoder = SoftTemplateEncoder(train_dataset.lang,  embedding_size, hidden_size*2)
        soft_template_encoder.to(device)
        comparative_classified_soft_template = initSoftTemplateEmbedding(soft_template_encoder, embedding_size, train_dataset.lang)

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         embedding_size,
                                         hidden_size,
                                         decoder_type,
                                         comparative_classified_soft_template)

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    encoder_decoder = encoder_decoder.to(device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)


    train(encoder_decoder,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length)


if __name__ == '__main__':

    features, questions, comparative_question_types, comparative_classified_data = features_data_preprocess(
        "../../data/comparative_table_and_features_data.csv")
    print(len(features))
    print(len(questions))
    print(len(comparative_question_types))
    model_name = "202203333"
    ues_cuda = True
    batch_size = 8
    scheduled_teacher_forcing = 0.7
    keep_prob = 0.7
    val_size = 0.2
    lr = 0.0001
    decoder_type = "copy"
    vocab_limit = 5000
    hidden_size = 100
    embedding_size = 100
    max_length = 50
    epochs = 100
    writer = SummaryWriter('./logs/%s_%s' % (model_name, str(int(time.time()))))
    if scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0 / epochs)
    else:
        schedule = np.ones(epochs) * scheduled_teacher_forcing
    # main(args.model_name, args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length)
    # main(str(int(time.time())), args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length)
    main(model_name, ues_cuda, batch_size, schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size,
         embedding_size, max_length)
