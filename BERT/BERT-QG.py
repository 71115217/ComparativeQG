import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from sklearn import model_selection
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining ,BertAdam
#Bert Generate One By One
from DataProcessForBert import getSourceSentences, getTargetSentences
from rouge import Rouge

# sourceSentences = getSourceSentences("superlative_source_train.txt")
# targetSentences = getTargetSentences("superlative_target_train.txt")
sourceSentences = getSourceSentences("source_all_spurious_SQL_data.txt")
targetSentences = getTargetSentences("target_all_questions_data.txt")
# input_text = "[CLS] which more medals Japan China [SEP] "
# target_text = ["which", "country", "got", "more", "medals", "japan" , "or", "china"]

# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = BertForMaskedLM.from_pretrained(modelpath)
model.to('cuda')

def get_example_pair(source_sentences,target_sentences):
    example_pairs = []
    index = 0
    for target_text in target_sentences:
        example_pair = dict()
        for i in range(0, len(target_text) + 1):
            tokenized_text = tokenizer.tokenize(source_sentences[index])
            tokenized_text.extend(target_text[:i])
            tokenized_text.append('[MASK]')
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

            loss_ids = [-1] * (len(tokenizer.convert_tokens_to_ids(tokenized_text)) - 1)
            if i == len(target_text):
                loss_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[SEP]'))[0])
            else:
                loss_ids.append(tokenizer.convert_tokens_to_ids([target_text[i]])[0])
            loss_tensors = torch.tensor([loss_ids]).to('cuda')

            example_pair[tokens_tensor] = loss_tensors
        example_pairs.append(example_pair)
        index = index + 1
        print(index)
    return example_pairs

example_pairs = get_example_pair(sourceSentences,targetSentences)
# Prepare optimizer
train_examples, test_examples = model_selection.train_test_split(example_pairs, test_size=0.3)

param_optimizer = list(model.named_parameters())
num_train_optimization_steps = int(len(train_examples) / 1)
# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=1e-5,
                             warmup=0.1,t_total=num_train_optimization_steps)

model.train()
for i in range(0,6):
  eveloss = 0
  print(i)
  index = 0
  for example_pair in train_examples:
      for k,v in example_pair.items():
        loss = model(k,masked_lm_labels=v)
        eveloss += loss.mean().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      index = index + 1
      print(index)
  print("step "+ str(i) + " : " + str(eveloss))

model.eval()
reference_sentences = []
predict_sentences = []

for example_pair in test_examples:
    target_sequence = list(example_pair.keys())[-1]
    print(target_sequence.cpu().numpy().tolist())
    target_sentence = tokenizer.convert_ids_to_tokens(target_sequence.cpu().numpy().tolist()[0])
    target_sentence = target_sentence[target_sentence.index('[SEP]')+1: -1]
    reference_sentences.append(target_sentence)
    # print(target_sentence.index('[SEP]'))
    predict_sentence = []
    new_input =  list(example_pair.keys())[0]
    while list(new_input.size())[1] < list(example_pair.values())[-1].size()[1]:
        predictions = model(new_input)
        # print(predictions[0][0])
        # print(predictions[0][0, len(predictions[0]) - 1])
        predicted_index = torch.argmax(predictions[0, len(predictions[0]) - 1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        predicted_tensor = torch.tensor([[predicted_index]]).cuda()
        end_tensor = torch.tensor([[103]]).cuda()
        new_input = new_input[:,:-1]

        print(new_input)
        new_input = torch.cat([new_input, predicted_tensor], 1)
        new_input = torch.cat([new_input, end_tensor],1)

        print(predicted_token)
        if ',' in predicted_token:
            break
        if '[SEP]' in predicted_token:
            break
        predict_sentence.append(predicted_token[0])
    predict_sentences.append(predict_sentence)
    print(predict_sentence)


def test(reference_sentences, predict_sentences):
    rouge = Rouge()
    meteor = 0
    bleu1 = 0
    bleu2 = 0
    bleu3 = 0
    bleu4 = 0
    for i in range(len(predict_sentences)):
        print(reference_sentences[i])
        print(predict_sentences[i])
        bleu1 += sentence_bleu([reference_sentences[i]], predict_sentences[i], (1, 0, 0, 0))
        bleu2 += sentence_bleu([reference_sentences[i]], predict_sentences[i], (0.5, 0.5, 0, 0))
        bleu3 += sentence_bleu([reference_sentences[i]], predict_sentences[i], (0.3, 0.3, 0.4, 0))
        bleu4 += sentence_bleu([reference_sentences[i]], predict_sentences[i], (0.25, 0.25, 0.25, 0.25))
    print("BLEU-1:%f", float(bleu1/len(predict_sentences)))
    print("BLEU-2:%f",float(bleu2 / len(predict_sentences)))
    print("BLEU-3:%f",float(bleu3 / len(predict_sentences)))
    print("BLEU-4:%f",float(bleu4 / len(predict_sentences)))
    rouge_avg_1 = 0
    rouge_avg_2 = 0
    rouge_avg_L = 0
    for index in range(len(predict_sentences)):
        rouge_score = rouge.get_scores(" ".join(predict_sentences[index]), " ".join(reference_sentences[index]))
        meteor += round(meteor_score([" ".join(predict_sentences[index])]," ".join(reference_sentences[index])), 4)
        rouge_1 = rouge_score[0]["rouge-1"]['f']
        rouge_2 = rouge_score[0]["rouge-2"]['f']
        rouge_L = rouge_score[0]["rouge-l"]['f']
        rouge_avg_1 = rouge_avg_1 + rouge_1
        rouge_avg_2 = rouge_avg_2 + rouge_2
        rouge_avg_L = rouge_avg_L + rouge_L
    ROUGE1 = float(rouge_avg_1/ len(predict_sentences))
    ROUGE2 = float(rouge_avg_2 / len(predict_sentences))
    ROUGEL = float(rouge_avg_L / len(predict_sentences))
    METEOR = float(meteor / len(predict_sentences))
    print("ROUGE-1:%s",str(ROUGE1))
    print("ROUGE-2:%s",str(ROUGE2))
    print("ROUGE-L:%s",str(ROUGEL))
    print("METEOR:%s", str(METEOR))
test(reference_sentences, predict_sentences)

