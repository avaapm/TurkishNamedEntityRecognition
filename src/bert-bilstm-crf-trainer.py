# Author  : Oguzhan Ozcelik
# Date    : 14.07.2022
# Subject : BERT fine-tuning with BiLSTM and CRF layer at the top of BERT architecture using Transformers library

import csv
import os
import random

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification, BertModel, Trainer, \
    TrainingArguments, logging, ProgressCallback, IntervalStrategy, DataCollatorForTokenClassification, SchedulerType
from torchcrf import CRF
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

logging.set_verbosity_error()
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

news = ["data/news/", "news_results"]
wikiann = ["data/wikiann/", "wikiann_results"]
fbner = ["data/fbner/", "fbner_results"]
twner = ["data/twner/", "twner_results"]
atisner = ["data/atisner/", "atisner_results"]

paths = [atisner, fbner, twner, news, wikiann]


class CoNLLReader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def read_conll(self, path):
        with open(path, encoding='utf-8') as train:
            text, labels = [], []
            sent, tags = [], []
            for line in train:
                line = line.rstrip().split()
                if not line:
                    assert len(sent) == len(tags)
                    text.append(sent)
                    labels.append(tags)
                    sent, tags = [], []
                else:
                    sent.append(line[0])
                    tags.append(line[1])

            df = pd.DataFrame(list(zip(text, labels)), columns=['tokens', 'ner_tags'])
        return df

    def collect_df(self):
        train_df = self.read_conll(self.train_path)
        test_df = self.read_conll(self.test_path)

        unique_labels = list(set([lbl for sent in [i for i in train_df['ner_tags'].values.tolist()] for lbl in sent]))
        label2id = {k: v for v, k in enumerate(sorted(unique_labels), start=4)}
        id2label = {v: k for v, k in enumerate(sorted(unique_labels), start=4)}
        label2id["[CLS]"] = 0
        id2label[0] = "[CLS]"
        label2id["[SEP]"] = 1
        id2label[1] = "[SEP]"
        label2id["[PAD]"] = 2
        id2label[2] = "[PAD]"
        label2id["[MASK]"] = 3
        id2label[3] = "[MASK]"
        train_df['ner_tags'] = train_df['ner_tags'].map(lambda x: list(map(label2id.get, x)))
        test_df['ner_tags'] = test_df['ner_tags'].map(lambda x: list(map(label2id.get, x)))

        return train_df, test_df, label2id, id2label


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, pretrained_path='dbmdz/bert-base-turkish-cased', config='dbmdz/bert-base-turkish-cased',
                 num_labels=7):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(config, local_files_only=True)
        self.bert = AutoModel.from_pretrained(pretrained_path,
                                              num_labels=self.num_labels,
                                              local_files_only=True)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(num_layers=1, input_size=self.config.hidden_size,
                              hidden_size=self.config.hidden_size // 2,
                              batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels,
                       batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        bert_out = self.bert(token_type_ids=token_type_ids,
                             input_ids=input_ids,
                             attention_mask=attention_mask)

        logits_ = bert_out[0]
        sequence_output = self.dropout(logits_)
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)
        return logits

    def to(self, device):
        self.bert = self.bert.to(device)
        self.crf = self.crf.to(device)
        self.dropout = self.dropout.to(device)
        self.bilstm = self.bilstm.to(device)
        self.classifier = self.classifier.to(device)
        return self


class BERT_CRF_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        logits = model(token_type_ids=inputs["token_type_ids"],
                       input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"])
        loss = None
        if labels is not None:
            mask = torch.ones_like(labels, dtype=torch.uint8)
            mask[labels == label2id['[PAD]']] = 0
            mask[labels == label2id['[MASK]']] = 0
            log_likelihood, tags = model.crf(logits, labels, mask=mask), model.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = model.crf.decode(logits)

        tags = {"logits": torch.Tensor(tags)}
        if return_outputs:
            return loss, tags
        else:
            return loss


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    counter = -1
    SEP_ix = word_ids[1:].index(None) + 1  # find the end of sentence index
    for word_id in word_ids:
        counter += 1
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = label2id['[MASK]'] if word_id is None else labels[word_id]
            new_labels.append(label)

        elif word_id is None:
            # Special token
            if counter == 0:
                new_labels.append(label2id['[CLS]'])  # add CLS token for the start of the sentence
            else:
                new_labels.append(label2id['[PAD]'])  # add PAD token for padding
        else:
            # Same word as previous token
            new_labels.append(label2id['[MASK]'])  # add MASK token for subwords

    new_labels[SEP_ix] = label2id['[SEP]']  # add SEP token for the end of the sentence
    return new_labels


def tokenize(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


for path in paths:
    precision = []
    recall = []
    f_scores = []
    if not os.path.isdir(path[1] + '/bert_bilstm_crf'):
        os.mkdir(path[1] + '/' + '/bert_bilstm_crf')

    seeds = random.sample(range(1, 1000), 10)

    for rep in range(10):
        print('DATA: ' + path[1] + '\tRepeat: ' + str(rep))
        data_handler = CoNLLReader(
            '/home/sysadmin/Desktop/PyCharm_Projects/ipm_revision/' + path[0] + 'train_CONLL.txt',
            '/home/sysadmin/Desktop/PyCharm_Projects/ipm_revision/' + path[0] + 'test_CONLL.txt')

        train_df, test_df, label2id, id2label = data_handler.collect_df()
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', local_files_only=True)
        model = BERT_BiLSTM_CRF(pretrained_path='dbmdz/bert-base-turkish-cased', num_labels=len(label2id))

        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['tokens', 'ner_tags'])
        test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['tokens', 'ner_tags'])
        train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        training_args = TrainingArguments(
            output_dir=path[1] + '/trainer_output',
            learning_rate=5e-5,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            disable_tqdm=True,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=50,
            seed=seeds[rep],
            save_strategy=IntervalStrategy.NO,
        )

        print("SEED: ", training_args.seed)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = BERT_CRF_Trainer(model=model,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   data_collator=data_collator,
                                   tokenizer=tokenizer,
                                   callbacks=[ProgressCallback])
        trainer.train()

        pred = trainer.predict(test_dataset)

        true_labels = pred.label_ids.tolist()
        preds = pred.predictions.astype(np.int64).tolist()

        true_labels = [[id2label.get(ele, ele) for ele in lst] for lst in true_labels]
        preds = [[id2label.get(ele, ele) for ele in lst] for lst in preds]

        true_labels_clean, preds_clean = [], []
        for ix in range(len(true_labels)):
            end_of_sent = true_labels[ix].index('[SEP]')
            true_labels_clean.append(true_labels[ix][1:end_of_sent])
            preds_clean.append(preds[ix][1:end_of_sent])

        post_true, post_pred = [], []
        for ix in range(len(true_labels_clean)):
            indices = [i for i, x in enumerate(true_labels_clean[ix]) if x == "[MASK]"]
            post_true.append([i for j, i in enumerate(true_labels_clean[ix]) if j not in indices])
            post_pred.append([i for j, i in enumerate(preds_clean[ix]) if j not in indices])

        f1 = f1_score(post_true, post_pred, average='weighted')
        prec = precision_score(post_true, post_pred, average='weighted')
        rec = recall_score(post_true, post_pred, average='weighted')

        precision.append(prec)
        recall.append(rec)
        f_scores.append(f1)
        report = classification_report(post_true, post_pred, digits=4)

        class_report = path[1] + '/bert_bilstm_crf/entities_repeat_' + str(rep)
        with open(class_report, "w") as text_file:
            text_file.write(report)

        pred = path[1] + '/bert_bilstm_crf/pred_repeat_' + str(rep)
        with open(pred, 'w') as f:
            wr = csv.writer(f, delimiter=' ')
            wr.writerows(post_pred)

        if rep == 0:
            test_set = path[1] + '/bert_bilstm_crf/test_set'
            with open(test_set, 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                wr.writerows(post_true)

    f_scores = [num for num in f_scores]
    precision = [num for num in precision]
    recall = [num for num in recall]

    result_txt = path[1] + '/bert_bilstm_crf/FINAL_RESULTS'

    with open(result_txt, 'w') as f:
        f.write("Repeat\tPrecision\tRecall\tF1_Score\n")
        for i in range(len(f_scores)):
            text = str(i) + "\t" + str(precision[i]) + "\t" + str(recall[i]) + "\t" + str(f_scores[i]) + "\n"
            f.write(text)
        text = "\nAvg.\t" + str(sum(precision) / len(precision)) + "\t" + \
               str(sum(recall) / len(recall)) + "\t" + str(sum(f_scores) / len(f_scores))
        f.write(text)
