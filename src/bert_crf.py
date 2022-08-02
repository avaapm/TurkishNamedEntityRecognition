# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Fine-tuning and testing of BERT-CRF model via Hugging Face Trainer API and Transformers library

import os
import random
import torch
import shutil
import json
import torch.nn as nn
import pickle
import numpy as np

from src.utils import DatasetLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, \
    TrainingArguments, logging, ProgressCallback, IntervalStrategy, DataCollatorForTokenClassification
from torchcrf import CRF
from datasets import Dataset
from seqeval.metrics import classification_report


class BERT_CRF_MODEL:
    def __init__(self, data_path, model_path):
        logging.set_verbosity_error()
        os.environ["WANDB_DISABLED"] = "true"

        with open('src/configs/trm_models_config.json') as f:
            config = json.load(f)

        # Load dataset
        self.data_path = data_path  # input data path either train or test
        self.model_path = model_path  # model will be saved to this path or loaded for test
        self.tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased', local_files_only=True)
        self.learning_rate = config['learning_rate']  # float: learning rate
        self.max_seq_length = config['max_seq_len']  # int: sequence length
        self.num_train_epochs = config['num_train_epochs']  # int: epoch number
        self.train_batch_size = config['train_batch_size']  # int: batch size
        self.eval_batch_size = config['test_batch_size']  # int: batch size
        self.label2id = {}
        self.id2label = {}

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        counter = -1
        SEP_ix = word_ids[1:].index(None) + 1  # find the end of sentence index
        for word_id in word_ids:
            counter += 1
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = self.label2id['[MASK]'] if word_id is None else labels[word_id]
                new_labels.append(label)

            elif word_id is None:
                # Special token
                if counter == 0:
                    new_labels.append(self.label2id['[CLS]'])  # add CLS token for the start of the sentence
                else:
                    new_labels.append(self.label2id['[PAD]'])  # add PAD token for padding
            else:
                # Same word as previous token
                new_labels.append(self.label2id['[MASK]'])  # add MASK token for subwords

        new_labels[SEP_ix] = self.label2id['[SEP]']  # add SEP token for the end of the sentence
        return new_labels

    def tokenize(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True)
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def train(self):
        dataset = DatasetLoader(self.data_path + 'train.tsv', None)
        train_df, self.label2id, self.id2label = dataset.bert_crf_loader(train=True, label2id={}, id2label={})

        global label2id
        label2id = self.label2id

        train_dataset = Dataset.from_pandas(train_df)
        model = BERT_CRF(pretrained_path='dbmdz/bert-base-turkish-cased', num_labels=len(self.label2id))
        train_dataset = train_dataset.map(self.tokenize, batched=True, remove_columns=['tokens', 'ner_tags'])
        train_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        training_args = TrainingArguments(
            output_dir=self.model_path,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            disable_tqdm=True,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=50,
            seed=random.randint(1, 2000),
            save_strategy=IntervalStrategy.NO,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = BERT_CRF_Trainer(model=model,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   data_collator=data_collator,
                                   tokenizer=self.tokenizer,
                                   callbacks=[ProgressCallback])
        trainer.train()

        torch.save(model.state_dict(), self.model_path + 'berturk_crf')

        with open(self.model_path + 'label2id.pkl', 'wb') as f:
            pickle.dump(self.label2id, f)

        with open(self.model_path + 'id2label.pkl', 'wb') as f:
            pickle.dump(self.id2label, f)

        if os.path.exists(self.model_path + 'runs'):
            shutil.rmtree(self.model_path + 'runs')

    def evaluate(self, result_path):
        dataset = DatasetLoader(None, self.data_path + 'test.tsv')
        with open(self.model_path + 'label2id.pkl', 'rb') as f:
            self.label2id = pickle.load(f)
        with open(self.model_path + 'id2label.pkl', 'rb') as f:
            self.id2label = pickle.load(f)

        global label2id
        label2id = self.label2id

        test_df = dataset.bert_crf_loader(train=False, label2id=self.label2id, id2label=self.id2label)
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(self.tokenize, batched=True, remove_columns=['tokens', 'ner_tags'])
        test_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        model = BERT_CRF(pretrained_path='dbmdz/bert-base-turkish-cased', num_labels=len(self.label2id))
        model.load_state_dict(torch.load(self.model_path + 'berturk_crf'))
        trainer = BERT_CRF_Trainer(model=model, callbacks=[ProgressCallback])
        pred = trainer.predict(test_dataset)

        true_labels = pred.label_ids.tolist()
        preds = pred.predictions.astype(np.int64).tolist()

        true_labels = [[self.id2label.get(ele, ele) for ele in lst] for lst in true_labels]
        preds = [[self.id2label.get(ele, ele) for ele in lst] for lst in preds]

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

        report = classification_report(post_true, post_pred, digits=4)
        print(report)
        with open(result_path + 'final_result', 'w') as f:
            f.write(report)


class BERT_CRF(nn.Module):
    def __init__(self, pretrained_path='dbmdz/bert-base-turkish-cased',
                 num_labels=7):
        super(BERT_CRF, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModelForTokenClassification.from_pretrained(pretrained_path,
                                                                    num_labels=self.num_labels,
                                                                    local_files_only=True)
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

        logits = bert_out[0]
        return logits

    def to(self, device):
        self.bert = self.bert.to(device)
        self.crf = self.crf.to(device)
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