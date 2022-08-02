# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Data preprocessing code for several models.

import csv
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                     s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class DatasetLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_words = []
        self.train_tags = []
        self.test_words = []
        self.test_tags = []
        self.information = None
        self.lstm_len = 50

    def file_loader(self, train: bool):
        if train:
            train_tsv = open(self.train_path)
            train_list = csv.reader(train_tsv, delimiter="\t", quoting=csv.QUOTE_NONE)

            for row in train_list:  # skip first line
                break

            sentence = 'Sentence: 0'
            temp_words, temp_tags = [], []
            for row in train_list:
                if row[0] == sentence:
                    temp_words.append(row[1])
                    temp_tags.append(row[2])
                else:
                    sentence = row[0]
                    self.train_words.append(temp_words)
                    self.train_tags.append(temp_tags)
                    temp_words, temp_tags = [], []
                    temp_words.append(row[1])
                    temp_tags.append(row[2])
            self.train_words.append(temp_words)
            self.train_tags.append(temp_tags)

        else:
            test_tsv = open(self.test_path)
            test_list = csv.reader(test_tsv, delimiter="\t", quoting=csv.QUOTE_NONE)

            for row in test_list:  # skip first line
                break

            sentence = 'Sentence: 0'
            temp_words, temp_tags = [], []
            for row in test_list:
                if row[0] == sentence:
                    temp_words.append(row[1])
                    temp_tags.append(row[2])
                else:
                    sentence = row[0]
                    self.test_words.append(temp_words)
                    self.test_tags.append(temp_tags)
                    temp_words, temp_tags = [], []
                    temp_words.append(row[1])
                    temp_tags.append(row[2])

            self.test_words.append(temp_words)
            self.test_tags.append(temp_tags)

            assert (len(self.train_words) == len(self.train_tags)), 'Train file error: mismatch words and labels'
            assert (len(self.test_words) == len(self.test_tags)), 'Test file error: mismatch words and labels'

    def bilstm_loader(self, train: bool, word2ix_test: dict, tag2ix_test: dict):
        if train:
            data = pd.read_csv(self.train_path, encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
            getter = SentenceGetter(data)
            sentences = getter.sentences
            max_len = self.lstm_len
            X = [[w[0] for w in s] for s in sentences]
            X_data = []
            for seq in X:
                new_seq = []
                for i in range(max_len):
                    try:
                        new_seq.append(seq[i])
                    except:
                        new_seq.append("<PAD>")
                X_data.append(new_seq)
            self.train_tags = sorted(list(set(data['Tag'].tolist())))
            self.train_words = sorted(list(set(data['Word'].tolist())))
            extras = ['<UNK>', '<PAD>']
            extras.extend(self.train_words)

            tag2index = {t: i for i, t in enumerate(self.train_tags)}
            word2index = {t: i for i, t in enumerate(extras)}
            y = [[tag2index[w[1]] for w in s] for s in sentences]
            y_data = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2index['O'])
            return X_data, y_data, tag2index, word2index

        else:
            data = pd.read_csv(self.test_path, encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
            getter = SentenceGetter(data)
            sentences = getter.sentences
            max_len = self.lstm_len
            X = [[w[0] for w in s] for s in sentences]
            X_data = []
            for seq in X:
                new_seq = []
                for i in range(max_len):
                    try:
                        new_seq.append(seq[i])
                    except:
                        new_seq.append("<PAD>")
                X_data.append(new_seq)

            X_test = []
            for sent in X_data:
                temp = []
                for word in sent:
                    if word not in word2ix_test.keys():
                        temp.append('<UNK>')
                    else:
                        temp.append(word)
                X_test.append(temp)

            self.test_tags = tag2ix_test.keys()
            self.test_words = sorted(list(set(data['Word'].tolist())))

            y = [[tag2ix_test[w[1]] for w in s] for s in sentences]
            y_data = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2ix_test['O'])
            return X_test, y_data

    def transformer_loader(self, train: bool):
        if train:
            frame = pd.read_csv(self.train_path, encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
            frame.rename(columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"}, inplace=True)
            self.train_tags = list(set(frame['labels'].unique().tolist()))
            self.train_words = list(set(frame['words'].unique().tolist()))

        else:
            frame = pd.read_csv(self.test_path, encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
            frame.rename(columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"}, inplace=True)
            self.test_tags = list(set(frame['labels'].unique().tolist()))
            self.test_words = list(set(frame['words'].unique().tolist()))

        return frame

    def bert_crf_loader(self, train: bool, label2id: dict, id2label: dict):
        if train:
            self.file_loader(train=train)
            df = pd.DataFrame(list(zip(self.train_words, self.train_tags)), columns=['tokens', 'ner_tags'])
            unique_labels = list(
                set([lbl for sent in [i for i in df['ner_tags'].values.tolist()] for lbl in sent]))
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

            df['ner_tags'] = df['ner_tags'].map(lambda x: list(map(label2id.get, x)))
            return df, label2id, id2label

        else:
            self.file_loader(train=train)
            df = pd.DataFrame(list(zip(self.test_words, self.test_tags)), columns=['tokens', 'ner_tags'])
            df['ner_tags'] = df['ner_tags'].map(lambda x: list(map(label2id.get, x)))

            return df

