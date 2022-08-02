# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Training and testing of SVM model.

import os
import json
import pickle
import numpy as np

from src.utils import DatasetLoader
from tqdm import tqdm
from zemberek import TurkishMorphology
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from seqeval.metrics import classification_report


class SVM_MODELS:
    def __init__(self, data_path, model_path, feature):
        if feature not in ['hc', 'ft']:
            raise ValueError("Invalid feature type. Expected one of: %s" % ['hc', 'ft'])
        with open('/src/configs/svm_config.json') as f:
            config = json.load(f)

        self.data_path = data_path  # input data path either train or test
        self.model_path = model_path  # model will be saved to this path or loaded for test
        self.feature = feature
        self.morphology = TurkishMorphology.create_with_defaults()
        self.max_iter = config["num_train_iteration"]
        self.loss = config["loss"]
        self.ft = config["fasttext_path"]

    def sent2features(self, sent):
        return [self.get_features(index, sent[index], sent) for index in range(len(sent))]

    def sent2features_fasttext(self, sent, ft):
        return [self.get_fasttext(ft, word) for word in sent]

    def pos_tagger(self, word: str):
        result = self.morphology.analyze(word).analysis_results
        analyzed = []
        pos = []
        final_pos = ''
        for i in result:
            analyzed.append(str(i))
        if analyzed:
            pos.append(result.__getitem__(0).item.primary_pos.short_form)
            pos.append(result.__getitem__(0).item.secondary_pos.short_form)
            for p in pos:
                if p == 'Prop' or p == 'Num':
                    final_pos = p
            if final_pos == '':
                final_pos = str(result.__getitem__(0).item.primary_pos.short_form)
        else:
            final_pos = 'Noun'
        if word.find("'") != -1:
            final_pos = 'Apost'
        return final_pos

    def stemmer(self, word: str):
        results = self.morphology.analyze(word)
        if not results.analysis_results:
            stem = word
        else:
            stem = results.analysis_results.__getitem__(0).item.root
        return stem

    def get_features(self, index: int, word: str, sentence: list):
        prev_word = 'BOS'  # beginning of the sentence
        next_word = 'EOS'  # end of the sentence
        if len(sentence) > index + 1:
            next_word = sentence[index + 1]
        if index - 1 > 0:
            prev_word = sentence[index - 1]
        tag = self.pos_tagger(word)
        prev_tag = self.pos_tagger(prev_word)
        next_tag = self.pos_tagger(next_word)
        dic = {
            "stem": self.stemmer(word),
            "postag": tag,
            "nextwordtag": next_tag,
            "previoustag": prev_tag,
            "title": word.istitle(),
            'lower': word.islower(),
            'upper': word.isupper(),
        }
        return dic

    def get_fasttext(self, ft, word):
        dic = {str(ix): ft.get_word_vector(word)[ix] for ix in range(len(ft.get_word_vector(word)))}
        return dic

    def shuffle_dataset(self, a, b):
        assert len(a) == len(b)
        p = np.random.RandomState().permutation(len(a))
        return a[p], b[p]

    def train(self):
        dataset = DatasetLoader(self.data_path + 'train_sentenced.tsv', None)
        dataset.file_loader(train=True)
        train_sentences = dataset.train_words
        train_labels = dataset.train_tags

        dataset_test = DatasetLoader(self.data_path + 'test_sentenced.tsv', None)
        dataset_test.file_loader(train=False)
        test_sentences = dataset.test_words

        if self.feature == "hc":
            X_train, X_test = [], []
            for sentence in tqdm(range(len(train_sentences)), total=len(train_sentences)):
                for i in range(len(train_sentences[sentence])):
                    X_train.append(self.get_features(i, train_sentences[sentence][i], train_sentences[sentence]))
            for sentence in tqdm(range(len(test_sentences)), total=len(test_sentences)):
                for i in range(len(test_sentences[sentence])):
                    X_test.append(self.get_features(i, test_sentences[sentence][i], test_sentences[sentence]))
            v = DictVectorizer(sparse=False, dtype=bool)

            X_train = v.fit_transform(X_train)
            X_test = v.transform(X_test)

            with open(self.model_path + "test_vectors", "wb") as fp:  # Pickle transformed test data
                pickle.dump(X_test, fp)

        elif self.feature == "ft":
            train_fasttext = [i for j in train_sentences for i in j]
            X_train = np.zeros((len(train_fasttext), 300))
            counter = 0
            for word in tqdm(train_fasttext, total=len(train_fasttext)):
                embedding_vector = self.ft.get_word_vector(word)
                X_train[counter] = embedding_vector
                counter += 1

        y_train = np.array([i for j in train_labels for i in j], dtype=object)
        classes = np.unique(y_train).tolist()

        X_train, y_train = self.shuffle_dataset(np.array(X_train), np.array(y_train))

        sgd = SGDClassifier(shuffle=False, n_jobs=-1, max_iter=1000, loss='hinge')
        chunk = 1000
        for ch in tqdm(range(int(len(X_train) / chunk)), total=int(len(X_train) / chunk)):
            sgd.partial_fit(X_train[chunk * ch:chunk * (ch + 1)], y_train[chunk * ch:chunk * (ch + 1)], classes)
            latest = chunk * (ch + 1)
        sgd.partial_fit(X_train[latest:], y_train[latest:], classes)

        with open(self.model_path + 'svm_model.pkl', 'wb') as f:
            pickle.dump(sgd, f)

    def evaluate(self, result_path):

        with open(self.model_path + 'svm_model.pkl', 'rb') as f:
            sgd = pickle.load(f)

        dataset = DatasetLoader(None, self.data_path + 'test.tsv')
        dataset.file_loader(train=False)
        test_sentences = dataset.test_words
        test_labels = dataset.test_tags

        if self.feature == "hc":
            with open(self.model_path + "test_vectors", "rb") as fp:  # Unpickling
                X_test = pickle.load(fp)

        elif self.feature == "ft":
            test_fasttext = [i for j in test_sentences for i in j]
            X_test = np.zeros((len(test_fasttext), 300))
            counter = 0
            for word in tqdm(test_fasttext, total=len(test_fasttext)):
                embedding_vector = self.ft.get_word_vector(word)
                X_test[counter] = embedding_vector
                counter += 1

        preds = sgd.predict(X_test)
        pred_list = []
        old_len = 0
        for i in test_labels:
            pred_list.append(list(preds[old_len:len(i) + old_len]))
            old_len += len(i)
        report = classification_report(test_labels, pred_list, digits=4)

        with open(result_path + 'final_result', 'w') as f:
            f.write(report)

        os.remove(self.model_path + "test_vectors")

        print(report)
