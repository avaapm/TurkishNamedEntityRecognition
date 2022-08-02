# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Training and testing of CRF model.

import json
import pickle
import numpy as np
import sklearn_crfsuite

from src.utils import DatasetLoader
from tqdm import tqdm
from zemberek import TurkishMorphology
from seqeval.metrics import classification_report


class CRF_MODELS:
    def __init__(self, data_path, model_path, feature):
        if feature not in ['hc', 'ft']:
            raise ValueError("Invalid feature type. Expected one of: %s" % ['hc', 'ft'])
        with open('/home/sysadmin/Desktop/PyCharm_Projects/ipm_github/src/configs/crf_config.json') as f:
            config = json.load(f)

        self.data_path = data_path  # input data path either train or test
        self.model_path = model_path  # model will be saved to this path or loaded for test
        self.feature = feature
        self.morphology = TurkishMorphology.create_with_defaults()
        self.max_iter = config["num_train_iteration"]
        self.c2 = config["l2_coeff"]
        self.algorithm = config["train_algorithm"]
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
        dataset = DatasetLoader(self.data_path + 'train.tsv', None)
        dataset.file_loader(train=True)
        train_sentences = dataset.train_words
        train_labels = dataset.train_tags

        X_train = []

        if self.feature == "hc":
            X_train = [self.sent2features(s) for s in tqdm(train_sentences, total=len(train_sentences))]

        elif self.feature == "ft":
            X_train = [self.sent2features_fasttext(s, self.ft) for s in tqdm(train_sentences, total=len(train_sentences))]

        y_train = [j for j in train_labels]

        X_train, y_train = self.shuffle_dataset(np.array(X_train), np.array(y_train))

        crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c2=self.c2,
            max_iterations=self.max_iter,
            all_possible_transitions=True,
            verbose=True
        )

        try:
            crf.fit(X_train, y_train)
        except AttributeError:
            pass

        with open(self.model_path + 'crf_tr_model.pkl', 'wb') as f:
            pickle.dump(crf, f)

    def evaluate(self, result_path):

        with open(self.model_path + 'crf_tr_model.pkl', 'rb') as f:
            crf = pickle.load(f)

        dataset = DatasetLoader(None, self.data_path + 'test.tsv')
        dataset.file_loader(train=False)
        test_sentences = dataset.test_words
        test_labels = dataset.test_tags

        X_test = []

        if self.feature == "hc":
            X_test = [self.sent2features(s) for s in tqdm(test_sentences, total=len(test_sentences))]

        elif self.feature == "ft":
            X_test = [self.sent2features_fasttext(s, self.ft) for s in tqdm(test_sentences, total=len(test_sentences))]

        pred_list = crf.predict(X_test)

        report = classification_report(test_labels, pred_list, digits=4)

        with open(result_path + 'final_result', 'w') as f:
            f.write(report)

        print(report)