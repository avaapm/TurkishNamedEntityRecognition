# Author  : Oguzhan Ozcelik
# Date    : 14.07.2022
# Subject : CRF and SVM training for Named Entity Recognition
# Models  : SVM, CRF
# Source  : https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2

import pickle
import os
import numpy as np
import pandas as pd
import csv
import fasttext

# IMPORT TENSORFLOW LIBRARIES
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.linear_model import SGDClassifier
import sklearn_crfsuite
from zemberek import TurkishMorphology
from collections import Counter
import random as rn

morphology = TurkishMorphology.create_with_defaults()

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


def stemmer(word: str):
    results = morphology.analyze(word)
    if not results.analysis_results:
        stem = word
    else:
        stem = results.analysis_results.__getitem__(0).item.root
    return stem


def pos_tagger(word: str):
    result = morphology.analyze(word).analysis_results
    analyzed = []
    pos = []
    final_pos = ''
    for i in result:
        analyzed.append(str(i))
    # print("Word: ", word)
    if analyzed:
        pos.append(result.__getitem__(0).item.primary_pos.short_form)
        pos.append(result.__getitem__(0).item.secondary_pos.short_form)
        # print(pos)
        for p in pos:
            if p == 'Prop' or p == 'Num':
                final_pos = p
        if final_pos == '':
            final_pos = str(result.__getitem__(0).item.primary_pos.short_form)
    else:
        final_pos = 'Noun'
    if word.find("'") != -1:
        final_pos = 'Apost'
    # print('POS', final_pos)
    return final_pos


def get_features(index: int, word: str, sentence: list):
    prev_word = 'BOS'  # beginning of the sentence
    next_word = 'EOS'  # end of the sentence
    if len(sentence) > index+1:
        next_word = sentence[index+1]
    if index-1 > 0:
        prev_word = sentence[index-1]
    tag = pos_tagger(word)
    prev_tag = pos_tagger(prev_word)
    next_tag = pos_tagger(next_word)
    dic = {
        "stem": stemmer(word),
        "postag": tag,
        "nextwordtag": next_tag,
        "previoustag": prev_tag,
        "title": word.istitle(),
        'lower': word.islower(),
        'upper': word.isupper(),
    }
    return dic


def get_fasttext(ft, word):
    dic = {str(ix): ft.get_word_vector(word)[ix] for ix in range(len(ft.get_word_vector(word)))}
    return dic


def sent2features(sent):
    return [get_features(index, sent[index], sent) for index in range(len(sent))]


def sent2features_fasttext(sent, ft):
    return [get_fasttext(ft, word) for word in sent]


def print_transitions(trans_features, dataset, title, feature):
    with open(dataset + '/' + feature + '/' + 'transitions_' + title + '_' + feature + '.txt', 'w', encoding='utf-8') as f:
        f.write(title + ' ' + dataset + '\n')
        for (label_from, label_to), weight in trans_features:
            f.write("%-6s -> %-7s %0.6f \n" % (label_from, label_to, weight))
            #print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def print_state_features(state_features, dataset, title, feature):
    with open(dataset + '/' + feature + '/' + 'features_' + title + '_' + feature + '.txt', 'w', encoding='utf-8') as f:
        f.write(title + ' ' + dataset + '\n')
        for (attr, label), weight in state_features:
            f.write("%0.6f %-8s %s \n" % (weight, label, attr))
            #print("%0.6f %-8s %s" % (weight, label, attr))


def shuffle_dataset(a, b, random_seed):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=random_seed).permutation(len(a))
    return a[p], b[p]


news = ["data/news/", "news_results"]
wikiann = ["data/wikiann/", "wikiann_results"]
fbner = ["data/fbner/", "fbner_results"]
tweet = ["data/twner/", "twner_results"]
atis = ["data/atisner/", "atisner_results"]

paths = [news, wikiann, fbner, tweet, atis]  # news, wikiann, fbner, tweet, atis

models = ["CRF"]  # Add model either 'SVM' or 'CRF'
features = ["handcrafted"]  # Add feature either 'fasttext' or 'handcrafted'

seeds = [rn.randint(0, 2000) for i in range(10)]

for path in paths:
    print('DATASET:\t', path[1])

    #  MAKE FOLDER
    if not os.path.isdir(path[1]):
        os.mkdir(path[1])

    data_train = pd.read_csv(path[0] + "train_SENTENCED.txt", encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
    data_test = pd.read_csv(path[0] + "test_SENTENCED.txt", encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)

    y_train = data_train.Tag.values
    classes = np.unique(y_train)
    classes = classes.tolist()

    getter = SentenceGetter(data_train)
    sent = getter.get_next()
    sentences_TRAIN = getter.sentences

    TRAIN_SENTENCES, TRAIN_LABELS = [], []
    for sentence in sentences_TRAIN:
        words, tags = [], []
        for word in sentence:
            words.append(word[0])
            tags.append(word[1])
        TRAIN_SENTENCES.append(words)
        TRAIN_LABELS.append(tags)

    getter = SentenceGetter(data_test)
    sent = getter.get_next()
    sentences_TEST = getter.sentences

    TEST_SENTENCES, TEST_LABELS = [], []
    for sentence in sentences_TEST:
        words, tags = [], []
        for word in sentence:
            words.append(word[0])
            tags.append(word[1])
        TEST_SENTENCES.append(words)
        TEST_LABELS.append(tags)

    test_set = path[1] + '/test_words.txt'
    with open(test_set, 'w') as f:
        wr = csv.writer(f, delimiter=' ')
        wr.writerows(TEST_SENTENCES)

    for mod in models:
        print("SEEDS: ", seeds)
        for feature in features:
            if not os.path.isdir(path[1] + '/' + feature):
                os.mkdir(path[1] + '/' + feature)
            
            print('Feature: ' + feature)
            if mod == 'CRF':
                if feature == 'handcrafted':
                    X_train = [sent2features(s) for s in tqdm(TRAIN_SENTENCES, total=len(TRAIN_SENTENCES))]
                    print(X_train[:10])
                    X_test = [sent2features(s) for s in tqdm(TEST_SENTENCES, total=len(TEST_SENTENCES))]
                else:
                    ft = fasttext.load_model('fasttext/cc.tr.300.bin')
                    X_train = [sent2features_fasttext(s, ft) for s in tqdm(TRAIN_SENTENCES, total=len(TRAIN_SENTENCES))]
                    X_test = [sent2features_fasttext(s, ft) for s in tqdm(TEST_SENTENCES, total=len(TEST_SENTENCES))]
                y_train = [j for j in TRAIN_LABELS]

            else:  # SVM
                train_fasttext = [i for j in TRAIN_SENTENCES for i in j]
                test_fasttext = [i for j in TEST_SENTENCES for i in j]

                if feature == 'handcrafted':
                    X_train, X_test = [], []
                    for sentence in tqdm(range(len(TRAIN_SENTENCES)), total=len(TRAIN_SENTENCES)):
                        for i in range(len(TRAIN_SENTENCES[sentence])):
                            X_train.append(get_features(i, TRAIN_SENTENCES[sentence][i], TRAIN_SENTENCES[sentence]))
                    for sentence in tqdm(range(len(TEST_SENTENCES)), total=len(TEST_SENTENCES)):
                        for i in range(len(TEST_SENTENCES[sentence])):
                            X_test.append(get_features(i, TEST_SENTENCES[sentence][i], TEST_SENTENCES[sentence]))
                    v = DictVectorizer(sparse=False, dtype=bool)
                    X_train = v.fit_transform(X_train)
                    X_test = v.transform(X_test)
                    print(X_train[:10])
                else:
                    ft = fasttext.load_model('fasttext/cc.tr.300.bin')
                    X_train = np.zeros((len(train_fasttext), 300))
                    counter = 0
                    for word in tqdm(train_fasttext, total=len(train_fasttext)):
                        embedding_vector = ft.get_word_vector(word)
                        X_train[counter] = embedding_vector
                        counter += 1
                    X_test = np.zeros((len(test_fasttext), 300))
                    counter = 0
                    for word in tqdm(test_fasttext, total=len(test_fasttext)):
                        embedding_vector = ft.get_word_vector(word)
                        X_test[counter] = embedding_vector
                        counter += 1
                y_train = np.array([i for j in TRAIN_LABELS for i in j], dtype=object)
            
            f1 = []
            recall = []
            precision = []
            times = []
            
            for repeat in range(10):
                seed = seeds[repeat]
                X_train, y_train = shuffle_dataset(np.array(X_train), np.array(y_train), seed)
                print(y_train[0:5])
                print("SEED: ", seed)
                print('REPEAT: ', repeat)
                if mod == 'SVM':
                    print(mod + " STARTS...")
                    sgd = SGDClassifier(shuffle=False, n_jobs=-1, max_iter=1000, loss='hinge')
                    chunk = 1000
                    for ch in tqdm(range(int(len(X_train) / chunk)), total=int(len(X_train) / chunk)):
                        sgd.partial_fit(X_train[chunk * ch:chunk * (ch + 1)], y_train[chunk * ch:chunk * (ch + 1)], classes)
                        latest = chunk * (ch + 1)
                    sgd.partial_fit(X_train[latest:], y_train[latest:], classes)

                    new_classes = classes.copy()
                    new_classes.pop()
                    preds = sgd.predict(X_test)

                elif mod == 'CRF':
                    print(mod + " STARTS...")
                    crf = sklearn_crfsuite.CRF(
                        algorithm='l2sgd',
                        # c1=0.1,
                        c2=0.1,
                        max_iterations=100,
                        all_possible_transitions=True,
                        verbose=True
                    )
                    try:
                        crf.fit(X_train, y_train)
                    except AttributeError:
                        pass
            
                    # save
                    with open('model.pkl', 'wb') as f:
                        pickle.dump(crf, f)
            
                    pred_list = crf.predict(X_test)
                    if repeat == 0:
                        print_transitions(Counter(crf.transition_features_).most_common(20), path[1], "likely", feature)
                        print_transitions(Counter(crf.transition_features_).most_common()[-20:], path[1], "unlikely", feature)
                        print_state_features(Counter(crf.state_features_).most_common(20), path[1], "likely", feature)
                        print_state_features(Counter(crf.state_features_).most_common()[-20:], path[1], "unlikely", feature)
            
                if mod == 'CRF':
                    report = classification_report(y_pred=pred_list, y_true=TEST_LABELS, digits=4)
                    print(report)
                    class_report = path[1] + '/' + feature + '/classification_report_' + mod + '_' + feature + '_' + str(repeat) + '.txt'
                    text_file = open(class_report, "w")
                    text_file.write(report)
                    text_file.close()
            
                else:
                    pred_list = []
                    old_len = 0
                    for i in TEST_LABELS:
                        pred_list.append(list(preds[old_len:len(i)+old_len]))
                        old_len += len(i)

                    report = classification_report(y_pred=pred_list, y_true=TEST_LABELS, digits=4)
                    print(report)
                    class_report = path[1] + '/' + feature + '/classification_report_' + mod + '_' + feature + '_' + str(repeat) + '.txt'
                    text_file = open(class_report, "w")
                    text_file.write(report)
                    text_file.close()

                pred_text = path[1] + '/' + feature + '/pred_' + mod + '_' + feature + '_' + str(repeat) + '.txt'
                with open(pred_text, 'w') as f:
                    wr = csv.writer(f, delimiter=' ')
                    wr.writerows(pred_list)
                f1.append(f1_score(TEST_LABELS, pred_list, average='weighted'))
                recall.append(recall_score(TEST_LABELS, pred_list, average='weighted'))
                precision.append(precision_score(TEST_LABELS, pred_list, average='weighted'))

            f1_end = str(round(sum(f1) / 10, 5))
            rec_end = str(round(sum(recall) / 10, 5))
            pre_end = str(round(sum(precision) / 10, 5))

            with open(path[1] + '/' + feature + '/FINAL_RESULT_' + mod + '_' + feature + '.txt', 'w') as f:
                for i in range(len(f1)):
                    text = str(i + 1) + "-Repeat --> F1 score: " + str(f1[i]) + "  Precision: " + str(precision[i]) + \
                           "  Recall: " + str(recall[i]) + '\n'
                    f.write(text)
                text = "\nAvg. 10 Repeat F1-score: " + f1_end + "  Precision: " + pre_end + "  Recall: " + rec_end
                f.write(text)

            ground_text = path[1] + '/true_' + mod + '_' + feature + '.txt'
            with open(ground_text, 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                wr.writerows(TEST_LABELS)

