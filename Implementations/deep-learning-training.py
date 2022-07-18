# Author  : Oguzhan Ozcelik
# Date    : 14.07.2022
# Subject : NER Task on BiLSTM, BiGRU, CNN, BiLSTM-CRF, BiGRU-CRF using Tensorflow keras library
# Models  : BiLSTM, BiGRU, CNN, BiLSTM-CRF, BiGRU-CRF
# Source  : https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/

import os
import time
import random as rn
import numpy as np
import pandas as pd
import csv

import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Bidirectional, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Sequential
from crf_head import CRF

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from zemberek import TurkishMorphology

#  *********************************************************************************************************************
#  HYPERPARAMETERS      HYPERPARAMETERS     HYPERPARAMETERS     HYPERPARAMETERS     HYPERPARAMETERS     HYPERPARAMETERS
#  *********************************************************************************************************************

SEED = rn.randint(1, 2000)  # random seed
EPOCH = 20
BATCH_SIZE = 16
EMBEDDING_SIZE = 300  # dimension of the embedding vector
MAX_LEN = 50  # max length of each sentence
DROPOUT = 0.0
LR_RATE = 0.005
LSTM_UNITS = 50
N_words = 50  # divide sentence in N_words
REPEAT = 10  # How many times do you repeat training

#  *********************************************************************************************************************
#  SEED     SEED        SEED        SEED        SEED        SEED        SEED        SEED        SEED        SEED
#  *********************************************************************************************************************

tf.random.set_seed(SEED)
morphology = TurkishMorphology.create_with_defaults()  # used for root extraction for embeddings

news = ["data/news/", "news_results"]
wikiann = ["data/wikiann/", "wikiann_results"]
fbner = ["data/fbner/", "fbner_results"]
twner = ["data/twner/", "twner_results"]
atisner = ["data/atisner/", "atisner_results"]

paths = [twner, atisner]
models = ["CNN"]  # Add model name to train

for path in paths:
    print('DATASET:\t', path[1])
 
    #  MAKE FOLDER
    if not os.path.isdir(path[1]):
        os.mkdir(path[1])

    #  *****************************************************************************************************************
    #  LOAD DATASET     LOAD DATASET        LOAD DATASET        LOAD DATASET        LOAD DATASET        LOAD DATASET
    #  *****************************************************************************************************************

    data = pd.read_csv(path[0] + "train_SENTENCED.tsv", encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
    tsv_file = open(path[0] + "train_SENTENCED.tsv", encoding="utf-8")
    read_tsv = csv.reader(tsv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

    data_TEST = pd.read_csv(path[0] + "test_SENTENCED.tsv", encoding="utf-8", sep='\t', quoting=csv.QUOTE_NONE)
    tsv_file_TEST = open(path[0] + "test_SENTENCED.tsv", encoding="utf-8")
    read_tsv_TEST = csv.reader(tsv_file_TEST, delimiter='\t', quoting=csv.QUOTE_NONE)

    print('TRAIN_DATA:\n', data.head(10))
    print('TEST_DATA:\n', data_TEST.head(10))

    words = []
    tags = []
    words_TEST = []
    tags_TEST = []

    for line in read_tsv:
        words.append(line[1])
        tags.append(line[2])

    for line in read_tsv_TEST:
        words_TEST.append(line[1])
        tags_TEST.append(line[2])

    words = list(set(words + words_TEST))  # all unique words in the corpus
    tags = list(set(tags))  # all unique tags in the corpus
    words.append("ENDPAD")
    tags.remove("Tag")
    n_tags = len(tags)  # tags number
    n_words = len(words)  # words number
    print("No. of Unique Words in train dataset: " + str(n_words))
    print("No. of Tags in train dataset: " + str(n_tags))
    print("Tags:\t", tags)


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


    getter = SentenceGetter(data)
    sent = getter.get_next()
    sentences = getter.sentences

    getter = SentenceGetter(data_TEST)
    sent = getter.get_next()
    sentences_TEST = getter.sentences

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    #  *****************************************************************************************************************
    #  DIVIDE SENTENCE IN PORTIONS (N_words each sentence)     DIVIDE SENTENCE IN PORTIONS (N_words each sentence)
    #  *****************************************************************************************************************

    sentences_new = []
    for i in range(len(sentences)):
        if len(sentences[i]) > N_words:
            k = round(len(sentences[i]) / N_words)
            for j in range(k + 1):
                sentences_new.append(sentences[i][(j * N_words):((j + 1) * N_words)])
        else:
            sentences_new.append(sentences[i])

    sentences_TEST_new = []
    for i in range(len(sentences_TEST)):
        if len(sentences_TEST[i]) > N_words:
            k = round(len(sentences_TEST[i]) / N_words)
            for j in range(k + 1):
                sentences_TEST_new.append(sentences_TEST[i][(j * N_words):((j + 1) * N_words)])
        else:
            sentences_TEST_new.append(sentences_TEST[i])

    
    #  *****************************************************************************************************************
    #  EMBEDDING        EMBEDDING      EMBEDDING       EMBEDDING        EMBEDDING       EMBEDDING       EMBEDDING
    #  *****************************************************************************************************************
    
    import fasttext
    import fasttext.util

    embed_start = time.time()
    ft = fasttext.load_model('fasttext/cc.tr.300.bin')

    num_tokens = len(words) + 2
    embedding_dim = EMBEDDING_SIZE

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word2idx.items():
        embedding_vector = ft.get_word_vector(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector

    #  *****************************************************************************************************************
    #  TOKENIZE     TOKENIZE        TOKENIZE        TOKENIZE        TOKENIZE        TOKENIZE        TOKENIZE
    #  *****************************************************************************************************************

    X = [[word2idx[w[0]] for w in s] for s in sentences_new]
    X_tr = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=0)

    X = [[word2idx[w[0]] for w in s] for s in sentences_TEST_new]
    X_te = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=0)

    y = [[tag2idx[w[1]] for w in s] for s in sentences_new]
    y_tr = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["O"])
    y = [[tag2idx[w[1]] for w in s] for s in sentences_TEST_new]
    y_te = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["O"])

    shuffler = np.random.permutation(len(y_tr))
    X_tr, y_tr = X_tr[shuffler], y_tr[shuffler]

    y_tr = [to_categorical(i, num_classes=n_tags) for i in y_tr]
    y_te = [to_categorical(i, num_classes=n_tags) for i in y_te]
    X_test = [to_categorical(i, num_classes=n_words) for i in X_te]

    #  *****************************************************************************************************************
    #  TRAIN THE MODEL      TRAIN THE MODEL     TRAIN THE MODEL     TRAIN THE MODEL     TRAIN THE MODEL
    #  *****************************************************************************************************************

    def pred2label(pred):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(idx2tag[p_i].replace("PAD", "O"))
            out.append(out_i)
        return out


    def pred2word(pred, length):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                if p_i == 0:
                    p_i = length
                out_i.append(idx2word[p_i].replace("ENDPAD", "-"))
            out.append(out_i)
        return out


    for mod in models:

        f1 = []
        recall = []
        precision = []
        times = []

        for repeat in range(1, REPEAT + 1):
            print(mod + " REPEAT:\t", repeat)
            
            test_pred = []
            
            if mod == 'BiLSTM':
                # BiLSTM
                model_bilstm = Sequential()
                model_bilstm.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                           trainable=False))
                model_bilstm.add(Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=DROPOUT),
                                               merge_mode="concat"))
                model_bilstm.add(TimeDistributed(Dense(100, activation="relu")))
                model_bilstm.add(Dense(n_tags, activation='sigmoid'))
                # OPTIMIZER
                adam = Adam(learning_rate=LR_RATE)
                model_bilstm.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                print(model_bilstm.summary())
                start = time.time()
                history = model_bilstm.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
                end = time.time()
                take = round(((end - start) / 60), 3)
                times.append(take)

                hour = take // 60
                minute = take - (hour * 60)
                if minute < 10:
                    take = '0' + str(hour) + ':0' + str(minute)
                else:
                    take = '0' + str(hour) + ':' + str(minute)
                print("It takes:\t", take, '\thh:mm')
                # EVALUATE
                test_pred = model_bilstm.predict(X_te, verbose=1)

            elif mod == 'BiGRU':
                model_gru = Sequential()
                model_gru.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                                        trainable=False))
                model_gru.add(Bidirectional(GRU(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=DROPOUT),
                                            merge_mode="concat"))
                model_gru.add(TimeDistributed(Dense(100, activation="relu")))
                model_gru.add(Dense(n_tags, activation='sigmoid'))
                # OPTIMIZER
                adam = Adam(learning_rate=LR_RATE)
                model_gru.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

                start = time.time()
                history = model_gru.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
                end = time.time()
                take = round(((end - start) / 60), 3)
                times.append(take)

                hour = take // 60
                minute = take - (hour * 60)
                if minute < 10:
                    take = '0' + str(hour) + ':0' + str(minute)
                else:
                    take = '0' + str(hour) + ':' + str(minute)
                print("It takes:\t", take, '\thh:mm')
                # EVALUATE
                test_pred = model_gru.predict(X_te, verbose=1)

            elif mod == 'BiLSTM-CRF':
                # BiLSTM
                model_bilstm_crf = Sequential()
                model_bilstm_crf.add(
                    Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                              trainable=False))
                model_bilstm_crf.add(
                    Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=DROPOUT),
                                  merge_mode="concat"))
                model_bilstm_crf.add(TimeDistributed(Dense(100, activation="relu")))
                model_bilstm_crf.add(Dense(n_tags))
                crf = CRF(n_tags)
                model_bilstm_crf.add(crf)
                # OPTIMIZER
                adam = Adam(learning_rate=LR_RATE)
                model_bilstm_crf.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])

                start = time.time()
                history = model_bilstm_crf.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
                end = time.time()
                take = round(((end - start) / 60), 3)
                times.append(take)

                hour = take // 60
                minute = take - (hour * 60)
                if minute < 10:
                    take = '0' + str(hour) + ':0' + str(minute)
                else:
                    take = '0' + str(hour) + ':' + str(minute)
                print("It takes:\t", take, '\thh:mm')
                # EVALUATE
                test_pred = model_bilstm_crf.predict(X_te, verbose=1)

            elif mod == 'BiGRU-CRF':
                model_gru_crf = Sequential()
                model_gru_crf.add(
                    Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                              trainable=False))
                model_gru_crf.add(Bidirectional(GRU(units=LSTM_UNITS, return_sequences=True, recurrent_dropout=DROPOUT),
                                                merge_mode="concat"))
                model_gru_crf.add(TimeDistributed(Dense(100, activation="relu")))
                model_gru_crf.add(Dense(n_tags))
                crf = CRF(n_tags)
                model_gru_crf.add(crf)
                # OPTIMIZER
                adam = Adam(learning_rate=LR_RATE)
                model_gru_crf.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])

                start = time.time()
                history = model_gru_crf.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)
                end = time.time()
                take = round(((end - start) / 60), 3)
                times.append(take)

                hour = take // 60
                minute = take - (hour * 60)
                if minute < 10:
                    take = '0' + str(hour) + ':0' + str(minute)
                else:
                    take = '0' + str(hour) + ':' + str(minute)
                print("It takes:\t", take, '\thh:mm')
                # EVALUATE
                test_pred = model_gru_crf.predict(X_te, verbose=1)

            elif mod == 'CNN':
                model_cnn = Sequential()
                model_cnn.add(Embedding(num_tokens, embedding_dim, input_length=50, embeddings_initializer=Constant(embedding_matrix),
                                           trainable=False))
                model_cnn.add(Conv1D(100, 5, padding="same", activation='relu', input_shape=(50, 300)))
                #model_cnn.add(GlobalMaxPool1D())
                model_cnn.add(TimeDistributed(Dense(100, activation="relu")))
                model_cnn.add(Dense(n_tags, activation='sigmoid'))
                # OPTIMIZER
                adam = Adam(learning_rate=LR_RATE)
                model_cnn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                print(model_cnn.summary())

                history = model_cnn.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1)

                # EVALUATE
                test_pred = model_cnn.predict(X_te, verbose=1)

            else:
                exit("Enter a valid model name!")

            idx2tag = {i: w for w, i in tag2idx.items()}
            idx2word = {i: w for w, i in word2idx.items()}
            liste = [k for k, v in idx2word.items() if v == 'ENDPAD']  # endpad index
            endpad_index = liste[0]

            pred_labels = pred2label(test_pred)

            if not os.path.isdir(path[1] + '/' + mod):
                os.mkdir(path[1] + '/' + mod)

            pred_text = path[1] + '/' + mod + '/pred_repeat_' + str(repeat)
            with open(pred_text, 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                wr.writerows(pred_labels)

            test_labels = pred2label(y_te)
            ground_text = path[1] + '/' + mod + '/ground_repeat_' + str(repeat)
            with open(ground_text, 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                wr.writerows(test_labels)

            if repeat == 1:
                test_words = pred2word(X_test, endpad_index)
                test_set = path[1] + '/' + mod + '/test_words'
                with open(test_set, 'w') as f:
                    wr = csv.writer(f, delimiter=' ')
                    wr.writerows(test_words)

            print(classification_report(test_labels, pred_labels, digits=4))

            f1.append(f1_score(test_labels, pred_labels, average='weighted'))
            recall.append(recall_score(test_labels, pred_labels, average='weighted'))
            precision.append(precision_score(test_labels, pred_labels, average='weighted'))

            class_report = path[1] + '/' + mod + '/entities_repeat_' + str(repeat)
            with open(class_report, "w") as text_file:
                text_file.write(classification_report(test_labels, pred_labels))

        string = "HYPERPARAMETERS\t**********" + \
                 "\nEPOCH NUMBER:  \t" + str(EPOCH) + \
                 "\nBATCH SIZE:    \t" + str(BATCH_SIZE) + \
                 "\nEMBEDDING SIZE:\t" + str(EMBEDDING_SIZE) + \
                 "\nMAX LENGTH:    \t" + str(MAX_LEN) + \
                 "\nLR RATE:       \t" + str(LR_RATE)

        with open(path[1] + '/' + mod + '/hyperparameters.txt', "w") as text_file:
            text_file.write(string)

        f1_end = str(sum(f1) / len(f1))
        rec_end = str(sum(recall) / len(recall))
        pre_end = str(sum(precision) / len(precision))

        result_txt = path[1] + '/' + mod + '/FINAL_RESULTS_' + mod
        # write all results to a .txt file
        with open(result_txt, 'w') as f:
            f.write("Repeat#\tPrecision\tRecall\tF1_Score\n")
            for i in range(len(f1)):
                text = str(i) + "\t" + str(precision[i]) + "\t" + str(recall[i]) + "\t" + str(f1[i]) + "\n"
                f.write(text)
            text = "\nAvg.\t" + str(sum(precision) / len(precision)) + "\t" + \
                   str(sum(recall) / len(recall)) + "\t" + str(sum(f1) / len(f1))
            f.write(text)
