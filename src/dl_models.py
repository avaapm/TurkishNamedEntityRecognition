# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Training and testing of BiLSTM, BiGRU, and CNN models.

import json
import pickle
import numpy as np
import fasttext.util
import tensorflow as tf

from src.utils import DatasetLoader
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Embedding, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Sequential
from seqeval.metrics import classification_report


class DL_MODELS:
    def __init__(self, data_path, model_path, model_name):
        if model_name not in ['bilstm', 'bigru', 'cnn']:
            raise ValueError("Invalid feature type. Expected one of: %s" % ['bilstm', 'bigru', 'cnn'])
        with open('/src/configs/dl_models_config.json') as f:
            config = json.load(f)

        self.data_path = data_path  # input data path either train or test
        self.model_path = model_path  # model will be saved to this path or loaded for test
        self.ft = fasttext.load_model(config["fasttext_path"])
        self.epoch = config["num_train_epochs"]
        self.lr_rate = config["learning_rate"]
        self.unit_size = config["unit_size"]
        self.train_batch = config["train_batch_size"]
        self.test_batch = config["test_batch_size"]
        self.dropout = config["dropout"]
        self.embed_dim = config["embedding_dim"]
        self.dense = config["dense_neuron_size"]
        self.filters = config["cnn_filter_size"]
        self.kernel = config["cnn_kernel_size"]
        self.model_name = model_name

    def pred2label(self, pred, idx2tag):
        out = []
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(idx2tag[p_i].replace("PAD", "O"))
            out.append(out_i)
        return out

    def train(self):
        data = DatasetLoader(train_path=self.data_path + 'train_sentenced.tsv', test_path=None)
        X_data, y_tr, tag2idx, word2idx = data.bilstm_loader(train=True, word2ix_test={}, tag2ix_test={})

        embedding_matrix = np.zeros((len(word2idx), self.embed_dim))
        for word, i in word2idx.items():
            if word in ['<UNK>', '<PAD>']:
                embedding_vector = np.zeros(self.embed_dim)
            else:
                embedding_vector = self.ft.get_word_vector(word)

            embedding_matrix[i] = embedding_vector

        X = [[word2idx[w] for w in s] for s in X_data]
        X_tr = pad_sequences(maxlen=self.unit_size, sequences=X, padding="post", value=word2idx['<PAD>'])

        shuffler = np.random.permutation(len(y_tr))
        X_tr, y_tr = X_tr[shuffler], y_tr[shuffler]

        y_tr = [to_categorical(i, num_classes=len(tag2idx)) for i in y_tr]

        if self.model_name == 'bilstm':
            model_bilstm = Sequential()
            model_bilstm.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size, embeddings_initializer=Constant(embedding_matrix),
                                       trainable=False))
            model_bilstm.add(Bidirectional(LSTM(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                                           merge_mode="concat"))
            model_bilstm.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model_bilstm.add(Dense(len(tag2idx), activation='sigmoid'))

            adam = Adam(learning_rate=self.lr_rate)
            model_bilstm.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model_bilstm.fit(X_tr, np.array(y_tr), batch_size=self.train_batch, epochs=self.epoch, verbose=1)
            model_bilstm.save(self.model_path)

        elif self.model_name == 'bigru':
            model_bigru = Sequential()
            model_bigru.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size, embeddings_initializer=Constant(embedding_matrix),
                                       trainable=False))
            model_bigru.add(
                Bidirectional(GRU(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                              merge_mode="concat"))
            model_bigru.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model_bigru.add(Dense(len(tag2idx), activation='sigmoid'))

            adam = Adam(learning_rate=self.lr_rate)
            model_bigru.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model_bigru.fit(X_tr, np.array(y_tr), batch_size=self.train_batch, epochs=self.epoch, verbose=1)
            model_bigru.save(self.model_path)

        elif self.model_name == 'cnn':
            model_cnn = Sequential()
            model_cnn.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size, embeddings_initializer=Constant(embedding_matrix),
                                       trainable=False))
            model_cnn.add(Conv1D(self.filters, self.kernel, padding="same", activation='relu', input_shape=(self.unit_size, self.embed_dim)))
            model_cnn.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model_cnn.add(Dense(len(tag2idx), activation='sigmoid'))

            adam = Adam(learning_rate=self.lr_rate)
            model_cnn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            model_cnn.fit(X_tr, np.array(y_tr), batch_size=self.train_batch, epochs=self.epoch, verbose=1)
            model_cnn.save(self.model_path)

        with open(self.model_path + 'word2idx.pkl', 'wb') as f:
            pickle.dump(word2idx, f)
        with open(self.model_path + 'tag2idx.pkl', 'wb') as f:
            pickle.dump(tag2idx, f)

    def evaluate(self, result_path):
        data = DatasetLoader(train_path=None, test_path=self.data_path + 'test_sentenced.tsv')
        with open(self.model_path + 'word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
        with open(self.model_path + 'tag2idx.pkl', 'rb') as f:
            tag2idx = pickle.load(f)

        X_data, y_te = data.bilstm_loader(train=False, word2ix_test=word2idx, tag2ix_test=tag2idx)
        y_te = [to_categorical(i, num_classes=len(tag2idx)) for i in y_te]

        X = [[word2idx[w] for w in s] for s in X_data]
        X_te = pad_sequences(maxlen=self.unit_size, sequences=X, padding="post", value=word2idx['<PAD>'])

        idx2tag = {i: w for w, i in tag2idx.items()}
        model_bilstm = tf.keras.models.load_model(self.model_path)

        test_pred = model_bilstm.predict(X_te, verbose=1, batch_size=self.test_batch)

        pred_labels = self.pred2label(test_pred, idx2tag=idx2tag)
        test_labels = self.pred2label(y_te, idx2tag=idx2tag)

        report = classification_report(test_labels, pred_labels, digits=4)

        with open(result_path + 'final_result', 'w') as f:
            f.write(report)

        print(report)
