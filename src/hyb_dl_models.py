# Author: Oguzhan Ozcelik
# Date: 02.08.2022
# Subject: Training and testing of BiLSTM-CRF and BiGRU-CRF models.

import json
import pickle
import numpy as np
import fasttext.util
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from src.utils import DatasetLoader
from tensorflow.keras.layers import LSTM, GRU, Embedding, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow_addons.text import crf_log_likelihood, crf_decode
from tensorflow.keras.models import Sequential
from seqeval.metrics import classification_report


class CRF_HEAD(L.Layer):
    # Source: https://github.com/tensorflow/addons/issues/1769#issuecomment-631920137
    # Since CRF layer has been removed from tensorflow_addons, the current solution is using
    # this reimplementation from the given github link.

    def __init__(self,
                 output_dim,
                 sparse_target=True,
                 **kwargs):
        """
        Args:
            output_dim (int): the number of labels to tag each temporal input.
            sparse_target (bool): whether the the ground-truth label represented in one-hot.
        Input shape:
            (batch_size, sentence length, output_dim)
        Output shape:
            (batch_size, sentence length, output_dim)
        """

        super(CRF_HEAD, self).__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.sparse_target = sparse_target
        self.sequence_lengths = None
        self.input_spec = L.InputSpec(min_ndim=3)
        self.supports_masking = False
        self.transitions = None

    def build(self, input_shape):

        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = L.InputSpec(min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError('The last dimension of the inputs to `CRF` '
                             'should be defined. Found `None`.')
        if f_shape[-1] != self.output_dim:
            print("F SHAPE:       ", f_shape[-1])
            print("OUTPUT DIM:    ", self.output_dim)
            raise ValueError('The last dimension of the input shape must be equal to output'
                             ' shape. Use a linear layer if needed.')
        self.input_spec = input_spec
        self.transitions = self.add_weight(name='transitions',
                                           shape=[self.output_dim, self.output_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def call(self, inputs, sequence_lengths=None, training=None, **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == 'int32'
            seq_len_shape = tf.convert_to_tensor(sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            self.sequence_lengths = K.flatten(sequence_lengths)
        else:
            self.sequence_lengths = tf.ones(tf.shape(inputs)[0], dtype=tf.int32) * (
                tf.shape(inputs)[1]
            )

        viterbi_sequence, _ = crf_decode(sequences,
                                         self.transitions,
                                         self.sequence_lengths)
        output = K.one_hot(viterbi_sequence, self.output_dim)
        return K.in_train_phase(sequences, output)

    @property
    def loss(self):
        def crf_loss(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
            log_likelihood, self.transitions = crf_log_likelihood(
                y_pred,
                tf.cast(K.argmax(y_true), dtype=tf.int32) if self.sparse_target else y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
            return tf.reduce_mean(-log_likelihood)

        return crf_loss

    @property
    def accuracy(self):
        def viterbi_accuracy(y_true, y_pred):
            # -1e10 to avoid zero at sum(mask)
            mask = K.cast(
                K.all(K.greater(y_pred, -1e10), axis=2), K.floatx())
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            y_pred, _ = crf_decode(y_pred, self.transitions, sequence_lengths)
            if self.sparse_target:
                y_true = K.argmax(y_true, 2)
            y_pred = K.cast(y_pred, 'int32')
            y_true = K.cast(y_true, 'int32')
            corrects = K.cast(K.equal(y_true, y_pred), K.floatx())
            return K.sum(corrects * mask) / K.sum(mask)

        return viterbi_accuracy

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'sparse_target': self.sparse_target,
            'supports_masking': self.supports_masking,
            'transitions': K.eval(self.transitions)
        }
        base_config = super(CRF_HEAD, self).get_config()
        return dict(base_config, **config)


class LSTM_GRU_CRF:
    def __init__(self, data_path, model_path, model_name):
        if model_name not in ['bilstm_crf', 'bigru_crf']:
            raise ValueError("Invalid feature type. Expected one of: %s" % ['bilstm_crf', 'bigru_crf'])
        with open('src/configs/dl_models_config.json') as f:
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
        data = DatasetLoader(train_path=self.data_path + 'train_sentenced.tsv',
                             test_path=None)
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

        if self.model_name == 'bilstm_crf':
            model_bilstm_crf = Sequential()
            model_bilstm_crf.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size,
                                           embeddings_initializer=Constant(embedding_matrix),
                                           trainable=False))
            model_bilstm_crf.add(
                Bidirectional(LSTM(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                              merge_mode="concat"))
            model_bilstm_crf.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model_bilstm_crf.add(Dense(len(tag2idx)))
            crf = CRF_HEAD(len(tag2idx))
            model_bilstm_crf.add(crf)
            adam = Adam(learning_rate=self.lr_rate)
            model_bilstm_crf.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])
            model_bilstm_crf.fit(X_tr, np.array(y_tr), batch_size=self.train_batch, epochs=self.epoch, verbose=1)
            model_bilstm_crf.save(self.model_path + self.model_name + '.h5')

        elif self.model_name == 'bigru_crf':
            model_bigru_crf = Sequential()
            model_bigru_crf.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size,
                                          embeddings_initializer=Constant(embedding_matrix),
                                          trainable=False))
            model_bigru_crf.add(
                Bidirectional(GRU(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                              merge_mode="concat"))
            model_bigru_crf.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model_bigru_crf.add(Dense(len(tag2idx)))
            crf = CRF_HEAD(len(tag2idx))
            model_bigru_crf.add(crf)
            adam = Adam(learning_rate=self.lr_rate)
            model_bigru_crf.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])

            model_bigru_crf.fit(X_tr, np.array(y_tr), batch_size=self.train_batch, epochs=self.epoch, verbose=1)
            model_bigru_crf.save(self.model_path + self.model_name + '.h5')

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

        model = Sequential()

        if self.model_name == 'bilstm_crf':

            embedding_matrix = np.zeros((len(word2idx), self.embed_dim))
            for word, i in word2idx.items():
                if word in ['<UNK>', '<PAD>']:
                    embedding_vector = np.zeros(self.embed_dim)
                else:
                    embedding_vector = self.ft.get_word_vector(word)

                embedding_matrix[i] = embedding_vector

            model.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size,
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=False))
            model.add(
                Bidirectional(LSTM(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                              merge_mode="concat"))
            model.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model.add(Dense(len(tag2idx)))
            crf = CRF_HEAD(len(tag2idx))
            model.add(crf)
            adam = Adam(learning_rate=self.lr_rate)
            model.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])
            model.load_weights(self.model_path + 'bilstm_crf.h5')

        elif self.model_name == 'bigru_crf':

            embedding_matrix = np.zeros((len(word2idx), self.embed_dim))
            for word, i in word2idx.items():
                if word in ['<UNK>', '<PAD>']:
                    embedding_vector = np.zeros(self.embed_dim)
                else:
                    embedding_vector = self.ft.get_word_vector(word)

                embedding_matrix[i] = embedding_vector

            model.add(Embedding(len(word2idx), self.embed_dim, input_length=self.unit_size,
                                embeddings_initializer=Constant(embedding_matrix),
                                trainable=False))
            model.add(
                Bidirectional(GRU(units=self.unit_size, return_sequences=True, recurrent_dropout=self.dropout),
                              merge_mode="concat"))
            model.add(TimeDistributed(Dense(self.dense, activation="relu")))
            model.add(Dense(len(tag2idx)))
            crf = CRF_HEAD(len(tag2idx))
            model.add(crf)
            adam = Adam(learning_rate=self.lr_rate)
            model.compile(optimizer=adam, loss=crf.loss, metrics=[crf.accuracy])
            print(model.summary())

            model.load_weights(self.model_path + 'bigru_crf.h5')

        test_pred = model.predict(X_te, verbose=1, batch_size=self.test_batch)

        pred_labels = self.pred2label(test_pred, idx2tag=idx2tag)
        test_labels = self.pred2label(y_te, idx2tag=idx2tag)

        report = classification_report(test_labels, pred_labels, digits=4)

        with open(result_path + 'final_result', 'w') as f:
            f.write(report)

        print(report)
