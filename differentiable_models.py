# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of the classes of differentiable IR models with their
# network architecture
# =============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

####Differentiable models"
"""This file contains the definitions of differantiable models introduced in chapter 7 for tf,tf-idf,jm,dir,bm25 that can be trained on a particular
collection. It contains the architecture of the rededifined differentiable model with the one linear layer for the prediction of TDV""" #HR
class diff_simple_TF(Model):
    def __init__(self, embedding_matrix, dropout_rate=0.1):
        super(diff_simple_TF, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   self.embedding_dim,
                                                   weights=[embedding_matrix],
                                                   mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.dropout_rate = dropout_rate

    def make_BoW(self, seq, index, sparse_index):
        #computing a mask tensor where the value zero is considered as a padding value that should be masked out #HR
        # Pour fabriquer un tenseur dense à partir de structure creuses comme la liste
        # des termes des requêtes présentes
        mask = tf.dtypes.cast(self.embedding.compute_mask(index), dtype=tf.float32)
        # Multiplying the mask with a tensor of seq where the dimensions of size 1 are removed #HR THe result is a tensor
        # of only values
        seq = tf.math.multiply(mask, tf.squeeze(seq))
        # Transform seq tensor into a single vector => Why ??
        seq = tf.reshape(seq, [-1])

        #Creating a sparse tensor and reordering its elements
        seq = tf.SparseTensor(indices=sparse_index, values=seq, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        #??? HR
        linearized = tf.matmul(seq.indices, tf.constant([[index.shape[0]], [1]], dtype=tf.int64))
        y, idx = tf.unique(tf.squeeze(linearized))
        values = tf.math.segment_sum(seq.values, idx)
        y = tf.expand_dims(y, 1)
        indices = tf.concat([y // index.shape[0], y % index.shape[0]], axis=1)
        seq = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        return tf.sparse.to_dense(seq)

    def call(self, q_index_float_32, q_index, q_sparse_index, d_index, d_sparse_index):
        """
        This calculates the relevance between queries and documents #HR
        Warning : seems to overload call operator #JPC
        """
        q = self.make_BoW(q_index_float_32, q_index, q_sparse_index)

        d = tf.nn.dropout(self.embedding(d_index), rate=self.dropout_rate)
        d = tf.nn.dropout(self.linear(d), rate=self.dropout_rate)
        d = self.make_BoW(d, d_index, d_sparse_index)

        rel = tf.math.reduce_sum(tf.math.multiply(q, d), axis=0)

        return rel, d

    def compute_index(self):
        """This computes the TDV weights for the terms in the vocabulary""" #HR
        index = [_ for _ in range(self.vocab_size)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))


class diff_TF_IDF(Model):
    def __init__(self, embedding_matrix, dropout_rate=0.1):
        super(diff_TF_IDF, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   self.embedding_dim,
                                                   weights=[embedding_matrix],
                                                   mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.dropout_rate = dropout_rate

    def make_BoW(self, seq, index, sparse_index):
        mask = tf.dtypes.cast(self.embedding.compute_mask(index), dtype=tf.float32)
        seq = tf.math.multiply(mask, tf.squeeze(seq))
        seq = tf.reshape(seq, [-1])

        seq = tf.SparseTensor(indices=sparse_index, values=seq, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        linearized = tf.matmul(seq.indices, tf.constant([[index.shape[0]], [1]], dtype=tf.int64))
        y, idx = tf.unique(tf.squeeze(linearized))
        values = tf.math.segment_sum(seq.values, idx)
        y = tf.expand_dims(y, 1)
        indices = tf.concat([y // index.shape[0], y % index.shape[0]], axis=1)
        seq = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        return tf.sparse.to_dense(seq)

    def call(self, q_index_float_32, q_index, q_sparse_index, d_index, d_sparse_index):
        q = self.make_BoW(q_index_float_32, q_index, q_sparse_index)

        d = tf.nn.dropout(self.embedding(d_index), rate=self.dropout_rate)
        d = tf.nn.dropout(self.linear(d), rate=self.dropout_rate)
        d = self.make_BoW(d, d_index, d_sparse_index)

        maxdf = tf.keras.backend.max(tf.math.reduce_sum(d, axis=1))

        idf = tf.math.log((maxdf + 1) / (1 + tf.math.reduce_sum(d, axis=1)))

        idf_d = tf.multiply(d, tf.reshape(idf, (-1, 1)))

        rel = tf.math.reduce_sum(tf.math.multiply(q, idf_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(self.vocab_size)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))


class diff_DIR(Model):
    def __init__(self, embedding_matrix, mu=2500.0, dropout_rate=0.1):
        super(diff_DIR, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   self.embedding_dim,
                                                   weights=[embedding_matrix],
                                                   mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.mu = tf.Variable(mu)
        self.dropout_rate = dropout_rate

    def make_BoW(self, seq, index, sparse_index):
        mask = tf.dtypes.cast(self.embedding.compute_mask(index), dtype=tf.float32)
        seq = tf.math.multiply(mask, tf.squeeze(seq))
        seq = tf.reshape(seq, [-1])

        seq = tf.SparseTensor(indices=sparse_index, values=seq, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        linearized = tf.matmul(seq.indices, tf.constant([[index.shape[0]], [1]], dtype=tf.int64))
        y, idx = tf.unique(tf.squeeze(linearized))
        values = tf.math.segment_sum(seq.values, idx)
        y = tf.expand_dims(y, 1)
        indices = tf.concat([y // index.shape[0], y % index.shape[0]], axis=1)
        seq = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        return tf.sparse.to_dense(seq)

    def call(self, q_index_float_32, q_index, q_sparse_index, d_index, d_sparse_index):
        q = self.make_BoW(q_index_float_32, q_index, q_sparse_index)

        d = tf.nn.dropout(self.embedding(d_index), rate=self.dropout_rate)
        d = tf.nn.dropout(self.linear(d), rate=self.dropout_rate)
        d = self.make_BoW(d, d_index, d_sparse_index)

        cfreq = tf.math.reduce_sum(d, axis=1) / tf.math.reduce_sum(d)

        smoothing = tf.math.log(self.mu / (tf.math.reduce_sum(d, axis=0) + self.mu))

        dir_d = tf.math.log(1 + d / (1 + self.mu * tf.reshape(cfreq, (-1, 1)))) + smoothing

        rel = tf.math.reduce_sum(tf.math.multiply(q, dir_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(self.vocab_size)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))


class diff_BM25(Model):
    def __init__(self, embedding_matrix, k1=1.2, b=0.75, dropout_rate=0.1):
        super(diff_BM25, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   self.embedding_dim,
                                                   weights=[embedding_matrix],
                                                   mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.k1 = tf.Variable(k1)
        self.b = tf.Variable(b)
        self.dropout_rate = dropout_rate

    def make_BoW(self, seq, index, sparse_index):
        mask = tf.dtypes.cast(self.embedding.compute_mask(index), dtype=tf.float32)
        seq = tf.math.multiply(mask, tf.squeeze(seq))
        seq = tf.reshape(seq, [-1])

        seq = tf.SparseTensor(indices=sparse_index, values=seq, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        linearized = tf.matmul(seq.indices, tf.constant([[index.shape[0]], [1]], dtype=tf.int64))
        y, idx = tf.unique(tf.squeeze(linearized))
        values = tf.math.segment_sum(seq.values, idx)
        y = tf.expand_dims(y, 1)
        indices = tf.concat([y // index.shape[0], y % index.shape[0]], axis=1)
        seq = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        return tf.sparse.to_dense(seq)

    def call(self, q_index_float_32, q_index, q_sparse_index, d_index, d_sparse_index):
        q = self.make_BoW(q_index_float_32, q_index, q_sparse_index)

        d = tf.nn.dropout(self.embedding(d_index), rate=self.dropout_rate)
        d = tf.nn.dropout(self.linear(d), rate=self.dropout_rate)
        d = self.make_BoW(d, d_index, d_sparse_index)

        maxdf = tf.keras.backend.max(tf.math.reduce_sum(d, axis=1))

        idf = tf.math.log((maxdf + 1) / (1 + tf.math.reduce_sum(d, axis=1)))

        d_length = tf.math.reduce_sum(d, axis=0)

        avg_d_length = tf.reduce_mean(d_length)

        bm25_d = tf.reshape(idf, (-1, 1)) * ((self.k1 + 1) * d) / (
                    d + self.k1 * ((1 - self.b) + self.b * d_length / avg_d_length))

        rel = tf.math.reduce_sum(tf.math.multiply(q, bm25_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(self.vocab_size)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))

# The class for Jelnik-Mercer model was absent in the original file. HR created it
class diff_JM(Model):
    def __init__(self, embedding_matrix, lamb=0.15, dropout_rate=0.1):
        super(diff_JM, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   self.embedding_dim,
                                                   weights=[embedding_matrix],
                                                   mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.lamb= tf.Variable(lamb)
        self.dropout_rate = dropout_rate

    def make_BoW(self, seq, index, sparse_index):
        mask = tf.dtypes.cast(self.embedding.compute_mask(index), dtype=tf.float32)
        seq = tf.math.multiply(mask, tf.squeeze(seq))
        seq = tf.reshape(seq, [-1])

        seq = tf.SparseTensor(indices=sparse_index, values=seq, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        linearized = tf.matmul(seq.indices, tf.constant([[index.shape[0]], [1]], dtype=tf.int64))
        y, idx = tf.unique(tf.squeeze(linearized))
        values = tf.math.segment_sum(seq.values, idx)
        y = tf.expand_dims(y, 1)
        indices = tf.concat([y // index.shape[0], y % index.shape[0]], axis=1)
        seq = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.vocab_size, index.shape[0]])
        seq = tf.sparse.reorder(seq)

        return tf.sparse.to_dense(seq)

    def call(self, q_index_float_32, q_index, q_sparse_index, d_index, d_sparse_index):
        q = self.make_BoW(q_index_float_32, q_index, q_sparse_index)

        d = tf.nn.dropout(self.embedding(d_index), rate=self.dropout_rate)
        d = tf.nn.dropout(self.linear(d), rate=self.dropout_rate)
        d = self.make_BoW(d, d_index, d_sparse_index)

        cfreq = tf.math.reduce_sum(d, axis=1) / tf.math.reduce_sum(d)

        d_length = tf.math.reduce_sum(d, axis=0)


        jm_d = tf.math.log(1+(self.lamb/(1-self.lamb))*(1/d_length)*(d/tf.reshape(cfreq,(-1,1))))

        rel = tf.math.reduce_sum(tf.math.multiply(q, jm_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(self.vocab_size)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))
