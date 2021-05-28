# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to evaluate baseline IR models without training , to do training
# to get the TDV weights for a certain differentiable model and to evaluate baseline
# IR models after taking into account the TDVs for a TREC Collection
# =============================================================================

import os
import pickle
import random
import string
import fasttext
import numpy as np
import pytrec_eval
import pandas as pd
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


import re
import pickle
import argparse
import importlib
import pytrec_eval


import differentiable_models
import baseline_models_and_tdv_implementation
from Trec_Collection import TrecCollection
import utils
###############Training on Trec collections############### #HR



#### Training and getting results for Trec collection#### #HR

def main():
    #enabling eager execution of tensorflow. It is enabled in version 2 but not in version1 #HR
    tf.enable_eager_execution()
    #parsing arguments #HR
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-w', '--weights_path', nargs="?", type=str)
    parser.add_argument('-f', '--folds', nargs="?", type=int, default=5)
    parser.add_argument('-e', '--nb_epoch', nargs="?", type=int)
    parser.add_argument('-l', '--l1_weight', nargs="?", type=float)
    parser.add_argument('-d', '--dropout_rate', nargs="?", type=float, default=0.0)
    parser.add_argument('--lr', nargs="?", type=float)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)
    parser.add_argument('--IR_model', nargs="?", type=str, default='tf')
    parser.add_argument('-u', '--update_embeddings', action="store_true")

    args = parser.parse_args()

    print(args, flush=True)

    # Loading indexed collection #HR
    Collection = TrecCollection()
    with open(args.indexed_path, 'rb') as f:
        Collection = pickle.load(f)

    Collection.doc_index[-1] = "-1"
    Collection.doc_index["-1"] = -1
    # Loading relevance judgements from collection #HR
    with open(args.coll_path + 'qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # ???? #HR
    id_titl = Collection.vocabulary['titl']

    for i in range(len(Collection.all_indexed_queries)):

        if Collection.all_indexed_queries[i][0] == id_titl and len(Collection.all_indexed_queries[i]) > 1:
            print("found it at ", i," ", Collection.all_indexed_queries[i][0])
            del Collection.all_indexed_queries[i][0]

    for i in range(len(Collection.indexed_queries)):
        for j in range(len(Collection.indexed_queries[i])):
            if Collection.indexed_queries[i][j][0] == id_titl and len(Collection.indexed_queries[i][j]) > 1:
                del Collection.indexed_queries[i][j][0]

    print('---------------------start-------------------',flush=True)
    # Getting collection vocabulary size and total number of elements in collection #HR
    coll_vocab_size, coll_tot_nb_elem = utils.evaluate_inverted_index(Collection.inverted_index)
    # Creating for each fold and for a certain experiment a directory for results,weights and plots data #HR


    for fold in range(args.folds):

        plot_values = dict()

        for model_name in ['tf',
                           'tf_idf',
                           'DIR',
                           'BM25','JM']:
            plot_values[model_name] = [[], []]

        if not os.path.exists(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name)

        if not os.path.exists(args.weights_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.weights_path + '/fold' + str(fold) + '/' + args.experiment_name)

        if not os.path.exists(args.plot_path + '/fold' + str(fold) + '/'):
            os.makedirs(args.plot_path + '/fold' + str(fold) + '/')
        # Computing metrics for baseline models for a certain fold and updating plot_values dictionnary #HR
                #HR modified the eval_baseline_index to a function eval_baseline_index_trec
        # The previous version did not work because of different call parameters
        utils.eval_baseline_index_trec(args.coll_path,
                            Collection,
                            fold,
                            qrel,
                            plot_values,
                            args.results_path,
                            args.experiment_name,
                            0)
        # Saving plot_values dict of a particular fold as a pickle #HR
        pickle.dump(plot_values, open(args.plot_path + '/fold' + str(fold) + '/' + args.experiment_name, 'wb'))
        # Initialization of batch size, the loss function,te optimizer and the model to train #HR
        batch_gen_time = []
        batch_size = 32
        y_true = tf.ones(batch_size, )
        loss_function = tf.keras.losses.Hinge()
        optimizer = tf.keras.optimizers.Adam(args.lr)

        if args.IR_model == 'tf':
            model = differentiable_models.diff_simple_TF(Collection.embedding_matrix, dropout_rate=args.dropout_rate)

        elif args.IR_model == 'tf_idf':
            model = differentiable_models.diff_TF_IDF(Collection.embedding_matrix, dropout_rate=args.dropout_rate)

        elif args.IR_model == 'DIR':
            model = differentiable_models.diff_DIR(Collection.embedding_matrix, dropout_rate=args.dropout_rate)

        elif args.IR_model == 'BM25':
            model = differentiable_models.diff_BM25(Collection.embedding_matrix, dropout_rate=args.dropout_rate)
        #HR added JM model
        elif args.IR_model == 'JM':
            model = differentiable_models.diff_JM(Collection.embedding_matrix, dropout_rate=args.dropout_rate)

        # Training the model #HR
        print("Start training for fold ", fold, " ", args.experiment_name, flush=True)
        epoch = 0
        prop_elem_index = 1.0
        while epoch < args.nb_epoch and prop_elem_index > 0.05:

            begin = time.time()
            # generation of batches from the trec collection for training #HR
            query_batches, positive_doc_batches, negative_doc_batches = Collection.generate_training_batches(
                fold, batch_size)

            rank_loss = 0.0
            reg_loss = 0.0
            all_non_zero = 0.0

            begin = time.time()

            for i in range(len(query_batches)):
                with tf.GradientTape() as tape:
                    # reshaping queries, pos_documents and neg_documents into a numpy ndarray #HR
                    # Selectionne les requêtes dans le batch
                    # query_batches est l'indice des requetes qui sont selectionnées pour ce batch
                    # queries : une liste des requêtes dans le batch, chaque query est une liste d'idf de token
                    # La pad c'est pour que toutes les queries soient de même taille
                    queries = tf.keras.preprocessing.sequence.pad_sequences(
                        [Collection.all_indexed_queries[j] for j in query_batches[i]], padding='post')

                    pos_documents = tf.keras.preprocessing.sequence.pad_sequences(
                        [Collection.indexed_docs[j] for j in positive_doc_batches[i]], padding='post')

                    neg_documents = tf.keras.preprocessing.sequence.pad_sequences(
                        [Collection.indexed_docs[j] for j in negative_doc_batches[i]], padding='post')
                    # Creating sparse querie, pos_document and neg_documents indexes #HR
                    # Crée des tenseurs creux à partir de la liste des requêtes sélectionnées
                    # Est-ce que column est un embedding ou juste la liste des idf des termes ?
                    # NON : c'est pas encore de l'embedding mais juste des odf de vocabulaire
                    # j est l'indice de la query, raw est une query, donc column est simplement un idf de vocabulaire
                    q_sparse_index = [[column, j] for j, raw in enumerate(queries) for column in raw]
                    pos_d_sparse_index = [[column, j] for j, raw in enumerate(pos_documents) for column in raw]
                    neg_d_sparse_index = [[column, j] for j, raw in enumerate(neg_documents) for column in raw]
                    # computing relevance and dense document for the negative and positive documents in the batch #HR
                    # Pourquoi cliper les requêtes ? Si c'est des IDF alors cela donne le motif binaire des termes
                    # Ca doit faire 11100000 s'il y a 3 termes
                    # ATTENTION : il faut que l'identifiant du vocabulaire soit différent de 0 , donc commence à 1 
                    pos_res, pos_d = model(np.clip(queries, 0, 1).astype(np.float32),
                                           queries,
                                           q_sparse_index,
                                           pos_documents,
                                           pos_d_sparse_index)

                    neg_res, neg_d = model(np.clip(queries, 0, 1).astype(np.float32),
                                           queries,
                                           q_sparse_index,
                                           neg_documents,
                                           neg_d_sparse_index)
                    # Computing the hinge loss and the regularization loss and total loss #HR
                    ranking_loss = loss_function(y_true=y_true, y_pred=pos_res - neg_res)

                    regularization_loss = tf.norm(pos_d + neg_d, ord=1)

                    rank_loss += ranking_loss.numpy()
                    reg_loss += regularization_loss.numpy()

                    all_non_zero += tf.math.count_nonzero(pos_d + neg_d).numpy()

                    loss = (1.0 - args.l1_weight) * ranking_loss + args.l1_weight * regularization_loss
                    # Calculating gradients #HR
                    if args.update_embeddings:
                        gradients = tape.gradient(loss, model.trainable_variables)
                    else:
                        gradients = tape.gradient(loss, model.trainable_variables[1:])
                # Back propagating the gradients #HR
                if args.update_embeddings:
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                else:
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables[1:]))

            # Compute the TDVs after the training and saving them #HR
            weights = model.compute_index()

            pickle.dump(weights, open(
                args.weights_path + '/fold' + str(fold) + '/' +  args.experiment_name + '/epoch_' + str(epoch), 'wb'))

            inverted_index, redefined_idf, redefined_docs_length, redefined_c_freq = utils.utils_compute_info_retrieval(
                Collection,
                weights,
                weighted=True)
            #JF
            #             inverted_index,idf,docs_length,c_freq = utils.compute_info_retrieval(Collection,
            #                                                                                  weights,
            #                                                                                  weighted=False)


            # Computing new vocab_size and total number of elements after introducting the TDV #HR
            vocab_size, tot_nb_elem = utils.evaluate_inverted_index(inverted_index)

            print(str(100 * vocab_size / coll_vocab_size)[0:5] + '% of the vocabulary is kept')
            print(str(100 * tot_nb_elem / coll_tot_nb_elem)[0:5] + '% of the index is kept', flush=True)

            prop_elem_index = tot_nb_elem / coll_tot_nb_elem
            #Evaluating baseline models with their new inverted index and new idf, doc length and collection frequencies #HR
                #HR modified the eval_learned_index to a function eval_learned_index_trec
    # The previous version did not work because of a different call parameters
            utils.eval_learned_index_trec(args.coll_path,
                                     Collection,
                                     args.IR_model,
                                     model,
                                     qrel,
                                     plot_values,
                                     args.plot_path,
                                     fold,
                                     inverted_index,
                                     weights,
                                     redefined_idf,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     #                                        idf,
                                     #                                        docs_length,
                                     #                                        c_freq,
                                     prop_elem_index,
                                     args.results_path,
                                     args.experiment_name,
                                     epoch + 1)
            epoch += 1
        print("finish training for fold ", fold, " ", args.experiment_name, flush=True) #HR

    print("-----------------Finished-------------------",flush=True) #HR

if __name__ == "__main__":
    main()
