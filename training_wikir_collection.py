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

#Packages
import os
import time
import pickle
import argparse
import pytrec_eval
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
#Coded files
import utils
import std_tokenizer
import wikIR_Collection
import differentiable_models

################## Training and getting results from WikIR collection ############

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-w', '--weights_path', nargs="?", type=str)
    parser.add_argument('-e', '--nb_epoch', nargs="?", type=int)
    parser.add_argument('-l', '--l1_weight', nargs="?", type=float)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)
    parser.add_argument('-u', '--update_embeddings', action="store_true")
    #HR added the choice of a particular differentiable model. There was no choices
    #in the original file
    parser.add_argument('--IR_model', nargs="?", type=str, default='tf')
    #HR added the option to choose a lerning rate and dropout rate
    parser.add_argument('--lr', nargs="?", type=float)
    parser.add_argument('-d', '--dropout_rate', nargs="?", type=float, default=0.0)
    args = parser.parse_args()

    print(args, flush=True)

    if not os.path.exists(args.results_path + '/validation/' + args.experiment_name):
        os.makedirs(args.results_path + '/validation/' + args.experiment_name)

    if not os.path.exists(args.results_path + '/test/' + args.experiment_name):
        os.makedirs(args.results_path + '/test/' + args.experiment_name)

    if not os.path.exists(args.weights_path + '/' + args.experiment_name):
        os.makedirs(args.weights_path + '/' + args.experiment_name)

    if not os.path.exists(args.plot_path + '/validation/'):
        os.makedirs(args.plot_path + '/validation/')

    if not os.path.exists(args.plot_path + '/test/'):
        os.makedirs(args.plot_path + '/test/')

    #Initializing the results plot values for the different models #HR
    validation_plot_values = dict()
    test_plot_values = dict()

    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25',
                        'JM']:
        validation_plot_values[model_name] = [[], []]
        test_plot_values[model_name] = [[], []]

    #Loading indexed collection #HR

    Collection = wikIR_Collection.Collection()
    with open(args.indexed_path, 'rb') as f:
        Collection = pickle.load(f)

    Collection.doc_index[-1] = "-1"
    Collection.doc_index["-1"] = -1

    #Loading validation and test query relavance values #HR
    with open(args.coll_path + 'validation/qrels', 'r') as f_qrel:
        validation_qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.coll_path + 'test/qrels', 'r') as f_qrel:
        test_qrel = pytrec_eval.parse_qrel(f_qrel)

    print('------------------------------start--------------------------',flush=True)
    #Computing collection vocabulary size and total number of elements #HR
    coll_vocab_size, coll_tot_nb_elem = utils.evaluate_inverted_index(Collection.inverted_index)
    #Evaluating the baseline models without TDV weights and saving the results of calidation and test partitions #HR
    #HR modified the eval_baseline_index to a function eval_baseline_index_wikir
    # The previous version did not work because of a different call parameters
    utils.eval_baseline_index_wikir(args.coll_path,
                              Collection,
                              validation_qrel,
                              test_qrel,
                              validation_plot_values,
                              test_plot_values,
                              args.results_path,
                              args.experiment_name,
                              0)

    pickle.dump(validation_plot_values, open(args.plot_path + '/validation/' + args.experiment_name, 'wb'))
    pickle.dump(test_plot_values, open(args.plot_path + '/test/' + args.experiment_name, 'wb'))

    #Initialization of batch size, the loss function and optimizer for the training  #HR
    batch_gen_time = []
    batch_size = 64
    y_true = tf.ones(batch_size, )
    loss_function = tf.keras.losses.Hinge()
    optimizer = tf.keras.optimizers.Adam(args.lr)
    #Loading the differentiable model used for the training #HR
    #HR added options for different IR models. In the original version only the
    # simple tf model was present
    if args.IR_model == 'tf':
        model = differentiable_models.diff_simple_TF(Collection.embedding_matrix, dropout_rate=args.dropout_rate)
    #HR
    elif args.IR_model == 'tf_idf':
        model = differentiable_models.diff_TF_IDF(Collection.embedding_matrix, dropout_rate=args.dropout_rate)
    #HR
    elif args.IR_model == 'DIR':
        model = differentiable_models.diff_DIR(Collection.embedding_matrix, dropout_rate=args.dropout_rate)
    #HR
    elif args.IR_model == 'BM25':
        model = differentiable_models.diff_BM25(Collection.embedding_matrix, dropout_rate=args.dropout_rate)
    #HR
    elif args.IR_model == 'JM':
        model = differentiable_models.diff_JM(Collection.embedding_matrix, dropout_rate=args.dropout_rate)

    #Starting the training
    print("Start training ", args.experiment_name, flush=True)
    epoch = 0
    prop_elem_index = 1.0
    while epoch < args.nb_epoch and prop_elem_index > 0.2:

        begin = time.time()
        rank_loss = 0.0
        reg_loss = 0.0
        all_non_zero = 0.0

        begin = time.time()
        #Iterating using the generator of training batches
        for query_batch, positive_doc_batch, negative_doc_batch in Collection.generate_training_batches(batch_size):
            with tf.GradientTape() as tape:
                # reshaping queries, pos_documents and neg_documents into a numpy  ndarray #HR
                # i est le numéro les batchs qui on été prévu
                # j est le numéro interne du document, il sert à acceder à la version "direct" de chaque document
                # Tous les documents sont sous la forme d'une liste d'identifiant de vocabulaire
                queries = tf.keras.preprocessing.sequence.pad_sequences(
                    [Collection.training_queries[internal_query_ID] for internal_query_ID in query_batch], padding='post')
                pos_documents = tf.keras.preprocessing.sequence.pad_sequences(
                    [Collection.direct_structure.documents[internal_doc_ID] for internal_doc_ID in positive_doc_batch], padding='post')
                neg_documents = tf.keras.preprocessing.sequence.pad_sequences(
                    [Collection.direct_structure.documents[internal_doc_ID] for internal_doc_ID in negative_doc_batch], padding='post')
                # Creating sparse querie, pos_document and neg_documents indexes #HR
                q_sparse_index = [[column, j] for j, raw in enumerate(queries) for column in raw]
                pos_d_sparse_index = [[column, j] for j, raw in enumerate(pos_documents) for column in raw]
                neg_d_sparse_index = [[column, j] for j, raw in enumerate(neg_documents) for column in raw]
                # computing relevance and dense document for the negative and positive documents in the batch #HR
                pos_res , pos_d = model(np.clip(queries, 0, 1).astype(np.float32),
                                     queries,
                                     q_sparse_index,
                                     pos_documents,
                                     pos_d_sparse_index)

                neg_res, neg_d = model(np.clip(queries, 0, 1).astype(np.float32),
                                     queries,
                                     q_sparse_index,
                                     neg_documents,
                                     neg_d_sparse_index)

                #Computing the hinge loss , the regularization loss and total loss #HR
                ranking_loss = loss_function(y_true=y_true, y_pred=pos_res - neg_res)
                regularization_loss = tf.norm( pos_d + neg_d, ord=1)

                rank_loss += ranking_loss.numpy()
                reg_loss += regularization_loss.numpy()

                all_non_zero += tf.math.count_nonzero( pos_d + neg_d).numpy()

                loss = (1.0 - args.l1_weight) * ranking_loss + args.l1_weight * regularization_loss
                #Calculating gradients #HR
                if args.update_embeddings:
                    gradients = tape.gradient(loss, model.trainable_variables)
                else:
                    gradients = tape.gradient(loss, model.trainable_variables[1:])
            #Back propagating the gradients #HR
            if args.update_embeddings:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables[1:]))
        #Computing TDVs and saving them #HR
        weights = model.compute_index()

        pickle.dump(weights, open('weights/' + args.experiment_name + '/epoch_' + str(epoch), 'wb'))
        #updating the inverted index and computing the new idf, doc lengths and collection frequencies #HR
        inverted_index, redefined_idf, redefined_docs_length, redefined_c_freq = utils.compute_info_retrieval(
            Collection,
            weights,
            weighted=True)

        # Computing new vocab_size and total number of elements after introducting the TDV #HR
        vocab_size, tot_nb_elem = utils.evaluate_inverted_index(inverted_index)

        print(str(100 * vocab_size / coll_vocab_size)[0:5] + '% of the vocabulary is kept')
        print(str(100 * tot_nb_elem / coll_tot_nb_elem)[0:5] + '% of the index is kept', flush=True)

        prop_elem_index = tot_nb_elem / coll_tot_nb_elem

        #Evaluating baseline models with their new inverted index and new idf, doc length and collection frequencies
        #HR modified the eval_learned_index to a function eval_learned_index_wikir
        # The previous version did not work because of  different call parameters
        utils.eval_learned_index_wikir(args.coll_path,
                                 Collection,
                                 args.IR_model,
                                 model,
                                 validation_qrel,
                                 test_qrel,
                                 validation_plot_values,
                                 test_plot_values,
                                 args.plot_path,
                                 inverted_index,
                                 redefined_idf,
                                 redefined_docs_length,
                                 redefined_c_freq,
                                 prop_elem_index,
                                 args.results_path,
                                 args.experiment_name,
                                 epoch)
        epoch += 1


if __name__ == "__main__":
    main()
