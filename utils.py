# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of the classes of differentiable IR models with their 
# network architecture
# =============================================================================
import os
import json
import pickle
import random
import collections
import numpy as np
import pytrec_eval
import tensorflow as tf
from collections import Counter

import baseline_models_and_tdv_implementation

####Useful functions for reading qrel files when you are loading the collection #HR
def read_qrels(path):
    """Reads qrel files for both Wikir and Trec collection""" #HR
    qrels = []
    with open(path,'r') as f:
        for line in f:
            rel = line.split()
            qrels.append([rel[0],rel[2]])
    return qrels

def read_trec_train_qrels(path):
    """read qrel files for trec collection for training and build dictionnary containing qrels for positive documents and negative documents""" #HR
    pos_qrels = []
    neg_qrels = dict()
    with open(path,'r') as f:
        for line in f:
            rel = line.split()
            if rel[3] == '1':
                pos_qrels.append([rel[0],rel[2]])
            else :
                if rel[0] not in neg_qrels:
                    neg_qrels[rel[0]] = [rel[2]]
                else:
                    neg_qrels[rel[0]].append(rel[2])
    return {'pos':pos_qrels,'neg':neg_qrels}







####Useful functions for after computation of TDV weights by training models to update inverted index, compute some properties like idf,evaluate performance
### of baseline models and their associated TDV implementation  #HR

def build_inverted_index(Collection, weights):
    """Function that updates the inverted index of a collection by erasing the tokens that have 0 as TDV and multiply the rest by their TDV weight""" #HR
    inverted_index = dict()
    for key, value in Collection.inverted_index.items():
        if weights[key] == 0:
            continue
        inverted_index[key] = Counter()
        for doc_id in value:
            inverted_index[key][doc_id] += weights[key] * Collection.inverted_index[key][doc_id]
    return inverted_index


def compute_idf(Collection, inverted_index, weights=None):
    """Functions that compute the idf with or without introduction of TDV""" #HR
    nb_docs = len(Collection.doc_index)
    if weights is None:
        return {token: np.log((nb_docs + 1) / (1 + len(inverted_index[token]))) for token in inverted_index}
    else:
        sums = {key: sum(inverted_index[key].values()) for key in inverted_index}
        maxdf = max(sums.values())
        return {token: np.log((maxdf + 1) / (1 + sums[token])) for token in inverted_index}


    # Here in the following function we give the weights when we want the docslengths to be the number of occurence
    # the weights are here for regularization purposes
def compute_docs_length(inverted_index, weights=None):
    """Function that computes document length with TDV or without it""" #HR
    docs_length = Counter()

    if weights is None:
        for term, posting in inverted_index.items():
            for doc_id, nb_occurence in posting.items():
                docs_length[doc_id] += nb_occurence

    else:
        for term, posting in inverted_index.items():
            for doc_id, nb_occurence in posting.items():
                docs_length[doc_id] += nb_occurence / weights[term]

    return docs_length


def compute_collection_frequencies(docs_length, inverted_index):
    """Function that computes frequency of tokens in a  collection""" #HR
    coll_length = sum([value for key, value in docs_length.items()])
    return {token: sum([freq for _, freq in inverted_index[token].items()]) / coll_length for token in inverted_index}


def evaluate_inverted_index(inverted_index):
    """Function that takes an inverted index and calculate its vocabulary size and total number of elements""" #HR
    vocab_size = len(inverted_index)
    tot_nb_elem = 0
    for key, value in inverted_index.items():
        tot_nb_elem += len(value)
    return vocab_size, tot_nb_elem


def compute_metrics(coll_path, Collection, queries_index, qrel, results, model_name, save_res=False):
    """Function that saves the results of retrieval: the top_k documents according to their score for
    a certain model identified by model_name. Then, it computes different metrics for IR using the pytrec_eval
    package""" #HR
    Collection.save_results(queries_index, results, model_name, top_k=1000)

    with open(model_name, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    if not save_res:
        os.remove(model_name)

    #measures = {"map", "ndcg_cut", "recall", "P"}
    measures = {"ndcg_cut"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)

    all_metrics = evaluator.evaluate(run)

    metrics = {'P_5': 0,
               'P_10': 0,
               'P_20': 0,
               'ndcg_cut_5': 0,
               'ndcg_cut_10': 0,
               'ndcg_cut_20': 0,
               'ndcg_cut_1000': 0,
               'map': 0,
               'recall_1000': 0}

    nb_queries = len(all_metrics)
    for key, values in all_metrics.items():
        for metric in metrics:
            metrics[metric] += values[metric] / nb_queries

    return metrics



def utils_compute_info_retrieval(Collection, weights, weighted=True):
    """Computes inverted index, idf, document length and c_frequency for a collection with TDV weights""" #HR
    inverted_index = build_inverted_index(Collection, weights)
    if weighted:
        idf = compute_idf(Collection, inverted_index, weights)
        docs_length = compute_docs_length(inverted_index)
        c_freq = compute_collection_frequencies(docs_length, inverted_index)
    else:
        idf = compute_idf(Collection, inverted_index)
        docs_length = compute_docs_length(inverted_index, weights)
        c_freq = compute_collection_frequencies(docs_length, inverted_index)
    return inverted_index, idf, docs_length, c_freq


# HR added this function to evaluate baseline models on TREC. It is a modified version of eval_baseline_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR
def eval_baseline_index_trec(coll_path,
                        Collection,
                        fold,
                        qrel,
                        plot_values,
                        results_path,
                        experiment_name,
                        epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR
    print('tf')

    results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_queries[fold],
                                  Collection.inverted_index)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.queries_index[fold],
                              qrel,
                              results,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/' + str(epoch))

    plot_values['tf'][0].append(1.0)
    plot_values['tf'][1].append(metrics)

    print('tf_idf')

    results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_queries[fold],
                     Collection.inverted_index,
                     Collection.idf)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.queries_index[fold],
                              qrel,
                              results,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/' + str(epoch))

    plot_values['tf_idf'][0].append(1.0)
    plot_values['tf_idf'][1].append(metrics)

    print('DIR')

    results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_queries[fold],
                                 Collection.inverted_index,
                                 Collection.docs_length,
                                 Collection.c_freq)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.queries_index[fold],
                              qrel,
                              results,
                              results_path + '/fold' + str(fold) + '/' + experiment_name + '/DIR/' + str(epoch))

    plot_values['DIR'][0].append(1.0)
    plot_values['DIR'][1].append(metrics)

    print('BM25')

    results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_queries[fold],
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.idf)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.queries_index[fold],
                              qrel,
                              results,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/' + str(epoch))

    plot_values['BM25'][0].append(1.0)
    plot_values['BM25'][1].append(metrics)

    print('JM')

    results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_queries[fold],
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.c_freq)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.queries_index[fold],
                              qrel,
                              results,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/' + str(epoch))

    plot_values['JM'][0].append(1.0)
    plot_values['JM'][1].append(metrics)


    
    # HR added this function to evaluate baseline models on wikIR collections. It is a modified version of eval_baseline_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too
def eval_baseline_index_wikir(coll_path,
                        Collection,
                        validation_qrel,
                        test_qrel,
                        validation_plot_values,
                        test_plot_values,
                        results_path,
                        experiment_name,
                        epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR
    print('tf',flush=True)
    #validation
    results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_validation_queries,
                                  Collection.inverted_index)

    if not os.path.exists(results_path + '/validation/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/validation/' + experiment_name + '/tf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' + experiment_name + '/tf/' + str(epoch))

    validation_plot_values['tf'][0].append(1.0)
    validation_plot_values['tf'][1].append(metrics)
    
    #Test
    results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_test_queries,
                                  Collection.inverted_index)

    if not os.path.exists(results_path + '/test/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/test/' + experiment_name + '/tf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' + experiment_name + '/tf/' + str(epoch))

    test_plot_values['tf'][0].append(1.0)
    test_plot_values['tf'][1].append(metrics)

    print('tf_idf',flush=True)
    #validation
    results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_validation_queries,
                     Collection.inverted_index,
                     Collection.idf)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/tf_idf/' + str(epoch))

    validation_plot_values['tf_idf'][0].append(1.0)
    validation_plot_values['tf_idf'][1].append(metrics)
    #test
    results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_test_queries,
                     Collection.inverted_index,
                     Collection.idf)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/tf_idf/' + str(epoch))

    test_plot_values['tf_idf'][0].append(1.0)
    test_plot_values['tf_idf'][1].append(metrics)

    print('DIR',flush=True)
    #validation
    results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_validation_queries,
                                 Collection.inverted_index,
                                 Collection.docs_length,
                                 Collection.c_freq)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' + experiment_name + '/DIR/' + str(epoch))

    validation_plot_values['DIR'][0].append(1.0)
    validation_plot_values['DIR'][1].append(metrics)

    #test
    
    results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_test_queries,
                                 Collection.inverted_index,
                                 Collection.docs_length,
                                 Collection.c_freq)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' + experiment_name + '/DIR/' + str(epoch))

    test_plot_values['DIR'][0].append(1.0)
    test_plot_values['DIR'][1].append(metrics)
    
    print('BM25',flush=True)
    #validation
    results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_validation_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.idf)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/BM25/' + str(epoch))

    validation_plot_values['BM25'][0].append(1.0)
    validation_plot_values['BM25'][1].append(metrics)
    
    #test
    results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_test_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.idf)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/BM25/' + str(epoch))

    test_plot_values['BM25'][0].append(1.0)
    test_plot_values['BM25'][1].append(metrics)
    
    print('JM',flush=True)
    #validation
    
    results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_validation_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.c_freq)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/JM/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/JM/' + str(epoch))

    validation_plot_values['JM'][0].append(1.0)
    validation_plot_values['JM'][1].append(metrics)

    #test
    results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_test_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.c_freq)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/JM/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/JM/' + str(epoch))

    test_plot_values['JM'][0].append(1.0)
    test_plot_values['JM'][1].append(metrics)

    # HR added this function to evaluate baseline models on TREC after training to get the TDV weights. It is a modified version of eval_learned_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR
def eval_learned_index_trec(coll_path,
                       Collection,
                       IR_model,
                       model,
                       qrel,
                       plot_values,
                       plot_path,
                       fold,
                       inverted_index,
                       weights,
                       redefined_idf,
                       redefined_docs_length,
                       redefined_c_freq,
                       #                        idf,
                       #                        docs_length,
                       #                        c_freq,
                       prop_elem_index,
                       results_path,
                       experiment_name,
                       epoch):
    """Evaluate the performance of baseline models and their corresponding weighted (TDV) versions and saves the results in a pickle object file
    This is is to be used after training the neural model (calculating the TDV weights of terms) """ #HR

    if IR_model == 'tf':

        print('tf')

        results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_queries[fold],
                            inverted_index)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/')

        #         print(results,flush=True)

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.queries_index[fold],
                                  qrel,
                                  results,
                                  results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/' + str(epoch))

        plot_values['tf'][0].append(prop_elem_index)
        plot_values['tf'][1].append(metrics)

#         print('weighted_tf')

#         results = baseline_models_and_tdv_implementation.weighted_simple_tf(Collection.indexed_queries[fold],
#                                      inverted_index,
#                                      weights)

#         if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_tf/'):
#             os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_tf/')

#         metrics = compute_metrics(coll_path,
#                                   Collection,
#                                   Collection.queries_index[fold],
#                                   qrel,
#                                   results,
#                                   results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_tf/' + str(epoch))

#         plot_values['weighted_tf'][0].append(prop_elem_index)
#         plot_values['weighted_tf'][1].append(metrics)

    if IR_model == 'tf_idf':

        print('tf_idf')

        results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_queries[fold],
                         inverted_index,
                         redefined_idf)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.queries_index[fold],
                                  qrel,
                                  results,
                                  results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/' + str(epoch))

        plot_values['tf_idf'][0].append(prop_elem_index)
        plot_values['tf_idf'][1].append(metrics)

#         print('weighted_tf_idf')

#         results = baseline_models_and_tdv_implementation.weighted_tf_idf(Collection.indexed_queries[fold],
#                                   inverted_index,
#                                   weights,
#                                   redefined_idf)

#         if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_tf_idf/'):
#             os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_tf_idf/')

#         metrics = compute_metrics(coll_path,
#                                   Collection,
#                                   Collection.queries_index[fold],
#                                   qrel,
#                                   results,
#                                   results_path + '/fold' + str(fold) + '/' + model_name + '/weighted_tf_idf/' + str(
#                                       epoch))

#         plot_values['weighted_tf_idf'][0].append(prop_elem_index)
#         plot_values['weighted_tf_idf'][1].append(metrics)

    if IR_model == 'DIR':

        mu = model.mu.numpy()

        print('DIR')

        results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_queries[fold],
                                     inverted_index,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     mu=mu)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.queries_index[fold],
                                  qrel,
                                  results,
                                  results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/' + str(epoch))

        plot_values['DIR'][0].append(prop_elem_index)
        plot_values['DIR'][1].append(metrics)

#         print('weighted_DIR')

#         results = baseline_models_and_tdv_implementation.weighted_dir_language_model(Collection.indexed_queries[fold],
#                                               inverted_index,
#                                               weights,
#                                               redefined_docs_length,
#                                               redefined_c_freq,
#                                               mu=mu)

#         if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_DIR/'):
#             os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_DIR/')

#         metrics = compute_metrics(coll_path,
#                                   Collection,
#                                   Collection.queries_index[fold],
#                                   qrel,
#                                   results,
#                                   results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_DIR/' + str(epoch))

#         plot_values['weighted_DIR'][0].append(prop_elem_index)
#         plot_values['weighted_DIR'][1].append(metrics)

    if IR_model == 'BM25':

        k1 = model.k1.numpy()
        b = model.b.numpy()

        print('BM25')

        results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_queries[fold],
                             inverted_index,
                             redefined_docs_length,
                             redefined_idf,
                             k1=k1,
                             b=b)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.queries_index[fold],
                                  qrel,
                                  results,
                                  results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/' + str(epoch))

        plot_values['BM25'][0].append(prop_elem_index)
        plot_values['BM25'][1].append(metrics)

#         print('weighted_BM25')

#         results = baseline_models_and_tdv_implementation.weighted_Okapi_BM25(Collection.indexed_queries[fold],
#                                       inverted_index,
#                                       weights,
#                                       redefined_docs_length,
#                                       redefined_idf,
#                                       k1=k1,
#                                       b=b)

#         if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_BM25/'):
#             os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_BM25/')

#         metrics = compute_metrics(coll_path,
#                                   Collection,
#                                   Collection.queries_index[fold],
#                                   qrel,
#                                   results,
#                                   results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_BM25/' + str(
#                                       epoch))

#         plot_values['weighted_BM25'][0].append(prop_elem_index)
#         plot_values['weighted_BM25'][1].append(metrics)

    if IR_model == 'JM':
        lamb=model.lamb.numpy()

        print('JM')

        results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_queries[fold],
                             inverted_index,
                             redefined_docs_length,
                             redefined_c_freq,
                             lamb)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.queries_index[fold],
                                  qrel,
                                  results,
                                  results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/' + str(epoch))

        plot_values['JM'][0].append(prop_elem_index)
        plot_values['JM'][1].append(metrics)

#         print('weighted_JM')

#         results = baseline_models_and_tdv_implementation.weighted_JM_language_model(Collection.indexed_queries[fold],
#                                       inverted_index,
#                                       weights,
#                                       redefined_docs_length,
#                                       redefined_c_freq,
#                                       lamb)

#         if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_JM/'):
#             os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_JM/')

#         metrics = compute_metrics(coll_path,
#                                   Collection,
#                                   Collection.queries_index[fold],
#                                   qrel,
#                                   results,
#                                   results_path + '/fold' + str(fold) + '/' +  experiment_name + '/weighted_JM/' + str(
#                                       epoch))

#         plot_values['weighted_JM'][0].append(prop_elem_index)
#         plot_values['weighted_JM'][1].append(metrics)


    pickle.dump(plot_values, open(plot_path + '/fold' + str(fold) + '/' +  experiment_name, 'wb'))

    # HR added this function to evaluate baseline models on TREC after training to get the TDV weights. It is a modified version of eval_learned_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR
    
def eval_learned_index_wikir(coll_path,
                       Collection,
                       IR_model,
                       model, 
                       validation_qrel,
                       test_qrel,
                       validation_plot_values,
                       test_plot_values,
                       plot_path,
                       inverted_index,
                       redefined_idf,
                       redefined_docs_length,
                       redefined_c_freq,
                       #                        idf,
                       #                        docs_length,
                       #                        c_freq,
                       prop_elem_index,
                       results_path,
                       experiment_name,
                       epoch):
    """Evaluate the performance of baseline models and their corresponding weighted (TDV) versions and saves the results in a pickle object file
    This is is to be used after training the neural model (calculating the TDV weights of terms) """ #HR

    if IR_model == 'tf':

        print('tf')
        #validation
        results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_validation_queries,
                            inverted_index)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/tf/')


        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/tf/' + str(epoch))

        validaton_plot_values['tf'][0].append(prop_elem_index)
        validation_plot_values['tf'][1].append(metrics)
        
        #test
        results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_test_queries,
                            inverted_index)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/tf/')


        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/tf/' + str(epoch))

        test_plot_values['tf'][0].append(prop_elem_index)
        test_plot_values['tf'][1].append(metrics)



    if IR_model == 'tf_idf':

        print('tf_idf')
        #validation
        results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_validation_queries,
                         inverted_index,
                         redefined_idf)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/tf_idf/' + str(epoch))

        validation_plot_values['tf_idf'][0].append(prop_elem_index)
        validation_plot_values['tf_idf'][1].append(metrics)
        
        #test
        results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_test_queries,
                         inverted_index,
                         redefined_idf)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/tf_idf/' + str(epoch))

        test_plot_values['tf_idf'][0].append(prop_elem_index)
        test_plot_values['tf_idf'][1].append(metrics)


    if IR_model == 'DIR':

        mu = model.mu.numpy()

        print('DIR')
        #validation
        results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_validation_queries,
                                     inverted_index,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     mu=mu)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/DIR/' + str(epoch))

        validation_plot_values['DIR'][0].append(prop_elem_index)
        validation_plot_values['DIR'][1].append(metrics)
        
        #test
        results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_test_queries,
                                     inverted_index,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     mu=mu)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/DIR/' + str(epoch))

        test_plot_values['DIR'][0].append(prop_elem_index)
        test_plot_values['DIR'][1].append(metrics)



    if IR_model == 'BM25':

        k1 = model.k1.numpy()
        b = model.b.numpy()

        print('BM25')
        #validation
        results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_validation_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_idf,
                             k1=k1,
                             b=b)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/validation/'  +  experiment_name + '/BM25/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/'  +  experiment_name + '/BM25/' + str(epoch))

        validation_plot_values['BM25'][0].append(prop_elem_index)
        validation_plot_values['BM25'][1].append(metrics)
        
        #test
        results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_test_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_idf,
                             k1=k1,
                             b=b)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/test/'  +  experiment_name + '/BM25/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/'  +  experiment_name + '/BM25/' + str(epoch))

        test_plot_values['BM25'][0].append(prop_elem_index)
        test_plot_values['BM25'][1].append(metrics)


    if IR_model == 'JM':
        lamb=model.lamb.numpy()

        print('JM')
        #validation
        results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_validation_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_c_freq,
                             lamb)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/JM/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/JM/' + str(epoch))

        validation_plot_values['JM'][0].append(prop_elem_index)
        validation_plot_values['JM'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_test_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_c_freq,
                             lamb)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/JM/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/JM/' + str(epoch))

        test_plot_values['JM'][0].append(prop_elem_index)
        test_plot_values['JM'][1].append(metrics)


    pickle.dump(validation_plot_values, open(plot_path + '/validation/' +  experiment_name, 'wb'))
    pickle.dump(test_plot_values, open(plot_path + '/test/' +  experiment_name, 'wb'))
