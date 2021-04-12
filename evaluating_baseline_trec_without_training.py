# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 29 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to Evaluate baseline IR models before training for TDV
# for a TREC Collection 
# =============================================================================
import os
import pickle
import argparse

import pytrec_eval

from Trec_Collection import TrecCollection
import utils



def main():
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-f', '--folds', nargs="?", type=int, default=5)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)


    args = parser.parse_args()

    print(args, flush=True)

    # Loading indexed collection
    Collection = TrecCollection()
    with open(args.indexed_path, 'rb') as f:
        Collection = pickle.load(f)

    Collection.doc_index[-1] = "-1"
    Collection.doc_index["-1"] = -1
    # Loading relevance judgements from collection
    with open(args.coll_path + 'qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # ????
    id_titl = Collection.vocabulary['titl']

    for i in range(len(Collection.all_indexed_queries)):

        if Collection.all_indexed_queries[i][0] == id_titl and len(Collection.all_indexed_queries[i]) > 1:
            del Collection.all_indexed_queries[i][0]

    for i in range(len(Collection.indexed_queries)):
        for j in range(len(Collection.indexed_queries[i])):
            if Collection.indexed_queries[i][j][0] == id_titl and len(Collection.indexed_queries[i][j]) > 1:
                del Collection.indexed_queries[i][j][0]

    print('---------------------start-------------------', flush=True)
    # Getting collection vocabulary size and total number of elements in collection
    coll_vocab_size, coll_tot_nb_elem = utils.evaluate_inverted_index(Collection.inverted_index)
    # Creating for each fold and for a certain experiment a directory for results and plots data
    plot_values_folds_list=[]
    for fold in range(args.folds):

        plot_values = dict()

        for model_name in ['tf',
                           'tf_idf',
                           'DIR',
                           'BM25', 'JM']:
            plot_values[model_name] = [[], []]

        if not os.path.exists(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name)


        if not os.path.exists(args.plot_path + '/fold' + str(fold) + '/'):
            os.makedirs(args.plot_path + '/fold' + str(fold) + '/')
        # Computing metrics for baseline models for a certain fold and updating plot_values dictionnary
        utils.eval_baseline_index_trec(args.coll_path,
                                       Collection,
                                       fold,
                                       qrel,
                                       plot_values,
                                       args.results_path,
                                       args.experiment_name,
                                       0)
        # appending plot values to the list
        plot_values_folds_list.append(plot_values)
    #Evaluating baseline models without training.
        
    ndcg5 =dict()
    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25', 'JM']:
        ndcg5[model_name] = []
    for fold in range(args.folds):
        for model_name in ['tf',
                           'tf_idf',
                           'DIR',
                           'BM25', 'JM']:
            ndcg5[model_name].append(plot_values_folds_list[fold][model_name][1][0]['ndcg_cut_5'])
    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25', 'JM']:
        average=sum(ndcg5[model_name])/args.folds
        maximum=max(ndcg5[model_name])
        print("ndcg5 ",model_name," average of folds", average," of collection ",os.path.basename(args.coll_path) ,flush=True)
        print("ndcg5 ", model_name, " max of folds", maximum," of collection ",os.path.basename(args.coll_path), flush=True)

    print("-----------------Finished-------------------", flush=True)


if __name__ == "__main__":
    main()
