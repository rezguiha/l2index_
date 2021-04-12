# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 29 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to Evaluate baseline IR models before training for TDV
# for wikIR or wikIRS collections
# =============================================================================
#Packages
import os
import pickle
import argparse
import pytrec_eval

#Code files
import utils
import wikIR_Collection



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)

    args = parser.parse_args()

    print(args, flush=True)

    if not os.path.exists(args.results_path + '/validation/' + args.experiment_name):
        os.makedirs(args.results_path + '/validation/' + args.experiment_name)

    if not os.path.exists(args.results_path + '/test/' + args.experiment_name):
        os.makedirs(args.results_path + '/test/' + args.experiment_name)


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

    #Evaluating the baseline models without TDV weights and saving the results of validation and test partitions #HR
    utils.eval_baseline_index_wikir(args.coll_path,
                              Collection,
                              validation_qrel,
                              test_qrel,
                              validation_plot_values,
                              test_plot_values,
                              args.results_path,
                              args.experiment_name,
                              0)
    #Printing ndcg5 values for validation and test partitions #HR
    ndcg5_val = dict()
    ndcg5_test = dict()
    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25', 'JM']:
        ndcg5_val[model_name] = validation_plot_values[model_name][1][0]['ndcg_cut_5']
        print("ndcg5 validation ",model_name," of collection ",os.path.basename(args.coll_path)," ",ndcg5_val[model_name] ,flush=True)
        ndcg5_test[model_name] = test_plot_values[model_name][1][0]['ndcg_cut_5']
        print("ndcg5 test ",model_name," of collection ",os.path.basename(args.coll_path)," ",ndcg5_test[model_name] ,flush=True)


if __name__ == "__main__":
    main()