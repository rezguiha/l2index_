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
import time
#Code files
import utils
import wikIR_Collection_opt
import Inverted_structure
import baseline_models_and_tdv_implementation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)

    args = parser.parse_args()

    print(args, flush=True)
    start0=time.time()




    #Loading indexed_structure#HR
    start=time.time()
    inverted_structure=Inverted_structure.Inverted_structure()
    inverted_structure.load(args.indexed_path)
    end=time.time()
    print("Time for loading indexed wikIR collection ",end-start,flush=True)
    #Computing idf and collection frequencies
    start=time.time()
    inverted_structure.compute_idf()
    inverted_structure.compute_collection_frequencies()
    end=time.time()
    print("Time for computing idf and collection frequencies ", end-start,flush=True)
    #Loading queries and processing queries
    start=time.time()
    Collection = wikIR_Collection_opt.Collection()
    Collection.load_queries(args.coll_path)
    _,_,test_queries_struct=Collection.process_queries()
    end=time.time()
    print("Time to load queries and process them ",end-start,flush=True) 

    print('------------------------------start--------------------------',flush=True)


    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25', 'JM']:
        start=time.time()
        
        if model_name=='tf':
            baseline_model=baseline_models_and_tdv_implementation.simple_tf(test_queries_struct,inverted_structure)
        if model_name=='tf_idf':
            baseline_model=baseline_models_and_tdv_implementation.tf_idf(test_queries_struct,inverted_structure)
        if model_name=='DIR':
            baseline_model=baseline_models_and_tdv_implementation.dir_language_model(test_queries_struct,inverted_structure)
        if model_name=='BM25':
            baseline_model=baseline_models_and_tdv_implementation.Okapi_BM25(test_queries_struct,inverted_structure)
        if model_name=='JM':
            baseline_model=baseline_models_and_tdv_implementation.JM_language_model(test_queries_struct,inverted_structure)
        count=0
        for result in baseline_model.runQueries():
            count+=1
        end=time.time()
        print("Average time to compute results for model ", model_name, " = ", round((end-start)*1000/count),flush=True)
    print('-----------------------------------Finished -----------------------',flush=True)
if __name__ == "__main__":
    main()