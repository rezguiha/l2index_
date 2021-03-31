# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: April 13 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Parsing arguments and launching steps to build a TREC collection
# =============================================================================
import argparse
from Trec_Collection import TrecCollection
import Trec_Collection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--index_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)
    parser.add_argument('-l', '--language', nargs="?", type=str, default='english')
    parser.add_argument('-b', '--build_folds_preprocess', nargs="?", type=bool, default=False) #HR

    args = parser.parse_args()
    print(args)

    print('-----------------start--------------',flush=True)
    # Set to True only if the collection haven't been processed by creating folds containing documents
    # , queries and qrels for each fold and creating csv files from Xml format
    #HR
    if args.build_folds_preprocess:
        print("Creating folds for the K-fold cross validation and preprocessing",flush=True)
        Trec_Collection.read_collection(args.coll_path)
    
    print("Loading collection (documents ,folds queries and qrels and folds training qrels",flush=True)
    Collection = TrecCollection(k=5)
    Collection.load_collection(args.coll_path)

    print("Standard preprocess",flush=True)
    Collection.standard_preprocess(remove_stopwords=True,
                                   min_occ=5)
    print("Compute inverted index",flush=True)
    Collection.compute_info_retrieval()
    print("Compute fasttext embeddings", flush=True)
    Collection.compute_fasttext_embedding(args.fasttext_path)
    print("Saving pickle file",flush=True)
    Collection.pickle_indexed_collection(args.index_path + 'indexed_collection')
    print("-----------------Finished----------------", flush=True)


if __name__ == "__main__":
    main()
