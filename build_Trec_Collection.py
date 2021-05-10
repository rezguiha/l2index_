# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: April 13 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Parsing arguments and launching steps to build a TREC collection
# =============================================================================
import argparse
import os
import time
from Trec_Collection import TrecCollection
import Trec_Collection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--original_coll_path', type=str,help='if present, then produces documents.xml from original source')
    parser.add_argument('-m', '--max_files', type=int,help='if present, then only this number of files are treated from original source')
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--index_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)
    parser.add_argument('-l', '--language', nargs="?", type=str, default='english')
    parser.add_argument('-b', '--build_folds_preprocess', nargs="?", type=bool, default=False) #HR

    args = parser.parse_args()

    print('-----------------start--------------',flush=True)
    # Set to True only if the collection haven't been processed by creating folds containing documents
    # , queries and qrels for each fold and creating csv files from Xml format
    #HR

    # For total time execution
    startTime = time.time()

    if args.max_files is not None:
        max_files = args.max_files
        print("Max files treated:",max_files)
    else:
        max_files = -1;

    # if we add the argument --original_coll_path then the original files
    # are uncompressed ans assembled into the XML file documents.xml
    # Do nothing if documents.xml aready exists
    if args.original_coll_path is not None:
        if not os.path.isfile(os.path.join(args.coll_path,'documents.xml')):
            print("Build documents.xml from "+args.original_coll_path)
            Trec_Collection.installFromOrigin(args.original_coll_path,args.coll_path,max_files)
        else:
            print('documents.xml => OK')

    if args.build_folds_preprocess:
        print("Creating folds for the K-fold cross validation and preprocessing",flush=True)
        Trec_Collection.read_collection(args.coll_path)

    print("Loading collection (documents ,folds queries and qrels and folds training qrels",flush=True)
    Collection = TrecCollection(k=5)
    nbDocs = Collection.load_collection(args.coll_path)
    print("Number of document:",nbDocs)
    print("Process time estimation:",round(106.0*nbDocs/6893.0),"s")

    print("=> Standard preprocess (time estimation:",round(84.0*nbDocs/6893.0),"s)",flush=True)
    tic = time.time()
    Collection.standard_preprocess(remove_stopwords=True, min_occ=5)
    toc = time.time()
    print("<=",round(toc-tic),"s")

    print("=> Compute inverted index",flush=True)
    tic = time.time()
    Collection.compute_info_retrieval()
    toc = time.time()
    print("<=",round(toc-tic),"s")


    print("=> Compute fasttext embeddings", flush=True)
    tic = time.time()
    Collection.compute_fasttext_embedding(args.fasttext_path)
    toc = time.time()
    print("<=",round(toc-tic),"s")

    print("Saving pickle file",flush=True)
    Collection.pickle_indexed_collection(args.index_path + 'indexed_collection')

    totalTime = time.time()-startTime
    print("Total time: ",round(totalTime),"s")
    print("Number of doc processed per s:",round(nbDocs/totalTime))
    print("-----------------Finished----------------", flush=True)


if __name__ == "__main__":
    main()
