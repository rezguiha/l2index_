import argparse
import wikIR_Collection


################2. Building Wikir Collections####################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)

    args = parser.parse_args()

    print(args, flush=True)

    print('Reading collection', flush=True)

    Collection_wikir = wikIR_Collection.Collection('english')
    Collection_wikir.load_collection(args.coll_path)

    print('Standard Preprocess', flush=True)
    Collection_wikir.standard_preprocess(remove_stopwords=True,
                                         min_occ=5)

    print('Compute inverted index', flush=True)
    Collection_wikir.compute_info_retrieval()
    print('Compute fasstetext embeddings', flush=True)
    Collection_wikir.compute_fasttext_embedding(args.fasttext_path)
    print('Saving pickle file',flush=True)
    Collection_wikir.pickle_indexed_collection(args.indexed_path + '/indexed_collection')
    print('Finished',flush=True)
if __name__ == "__main__":
    main()
