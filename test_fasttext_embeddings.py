from Inverted_structure import Inverted_structure
import argparse
import numpy as np
import time
import array as arr
import fasttext
def test_fasttext_inverted_structure(indexed_path,fasttext_path):
    inverted_structure=Inverted_structure()
    inverted_structure.load(indexed_path)
    vocab_size=inverted_structure.get_vocabulary_size()
    print("Vocabulary size is =",vocab_size,flush=True)
    start=time.time()
    embedding_matrix=np.zeros((vocab_size,300),dtype=np.float32)
    model = fasttext.load_model(fasttext_path)
    index=0
    for token in inverted_structure.vocabulary:
        embedding_matrix[index] = model[token]
        index+=1
    end=time.time()
    print("Normal Time to compute embeddings = ",round(end-start),"s",flush=True)
    start=time.time()
    inverted_structure.compute_and_save_fasttext_embeddings(fasttext_path,indexed_path)
    end=time.time()
    print("Time to compute and save embeddings =",round(end-start),"s",flush=True)
    start=time.time()
    inverted_structure.load_fasttext_embeddings(indexed_path)
    end=time.time()
    print("Time to load embeddings = ",round(end-start),"s",flush=True)
    
    if np.allclose(inverted_structure.embedding_matrix,embedding_matrix):
        print("Save and load of fasttext embedding is successful",flush=True)
    else:
        print("Save and load of fasttext embedding is not successful",flush=True)
    
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indexed_path')
    parser.add_argument('-f','--fasttext_path')
    args = parser.parse_args()

    test_fasttext_inverted_structure(args.indexed_path,args.fasttext_path)


if __name__ == "__main__":
    main()