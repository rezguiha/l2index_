# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: April 13 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of some global methods and the TREC definition class
# with its internal methods
# =============================================================================
import os
import time
import pickle
import random
import string
import fasttext
import numpy as np
import pytrec_eval
import pandas as pd
import subprocess

from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords

import std_tokenizer
import utils
from Inverted_structure import Inverted_structure
from  Queries import Queries

####Useful methods for trec collection and trec collection class definition#### HR

def installFromOrigin(sourcePath: str,localPath: str, max_files: int = -1):
    """
    Install a collection from original sources
    Create a XML file documents.xml, into the localPath directory
    """
    # Number of files treated
    nbFilesTreated = 0;
    # Local file name
    localFileName = os.path.join(localPath,'documents.xml')
    fd = open(localFileName,'a')
    with os.scandir(sourcePath) as entries:
        for entry in entries:
            if (entry.name.endswith('.Z')):
                nbFilesTreated += 1
                path = os.path.join(sourcePath,entry.name)
                zcat = subprocess.Popen(['zcat', path], stdout=fd)
                if max_files > 0 and nbFilesTreated >= max_files:
                    break
    fd.close()


def build_folds(queries_ids, k=5):
    """Builds folds for the K-fold cross validation """ #HR
    nb_queries = len(queries_ids)
    nb_elem = int(nb_queries / k)
    random.shuffle(queries_ids)
    folds = []
    for i in range(k):
        folds.append(queries_ids[i * nb_elem:(i + 1) * nb_elem])
    return folds


def read_queries(queries_path):
    """ Function that reads the queries from a path of the file containing those queries. It returns  dict of
    query ids and query texts""" #HR
    queries_ids = []
    queries_text = []
    with open(queries_path, 'r') as f:
        for line in f:
            if line.startswith("<num>"):
                queries_ids.append(int(line[line.find(':') + 1:-1]))
            if line.startswith("<title>"):
                pos = max(line.find(':'), line.find('>'))
                queries_text.append("".join([char for char in line[pos + 2:-1] if char not in string.punctuation]))

    return dict(zip(queries_ids, queries_text))


def read_documents(documents_path):
    """Same function but for documents""" #HR
    doc_ids = []
    doc_text = ['']
    fill_text = False
    with open(documents_path, 'r', encoding='latin1') as f:
        # Read each line, i is the line number (not used)
        for i, line in enumerate(f):
            if "<DOCNO>" in line:
                doc_ids.append(line.strip("<DOCNO> ").strip(" </DOCNO>\n"))
            if "<TEXT>" in line:
                fill_text = True
                continue
            if "</TEXT>" in line:
                continue
            if "</DOC>" in line:
                doc_text.append('')
                fill_text = False
            elif fill_text:
                doc_text[-1] += line
    del doc_text[-1]

    for i in range(len(doc_text)):
        doc_text[i] = " ".join(doc_text[i].replace('\n', ' ').split())
        doc_text[i] = "".join(char for char in doc_text[i] if char not in string.punctuation)

    return dict(zip(doc_ids, doc_text))


def save_qrel(path, qrels, subset):
    """Function that saves query document relavence scores into a file.""" #HR
    with open(path, 'w') as f:
        for query in subset:
            for doc, rel in qrels[str(query)].items():
                f.write(str(query) + '\t0\t' + doc + '\t' + str(rel) + '\n')


def save_queries_csv(coll_path, queries, folds):
    """Function that saves folds of queries into csv files for each fold""" #HR
    for i, elem in enumerate(folds):
        index = pd.Index([key for key in elem], name='id_left')
        d = {"text_left": [queries[key] for key in elem]}
        pd.DataFrame(data=d, index=index).to_csv(coll_path + '/fold' + str(i) + '/queries.csv')


def save_documents_csv(coll_path, documents):
    """Function that saves documents into a csv file""" #HR
    index = pd.Index([key for key in documents], name='id_right')
    d = {"text_right": [documents[key] for key in documents]}
    pd.DataFrame(data=d, index=index).to_csv(coll_path + '/documents.csv')

#This function has been modified by HR. It used to process the three TREC collections at once. Now it is defined
# to handle one collection at a time
def read_collection(collection_path, k=5):
    """Function that for every TREC collection reads queries , create folds for the Kfold cross validation
    ,reads the collection qrels ,save qrels and queries for each fold,reads documents
    on xml format and saves them into csv format""" #HR

    queries = read_queries(collection_path + '/queries')

    folds = build_folds(list(queries.keys()), k=k)

    with open(collection_path + '/qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    for i, fold in enumerate(folds):
        if not os.path.exists(collection_path + '/fold' + str(i)):
            os.makedirs(collection_path + '/fold' + str(i))
        save_qrel(collection_path + '/fold' + str(i) + '/qrels', qrel, fold)

    save_queries_csv(collection_path, queries, folds)

    documents = read_documents(collection_path + '/documents.xml')

    save_documents_csv(collection_path, documents)


class TrecCollection:
    def __init__(self, k=5, language='english'):
        """
        documents: pandas.DataFrame, [docid] => text string
        k: number of folders for the queries
        """
        self.documents = None
        self.k = k
        self.language = language
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

    def load_documents(self,collection_path) -> int:
        """Function that loads the documents in csv format and returns the number of the documents"""
        self.documents = pd.read_csv(collection_path + '/documents.csv', index_col='id_right', na_filter=False)
        return self.documents.size
    
    def load_folds_and_training_qrel(self,collection_path):
        """
        It loads  the folds containing the queries per fold
        in Csv format,qrels per fold and the training qrels per fold . It is run after the function read_collection
        Return the number of documents found in the cvs file
        """ #HR
        self.folds_qrels = []
        self.folds_training_qrels = []
        for i in range(self.k):
            self.folds_qrels.append(utils.read_qrels(collection_path + '/fold' + str(i) + '/qrels'))
            self.folds_training_qrels.append(utils.read_trec_train_qrels(collection_path + '/fold' + str(i) + '/qrels'))
                          
    def load_folds_queries(self, collection_path):
        """
        Loads the collection : it loads  the folds containing the queries per fold
        in Csv format,qrels per fold and the training qrels per fold . It is run after the function read_collection
        Return the number of documents found in the cvs file
        """ #HR
        
        self.folds_queries = []
        for i in range(self.k):
            self.folds_queries.append(pd.read_csv(collection_path + '/fold' + str(i) + '/queries.csv',
                                                  index_col='id_left',
                                                  na_filter=False))

        
    def build_inverted_index_and_vocabulary(self, file_path=None, save=True,minimum_occurence=5,proportion_of_frequent_words=0.2):
        """Function that builds the inverted index and  the vocabulary"""
        inverted_structure = Inverted_structure()
        start=time.time()
        # Iterating over the documents
        for document_ID, document_text in self.documents.iterrows():
            inverted_structure.inverse_document(document_ID, document_text[0])
        end=time.time()
        number_of_documents=inverted_structure.get_number_of_documents()
        print("Average time to inverse documents TREC",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Filtering vocabulary and posting lists
        start=time.time()
        inverted_structure.filter_vocabulary(minimum_occurence,proportion_of_frequent_words)  
        end=time.time()
        print("Average time to filter vocabulary,posting lists and update document lengths wikIR",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Saving      
        if save and file_path != None:
            if os.path.exists(file_path):
                start=time.time()
                inverted_structure.save(file_path)
                end=time.time()
                print("Saving time TREC",round(end-start), " s",flush=True)
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return inverted_structure
    
    
    def process_queries(self, file_path=None, save=False,vocabulary):
        # Processing training queries
        folds_processed_queries=[]
        for i in range(self.k):
            processed_queries = Queries(vocabulary)
            for query_ID, query_text in self.folds_queries[i].iterrows():
                processed_queries.process_query_and_get_ID(query_ID, query_text[0])
            folds_processed_queries.append(processed_queries)
            if save and file_path != None:
                if os.path.exists(file_path):
                    processed_queries.save(file_path, "fold"+str(i))
                else:
                    raise IOError('Path does not exist: %s' % file_path)
        return folds_processed_queries
    #To modify to fit the new inverted structure
    def generate_training_batches(self, fold, batch_size=64):
        """Function that builds batches of queries and their corresponding negative and positive documents
        for training for a particular fold. These batches are picked from outside the fold we want to
        test or validate on""" #HR
        positive_pairs = []
        negative_pairs = {}
        for i in range(self.k):
            if i != fold:
                positive_pairs += self.folds_training_indexed_qrels[i]['pos']
                negative_pairs.update(self.folds_training_indexed_qrels[i]['neg'])

        random.shuffle(positive_pairs)
        nb_docs = len(self.indexed_docs)
        nb_train_pairs = len(positive_pairs)
        query_batches = []
        positive_doc_batches = []
        negative_doc_batches = []
        pos = 0
        while (pos + batch_size < nb_train_pairs):
            query_batches.append([q for q, d in positive_pairs[pos:pos + batch_size]])
            positive_doc_batches.append([d for q, d in positive_pairs[pos:pos + batch_size]])
            neg_docs = []
            for elem in query_batches[-1]:
                neg_docs.append(random.choice(negative_pairs[self.all_queries_index[elem]]))
            negative_doc_batches.append(neg_docs)
            pos += batch_size
        return query_batches, positive_doc_batches, negative_doc_batches
