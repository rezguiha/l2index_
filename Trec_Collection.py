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

####Useful methods for trec collection and trec collection class definition#### HR

def installFromOrigin(sourcePath: str,localPath: str):
    """
    Install a collection from original sources
    Create a XML file documents.xml, into the localPath directory
    """
    # Local file name
    localFileName = os.path.join(localPath,'documents.xml')
    fd = open(localFileName,'a')
    with os.scandir(sourcePath) as entries:
        for entry in entries:
            if (entry.name.endswith('.Z')):
                path = os.path.join(sourcePath,entry.name)
                zcat = subprocess.Popen(['zcat', path], stdout=fd)


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
        self.documents = None
        self.k = k
        self.language = language
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

    def load_collection(self, collection_path):
        """Function that loads the collection : it loads documents and the folds containing the queries per fold
        in Csv format,qrels per fold and the training qrels per fold . It is run after the function read_collection""" #HR
        self.documents = pd.read_csv(collection_path + '/documents.csv', index_col='id_right', na_filter=False)

        self.folds_queries = []
        self.folds_qrels = []
        self.folds_training_qrels = []
        for i in range(self.k):
            self.folds_queries.append(pd.read_csv(collection_path + '/fold' + str(i) + '/queries.csv',
                                                  index_col='id_left',
                                                  na_filter=False))
            self.folds_qrels.append(utils.read_qrels(collection_path + '/fold' + str(i) + '/qrels'))
            self.folds_training_qrels.append(utils.read_trec_train_qrels(collection_path + '/fold' + str(i) + '/qrels'))

    def update_standard_vocabulary(self, sequences, remove_stopwords=True):
        """Function that updates the standard vocabulary using new sequences""" #HR
        count = 0
        if remove_stopwords:
            for _, sequence in sequences.iterrows():
                for word in sequence[0].split(" "):
                    temp = word.lower()
                    if temp not in self.stop_words:
                        self.vocabulary[self.stemmer.stem(temp)] += 1
                count += 1
        else:
            for _, sequence in sequences.iterrows():
                for word in sequence[0].split(" "):
                    self.vocabulary[self.stemmer.stem(word.lower())] += 1
                count += 1

    def build_standard_vocabulary(self,
                                  min_occ=2,
                                  remove_stopwords=True):
        """Function that builds the standard vocabulary from documents with minimum occurence equal to 2""" #HR
        self.vocabulary = Counter()

        self.update_standard_vocabulary(self.documents, remove_stopwords)

        for i in range(self.k):
            self.update_standard_vocabulary(self.folds_queries[i], remove_stopwords)

        del self.vocabulary['']

        self.vocabulary = {i + 1: elem[0] for i, elem in enumerate(self.vocabulary.most_common()) if elem[1] >= min_occ}

        for key in list(self.vocabulary):
            self.vocabulary[self.vocabulary[key]] = key

        self.vocabulary[0] = '<PAD>'
        self.vocabulary['<PAD>'] = 0

    def standard_preprocess(self,
                            remove_stopwords=True,
                            min_occ=5):
        """General function that preprocesses the Trec collection by building vocabulary, using the tokenizer
        to index documents, the folds of queries and all the queries. It is run after the method
        load_collection""" #HR
        self.build_standard_vocabulary(min_occ=min_occ,
                                       remove_stopwords=remove_stopwords)

        self.doc_index, self.indexed_docs = std_tokenizer.std_tokenizer_index(self.documents,
                                                                self.vocabulary,
                                                                self.stemmer)

        self.queries_index = []
        self.indexed_queries = []

        for i in range(self.k):
            queries_index, indexed_queries = std_tokenizer.std_tokenizer_index(self.folds_queries[i],
                                                                 self.vocabulary,
                                                                 self.stemmer)
            self.queries_index.append(queries_index)
            self.indexed_queries.append(indexed_queries)

        self.all_indexed_queries = []
        for elem in self.indexed_queries:
            self.all_indexed_queries += elem

        self.all_queries_index = dict()
        counter = 0
        for i in range(len(self.queries_index)):
            for j in range(int(len(self.queries_index[i]) / 2)):
                self.all_queries_index[counter] = self.queries_index[i][j]
                self.all_queries_index[self.queries_index[i][j]] = counter
                counter += 1

    def build_inverted_index(self):
        """Function that builds the inverted index of documents """ #HR
        self.inverted_index = dict()

        for token in self.vocabulary:
            if isinstance(token, int):
                self.inverted_index[token] = Counter()

        for i, indexed_document in enumerate(self.indexed_docs):
            for token in indexed_document:
                self.inverted_index[token][i] += np.float32(1.0)

    def compute_idf(self):
        """Funciton that computes the idf of every term in te inverted index""" #HR
        nb_docs = len(self.doc_index)
        self.idf = {token: np.log((nb_docs + 1) / (1 + len(self.inverted_index[token]))) for token in
                    self.inverted_index}

    def compute_docs_length(self):
        """Function that computes the length of each document in the collection""" #HR
        self.docs_length = {i: len(doc) for i, doc in enumerate(self.indexed_docs)}

    def compute_collection_frequencies(self):
        """Function that computes frequency of words in the collection""" #HR
        coll_length = sum([value for key, value in self.docs_length.items()])
        self.c_freq = {token: sum([freq for _, freq in self.inverted_index[token].items()]) / coll_length for token in
                       self.inverted_index}

    def index_relations(self):
        #HR ????

        self.folds_indexed_qrels = []
        self.folds_training_indexed_qrels = []

        for i in range(self.k):

            training_indexed_qrels = dict()
            training_indexed_qrels['pos'] = []
            training_indexed_qrels['neg'] = dict()
            for elem in self.folds_training_qrels[i]['pos']:
                if elem[1] in self.doc_index:
                    training_indexed_qrels['pos'].append([self.all_queries_index[elem[0]],
                                                          self.doc_index[elem[1]]])

            for key in self.folds_training_qrels[i]['neg']:
                training_indexed_qrels['neg'][key] = []
                for elem in self.folds_training_qrels[i]['neg'][key]:
                    if elem in self.doc_index:
                        training_indexed_qrels['neg'][key].append(self.doc_index[elem])

            self.folds_training_indexed_qrels.append(training_indexed_qrels)

            indexed_qrels = []
            for elem in self.folds_qrels[i]:
                if elem[1] in self.doc_index:
                    indexed_qrels.append([self.all_queries_index[elem[0]], self.doc_index[elem[1]]])

            self.folds_indexed_qrels.append(indexed_qrels)

    def compute_info_retrieval(self):
        """Function that builds the inverted index, the idf of the terms; documents length
        and frequencies of terms in the collection and indexes the relations""" #HR
        self.build_inverted_index()
        self.compute_idf()
        self.compute_docs_length()
        self.compute_collection_frequencies()
        self.index_relations()

    def save_results(self, index_queries, results, path, top_k=1000):
        with open(path, 'w') as f:
            for query, documents in enumerate(results):
                for i, scores in enumerate(documents.most_common(top_k)):
                    f.write(index_queries[query] + ' Q0 ' + self.doc_index[scores[0]] + ' ' + str(i) + ' ' + str(
                        scores[1]) + ' 0\n')

    def pickle_indexed_collection(self, path):
        """Function that writes the different indexed collection parts other than documents annd fold queries
        into a pickle format""" #HR
        self.documents = None
        self.folds_queries = None
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_fasttext_embedding(self, model_path):
        """Function that computes the embedding matrix using the fasttext embedding: vectors of length
        300 for every token in the vocabulary""" #HR
        model = fasttext.load_model(model_path)
        dim = model.get_dimension()
        vocab_size = int(len(self.vocabulary) / 2)
        self.embedding_matrix = np.zeros((vocab_size, dim))
        for _ in range(vocab_size):
            self.embedding_matrix[_] = model[self.vocabulary[_]]

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
