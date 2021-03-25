#packages
import pickle
import random
import fasttext
import numpy as np
import pandas as pd
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords
#Files used
import std_tokenizer
import utils

## Class Collection for Wikir###

class Collection:
    def __init__(self, language='english'):
        self.documents = None
        self.training_queries = None
        self.validation_queries = None
        self.test_queries = None
        self.language = language
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

    def load_collection(self, collection_path):
        """Function that loads an already processed collection and reads its csv files"""
        self.documents = pd.read_csv(collection_path + '/documents.csv', index_col='id_right', na_filter=False)
        self.training_queries = pd.read_csv(collection_path + '/training/queries.csv', index_col='id_left',
                                            na_filter=False)
        self.validation_queries = pd.read_csv(collection_path + '/validation/queries.csv', index_col='id_left',
                                              na_filter=False)
        self.test_queries = pd.read_csv(collection_path + '/test/queries.csv', index_col='id_left', na_filter=False)

        self.training_relevance = utils.read_qrels(collection_path + '/training/qrels')
        self.validation_relevance = utils.read_qrels(collection_path + '/validation/qrels')
        self.test_relevance = utils.read_qrels(collection_path + '/test/qrels')

    def save_xml(self, output_dir):
        """Function that saves the documents, the training queries ,validation queries and test queries
        generated into xml format"""
        with open(output_dir + '/documents.xml', 'w') as f:
            for key, value in self.documents.iterrows():
                f.write('<DOC>\n<DOCNO>' + str(key) + '</DOCNO>\n<TEXT>\n' + value[0] + '\n</TEXT></DOC>\n')

        with open(output_dir + '/training_queries.xml', 'w') as f:
            for key, value in self.training_queries.iterrows():
                f.write('<top>\n<num>' + str(key) + '</num><title>\n' + value[0] + '\n</title>\n</top>\n')

        with open(output_dir + '/validation_queries.xml', 'w') as f:
            for key, value in self.validation_queries.iterrows():
                f.write('<top>\n<num>' + str(key) + '</num><title>\n' + value[0] + '\n</title>\n</top>\n')

        with open(output_dir + '/test_queries.xml', 'w') as f:
            for key, value in self.test_queries.iterrows():
                f.write('<top>\n<num>' + str(key) + '</num><title>\n' + value[0] + '\n</title>\n</top>\n')

    def update_standard_vocabulary(self, sequences, remove_stopwords=True):
        """Function that updates the vocabulary on the basis of new sequences"""
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
        """Function that builds the vocabulary from documents and the different queries"""
        self.vocabulary = Counter()

        self.update_standard_vocabulary(self.documents, remove_stopwords)
        self.update_standard_vocabulary(self.training_queries, remove_stopwords)
        self.update_standard_vocabulary(self.validation_queries, remove_stopwords)
        self.update_standard_vocabulary(self.test_queries, remove_stopwords)

        self.vocabulary = {i + 1: elem[0] for i, elem in enumerate(self.vocabulary.most_common()) if elem[1] >= min_occ}

        for key in list(self.vocabulary):
            self.vocabulary[self.vocabulary[key]] = key

        self.vocabulary[0] = '<PAD>'
        self.vocabulary['<PAD>'] = 0

    def standard_preprocess(self,
                            remove_stopwords=True,
                            min_occ=5):
        """Function that preprocesses the collection by building the vocabulary and indexing the documents
        and the different queries"""
        print('Build voc', flush=True)
        self.build_standard_vocabulary(min_occ=min_occ,
                                       remove_stopwords=remove_stopwords)

        print('Index documents', flush=True)
        self.doc_index, self.indexed_docs = std_tokenizer.std_tokenizer_index(self.documents,
                                                                self.vocabulary,
                                                                self.stemmer)

        print('Index queries', flush=True)
        self.training_queries_index, self.indexed_training_queries = std_tokenizer.std_tokenizer_index(self.training_queries,
                                                                                         self.vocabulary,
                                                                                         self.stemmer)

        self.validation_queries_index, self.indexed_validation_queries = std_tokenizer.std_tokenizer_index(self.validation_queries,
                                                                                             self.vocabulary,
                                                                                             self.stemmer)

        self.test_queries_index, self.indexed_test_queries = std_tokenizer.std_tokenizer_index(self.test_queries,
                                                                                 self.vocabulary,
                                                                                 self.stemmer)

    def build_inverted_index(self):
        """Function that builds the inverted index from the vocabulary and the indexed documents"""
        self.inverted_index = dict()

        for token in self.vocabulary:
            if isinstance(token, int):
                self.inverted_index[token] = Counter()

        for i, indexed_document in enumerate(self.indexed_docs):
            for token in indexed_document:
                self.inverted_index[token][i] += np.float32(1.0)

    def compute_idf(self):
        """Function that computes the idf for every word"""
        nb_docs = len(self.doc_index)
        self.idf = {token: np.log((nb_docs + 1) / (1 + len(self.inverted_index[token]))) for token in
                    self.inverted_index}

    def compute_docs_length(self):
        """Function that computes documents length"""
        self.docs_length = {i: len(doc) for i, doc in enumerate(self.indexed_docs)}

    def compute_collection_frequencies(self):
        """Function that computes frequencies of eac word"""
        coll_length = sum([value for key, value in self.docs_length.items()])
        self.c_freq = {token: sum([freq for _, freq in self.inverted_index[token].items()]) / coll_length for token in
                       self.inverted_index}

    def index_relations(self):
        self.training_indexed_relevance = []
        for elem in self.training_relevance:
            self.training_indexed_relevance.append([self.training_queries_index[elem[0]], self.doc_index[elem[1]]])

        self.validation_indexed_relevance = []
        for elem in self.validation_relevance:
            self.validation_indexed_relevance.append([self.validation_queries_index[elem[0]], self.doc_index[elem[1]]])

        self.test_indexed_relevance = []
        for elem in self.test_relevance:
            self.test_indexed_relevance.append([self.test_queries_index[elem[0]], self.doc_index[elem[1]]])

    def compute_info_retrieval(self):
        """Function that builds inverted index , idf , document length, collection frequencies and indexed relations"""
        self.build_inverted_index()
        self.compute_idf()
        self.compute_docs_length()
        self.compute_collection_frequencies()
        self.index_relations()

    def save_results(self, index_queries, results, path, top_k=1000):
        """Function that saves the top 1000 results according to their score"""
        with open(path, 'w') as f:
            for query, documents in enumerate(results):
                for i, scores in enumerate(documents.most_common(top_k)):
                    f.write(index_queries[query] + ' Q0 ' + self.doc_index[scores[0]] + ' ' + str(i) + ' ' + str(
                        scores[1]) + ' 0\n')

    def pickle_indexed_collection(self, path):
        """Function that saves th computed self elements into a pickle file"""
        self.documents = None
        self.training_queries = None
        self.validation_queries = None
        self.test_queries = None
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_fasttext_embedding(self, model_path):
        """Function that computes the fasttext embedding : vectos of 300 dimension"""
        model = fasttext.load_model(model_path)
        dim = model.get_dimension()
        vocab_size = int(len(self.vocabulary) / 2)
        self.embedding_matrix = np.zeros((vocab_size, dim))
        for _ in range(vocab_size):
            self.embedding_matrix[_] = model[self.vocabulary[_]]

    def generate_training_batches(self, batch_size=64):
        "Function that generates training batches"
        random.shuffle(self.training_indexed_relevance)
        nb_docs = len(self.indexed_docs)
        nb_train_pairs = len(self.training_indexed_relevance)
        query_batches = []
        positive_doc_batches = []
        negative_doc_batches = []
        pos = 0
        while (pos + batch_size < nb_train_pairs):
            query_batches.append([q for q, d in self.training_indexed_relevance[pos:pos + batch_size]])
            positive_doc_batches.append([d for q, d in self.training_indexed_relevance[pos:pos + batch_size]])
            negative_doc_batches.append([random.randint(0, nb_docs - 1) for _ in range(len(positive_doc_batches[-1]))])
            pos += batch_size
        return query_batches, positive_doc_batches, negative_doc_batches

    def generate_test_batches(self, batch_size=64):
        """Function that generates test batch"""
        random.shuffle(self.test_indexed_relevance)
        nb_docs = len(self.indexed_docs)
        nb_test_pairs = len(self.test_indexed_relevance)
        query_batches = []
        positive_doc_batches = []
        negative_doc_batches = []
        pos = 0
        while (pos + batch_size < nb_test_pairs):
            query_batches.append([q for q, d in self.test_indexed_relevance[pos:pos + batch_size]])
            positive_doc_batches.append([d for q, d in self.test_indexed_relevance[pos:pos + batch_size]])
            negative_doc_batches.append([random.randint(0, nb_docs - 1) for _ in range(len(positive_doc_batches[-1]))])
            pos += batch_size
        return query_batches, positive_doc_batches, negative_doc_batches
