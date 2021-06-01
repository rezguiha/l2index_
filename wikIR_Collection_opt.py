##packages
from nltk.stem import snowball
from nltk.corpus import stopwords
import os
import pandas as pd
import time
#Files
import utils
from Queries import Queries
from Inverted_structure import Inverted_structure
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
        self.inverted_structure=Inverted_structure()
    def load_documents(self, collection_path):
        """Function that loads an already processed collection and reads its csv documents"""
        self.documents = pd.read_csv(collection_path + '/documents.csv', index_col='id_right', na_filter=False)
    def load_queries(self, collection_path):
        """Function that loads an already processed collection and reads its csv queries"""
        self.training_queries = pd.read_csv(collection_path + '/training/queries.csv', index_col='id_left',
                                            na_filter=False)
        self.validation_queries = pd.read_csv(collection_path + '/validation/queries.csv', index_col='id_left',
                                              na_filter=False)
        self.test_queries = pd.read_csv(collection_path + '/test/queries.csv', index_col='id_left', na_filter=False)

    def load_relevance(self, collection_path):
        """Function that loads an already processed collection and reads its  relevance"""
        self.training_relevance = utils.read_qrels(collection_path + '/training/qrels')
        self.validation_relevance = utils.read_qrels(collection_path + '/validation/qrels')
        self.test_relevance = utils.read_qrels(collection_path + '/test/qrels')

    def build_inverted_index_and_vocabulary(self, file_path,minimum_occurence=5,proportion_of_frequent_words=0.2):
        """Function that builds the inverted index ,the vocabulary ,posting lists,documents_length and direct structure and filters them"""
        inverted_structure = Inverted_structure()
        start=time.time()
        # Iterating over the documents and building the inverted and the direct structure
        for document_ID, document_text in self.documents.iterrows():
            inverted_structure.inverse_document(str(document_ID), document_text[0])
        end=time.time()
        number_of_documents=inverted_structure.get_number_of_documents()
        print("Total time to inverse documents wikIR",round(end-start), " s",flush=True)
        print("Average time to inverse documents wikIR",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Filtering vocabulary , posting lists, the direct structure and document lengths
        if os.path.exists(file_path):
            start=time.time()
            inverted_structure.filter_vocabulary(file_path,minimum_occurence,proportion_of_frequent_words)  
            end=time.time()
        else:
            raise IOError('Path does not exist: %s' % file_path)
        print("Total time to filter vocabulary,posting lists,direct_structure and update document lengths wikIR",round(end-start), " s",flush=True)
        print("Average time to filter vocabulary,posting lists,direct_structure and update document lengths wikIR",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Saving      
        start=time.time()
        inverted_structure.save(file_path)
        end=time.time()
        print("Saving time inverted structure wikIR",round(end-start), " s",flush=True)

        return inverted_structure
    def load_inverted_structure(self,file_path):
        self.inverted_structure=Inverted_structure()
        self.inverted_structure.load(file_path)
    def load_directed_structure(self,file_path):
        self.direct_structure=Directed_structure()
        self.direct_structure.load(file_path)
    def process_queries(self,vocabulary, file_path=None, save=False):
        # Processing training queries
        self.training_queries = Queries(vocabulary)
        for query_ID, query_text in self.training_queries.iterrows():
            self.training_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                self.training_queries.save(file_path, "training")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        # Processing validation queries
        self.validation_queries = Queries(vocabulary)
        for query_ID, query_text in self.validation_queries.iterrows():
            self.validation_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                self.validation_queries.save(file_path, "validation")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        # Processing test queries
        self.test_queries = Queries(vocabulary)
        for query_ID, query_text in self.test_queries.iterrows():
            self.test_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                self.test_queries.save(file_path, "test")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return self.training_queries, self.validation_queries, self.test_queries
    
    
    def generate_training_batches(self, batch_size=64):
        "Generator Function that generates training batches"
        random.shuffle(self.training_relevance)
        nb_docs = len(self.direct_structure.documents)
        nb_train_pairs = len(self.training_relevance)
        query_batches = arr.array('I')
        positive_doc_batches = arr.array('I')
        negative_doc_batches = arr.array('I')
        pos = 0
        while (pos + batch_size < nb_train_pairs):
            for q,d in self.training_relevance[pos:pos + batch_size]:
                query_batch.append(self.training_queries.queries_internal_IDs[str(q)])
                positive_doc_batch.append(self.inverted_structure.document_internal_IDs[str(d)])
                negative_doc_batch.append(random.randint(0, nb_docs - 1))
            yield query_batch, positive_doc_batch, negative_doc_batch
            pos += batch_size
            

    def generate_test_batches(self, batch_size=64):
        """Generator Function that generates test batch"""
        random.shuffle(self.test_relevance)
        nb_docs = len(self.direct_structure.documents)
        nb_train_pairs = len(self.training_relevance)
        query_batches = arr.array('I')
        positive_doc_batches = arr.array('I')
        negative_doc_batches = arr.array('I')
        pos = 0
        while (pos + batch_size < nb_train_pairs):
            for q,d in self.test_relevance[pos:pos + batch_size]:
                query_batch.append(self.training_queries.queries_internal_IDs[str(q)])
                positive_doc_batch.append(self.inverted_structure.document_internal_IDs[str(d)])
                negative_doc_batch.append(random.randint(0, nb_docs - 1))
            yield query_batch, positive_doc_batch, negative_doc_batch
            pos += batch_size
            
