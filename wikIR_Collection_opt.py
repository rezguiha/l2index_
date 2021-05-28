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

    def build_inverted_index_and_vocabulary(self, file_path=None, save=True,minimum_occurence=5,proportion_of_frequent_words=0.2):
        """Function that builds the inverted index ,the vocabulary ,posting lists,documents_length and direct structure and filters them"""
        inverted_structure = Inverted_structure()
        start=time.time()
        # Iterating over the documents and building the inverted and the direct structure
        for document_ID, document_text in self.documents.iterrows():
            inverted_structure.inverse_document(document_ID, document_text[0])
        end=time.time()
        number_of_documents=inverted_structure.get_number_of_documents()
        print("Total time to inverse documents wikIR",round(end-start), " s",flush=True)
        print("Average time to inverse documents wikIR",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Filtering vocabulary , posting lists, the direct structure and document lengths
        start=time.time()
        inverted_structure.filter_vocabulary(minimum_occurence,proportion_of_frequent_words)  
        end=time.time()
        print("Total time to filter vocabulary,posting lists,direct_structure and update document lengths wikIR",round(end-start), " s",flush=True)
        print("Average time to filter vocabulary,posting lists,direct_structure and update document lengths wikIR",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Saving      
        if save and file_path != None:
            if os.path.exists(file_path):
                start=time.time()
                inverted_structure.save(file_path)
                end=time.time()
                print("Saving time wikIR",round(end-start), " s",flush=True)
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return inverted_structure
    def load_inverted_structure(self,file_path):
        self.inverted_structure=Inverted_structure
        self.inverted_structure.load(file_path)
    def load_directed_structure(self,file_path):
        self.direct_structure=Directed_structure()
        self.direct_structure.load(file_path)
    def process_queries(self, file_path=None, save=False,vocabulary):
        # Processing training queries
        training_queries = Queries(vocabulary)
        for query_ID, query_text in self.training_queries.iterrows():
            training_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                training_queries.save(file_path, "training")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        # Processing validation queries
        validation_queries = Queries(vocabulary)
        for query_ID, query_text in self.validation_queries.iterrows():
            validation_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                validation_queries.save(file_path, "validation")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        # Processing test queries
        test_queries = Queries(vocabulary)
        for query_ID, query_text in self.test_queries.iterrows():
            test_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                test_queries.save(file_path, "test")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return training_queries, validation_queries, test_queries