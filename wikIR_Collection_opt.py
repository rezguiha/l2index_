##packages
from nltk.stem import snowball
from nltk.corpus import stopwords
import os
import pandas as pd
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

    def build_inverted_index_and_vocabulary(self, file_path=None, save=True):
        """Function that builds the inverted index and  the vocabulary"""
        inverted_structure = Inverted_structure()
        # Iterating over the documents
        for document_ID, document_text in self.documents.iterrows():
            inverted_structure.inverse_document(document_ID, document_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                inverted_structure.save(file_path)
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return inverted_structure

    def process_queries(self, file_path=None, save=False):
        # Processing training queries
        training_queries = Queries()
        for query_ID, query_text in self.training_queries.iterrows():
            training_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                training_queries.save(file_path, "training")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        # Processing validation queries
        validation_queries = Queries()
        for query_ID, query_text in self.validation_queries.iterrows():
            validation_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                validation_queries.save(file_path, "validation")
            else:
                raise IOError('Path does not exist: %s' % file_path)
                # Processing test queries
        test_queries = Queries()
        for query_ID, query_text in self.validation_queries.iterrows():
            test_queries.process_query_and_get_ID(query_ID, query_text[0])
        if save and file_path != None:
            if os.path.exists(file_path):
                test_queries.save(file_path, "test")
            else:
                raise IOError('Path does not exist: %s' % file_path)
        return training_queries, validation_queries, test_queries