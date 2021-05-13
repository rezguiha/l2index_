#packages
import pickle
from nltk.stem import snowball
from nltk.corpus import stopwords

### Definition of class queries
class Queries:
    def __init__(self):
        # External queries id, the position in this list is internal doc Id
        self.queries_IDs=[]
        # A list containing the processed queries which are a list of tokens
        self.processed_queries=[]
        # Stemmer and stop words
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))
    def process_query_and_get_ID(self,query_external_ID,non_processed_query_text):
        """Function that adds the external query ID to the list and processes the query and adds it to the processed queries list"""
        #Adding the exteral query ID
        self.queries_IDs.append(query_external_ID)
        query=[]
        #Processing the query
        for elem in non_processed_query_text.split(" "):
            word=elem.lower()
            if word not in self.stop_words:
                token=self.stemmer.stem(word)
                query.append(token)
        self.processed_queries.append(query)
    def get_number_of_queries(self):
        return len(self.queries_IDs)
    def get_external_ID_of_query(self,internal_query_ID):
        """Function that gets external query ID from the internal query ID"""
        try:
            external_query_ID=self.queries_IDs[internal_query_ID]
        except ValueError:
            print( "token not present in the list")
        except:
            print("Unknown Error")
        return external_query_ID
    def get_internal_ID_of_query(self,external_query_ID):
        """Function that gets the internal_query_ID from the external_query_ID"""
        internal_document_ID=None
        for i in range(len(self.queries_IDs)):
             if external_query_ID==self.queries_IDs[i]:
                internal_document_ID=i
        if internal_document_ID==None:
            raise ValueError
        else:
            return internal_document_ID
    def query(self):
        """Access all processed_queries as a generator """
        for query in self.processed_queries:
            yield query
    def save(self,file_path,name_of_file):
        """Saving External queries ID and processed queries"""
        #File path is the path to where to save the objects .name of files contains the name of the file for example (training, validation ...)
        #Saving external queries ID
        with open(file_path+'/'+name_of_file+'_queries_IDs', 'wb') as f:
            pickle.dump(self.queries_IDs, f, protocol=pickle.HIGHEST_PROTOCOL)
        #Saving processed queries
        with open(file_path+'/'+name_of_file+'_queries', 'wb') as f:
            pickle.dump(self.processed_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,file_path,name_of_file):
        """ Loading queries IDs and processed queries"""
        self.queries_IDs=[]
        self.processed_queries=[]
        with open(file_path+'/'+name_of_file+'_queries_IDs', 'rb') as f:
            self.queries_IDs=pickle.load(f)
        with open(file_path+'/'+name_of_file+'_queries', 'rb') as f:
            self.processed_queries=pickle.load(f)