#packages
import pickle

from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords
import array as arr

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
        

    
    def build_inverted_index_and_vocabulary(self):
        """Function that builds the inverted index and  the vocabulary"""
        document_IDs=[]
        vocabulary=dict()
        posting_lists=[]
        internal_doc_ID=0
        position=0
        #Iterating over the documents
        for key,element in self.documents.iterrows():
            #Keeping the document ID
            document_IDs.append(key)
            #Creating temporary dictionary for frequency calculation of terms in the document 
            tmp_dict_freq=Counter()
           
            for elem in element[0].split(" "):
                word=elem.lower()
                if word not in self.stop_words:
                    token=stemmer.stem(word)
                    tmp_dict_freq[token]+=1
                    if token not in vocabulary.keys():
                        position+=1
                        #Updating vocabulary for each token we have [length of posting list,position in the posting file]
                        vocabulary[token]=[1,position]
                        #Creating an empty arra of type unsigned int
                        posting_list=arr.array('I', [internal_doc_ID,1])
                        #Appending the new posting list 
                        posting_lists.append(posting_list)
            for key,value in tmp_dict_freq.items():
                #Updating length of posting list corresponding to the token
                vocabulary[token][0]+=1
                #Getting the position of the posting list
                pos=vocabulary[token][1]
                #Extending the posting list to include the document ID and the frequency in the document
                posting_lists[pos-1].extend([internal_doc_ID,value])               
            interal_doc_ID+=1
        #Writing the posting file
        
           
        
        
        
        

