#packages
import pickle
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords
import array as arr
import numpy as np
import matplotlib.pyplot as plt
#Defintion of Inverted structure class
class Inverted_structure:
    def __init__(self):
        # Extern documet idf, the position in this list is internal doc Id
        self.document_IDs=[]
        # All tocken posting list description : [posting_long,tocken_id]
        self.vocabulary=dict()
        # A list of posting list for each tocken internal id as an array
        self.posting_lists=[]
        # Size of each document by doc internal Id
        self.documents_length=arr.array('I')
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))

    def inverse_document(self,document_ID,document_text):
        """Function that updates posting lists and vocabulary from a document text and add the document ID to the list of document IDs"""
        self.document_IDs.append(document_ID)
        internal_doc_ID=len(self.document_IDs)-1
        internal_token_ID=len(self.vocabulary)
        tmp_dict_freq=Counter()
        #Preprocessing words and fillinf the temporary dictionary
        for elem in document_text.split(" "):
            word=elem.lower()
            if word not in self.stop_words:
                token=self.stemmer.stem(word)
                tmp_dict_freq[token]+=1
        #Computing the document length
        doc_length=sum(tmp_dict_freq.values())
        self.documents_length.append(doc_length)
        #Creating or updating the vocabulary and the posting lists
        for token,frequency in tmp_dict_freq.items():
            if token not in self.vocabulary:
                #Updating vocabulary with the new token [length of posting list=1,position in the posting file=internal token ID]
                self.vocabulary[token]=[1,internal_token_ID]
                #Creating an array of type unsigned int contraining a single input since the word is not in the vocabulary
                posting_list=arr.array('I', [internal_doc_ID,frequency])
                #Appending the new posting list
                self.posting_lists.append(posting_list)
                #Incrementing to get the next position
                internal_token_ID+=1
            else:
                #Updating length of posting list corresponding to the token
                self.vocabulary[token][0]+=1
                #Getting the position of the posting list
                pos=self.vocabulary[token][1]
                #Extending the posting list to include the document ID and the frequency in the document
                self.posting_lists[pos].extend([internal_doc_ID,frequency])
        del(tmp_dict_freq)

    def save(self,file_path):
        """A method that saves the posting file, the vocabulary and the document IDs"""
        #Writing the posting file
        with open(file_path+'/posting_file','wb') as f:
            for posting_list in self.posting_lists:
                posting_list.tofile(f)

        #Saving the vocabulary and the document IDS
        with open(file_path+'/vocabulary', 'wb') as f:
            pickle.dump(self.vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(file_path+'/document_IDs', 'wb') as f:
            pickle.dump(self.document_IDs, f, protocol=pickle.HIGHEST_PROTOCOL)

        #Saving the documents length
        with open(file_path+'/documents_length','wb') as f:
            self.documents_length.tofile(f)

    def load(self,file_path):
        """Function that loads the posting lists , vocabulary and document IDs"""
        #Initializing the objects to contain the posting lists,vocabulary and document IDs
        self.vocabulary=dict()
        self.document_IDs=[]
        self.posting_lists=[]
        self.documents_length=arr.array('I')
        #Loading the vocabulary
        with open(file_path+'/vocabulary', 'rb') as f:
            self.vocabulary=pickle.load(f)

        #Loading the document IDS
        with open(file_path+'/document_IDs', 'rb') as f:
            self.document_IDs=pickle.load(f)

        #Loading the posting lists
        with open(file_path+'/posting_file', 'rb') as f:
            #Going through the vocabulary in order of position to get the length of each posting lists and get the
            # posting list in the posting file in the order they were written in the file
            #value is [length of posting list,position]
            for token,value in sorted(self.vocabulary.items(),key=lambda t:t[1][1]):
                posting_list=arr.array('I')
                length_of_posting_list=value[0]
                posting_list.fromfile(f,2*length_of_posting_list)
                self.posting_lists.append(posting_list)

        #Loading documents length
        with open(file_path+'/documents_length', 'rb') as f:
            #We use frombytes because it does not require to enter the number of elements you want to retrieve. fromfile does.
            self.documents_length.frombytes(f.read())

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def token(self):
        """
        Acces to all tockens as a generator
        """
        for key in self.vocabulary.keys():
            yield key

    def existsToken(token):
        """
        Fast access to test if a key exists
        Do to use the token() générator for that
        """
        return tocken in self.vocabulary

    def get_number_of_documents(self):
        return len(self.document_IDs)

    def get_posting_list(self,token):
        """Function that returns a list of tuples (document_intenal_ID,frequency) from the list of posting lists"""
        try:
            length_of_posting_list,internal_token_ID=self.vocabulary[token]
        except KeyError:
            print( token +" is not present in the vocabulary . No posting list found")
        except:
            print("Unkown error")
        posting_list=self.posting_lists[internal_token_ID]

        return [(posting_list[2*i],posting_list[2*i+1]) for i in range(length_of_posting_list)]

    def posting_list(self,token):
        """
        list of tuples (document_intenal_ID,frequency) from the list of posting lists
        Similar to get_posting_list but expressed as a generator
        """
        try:
            length_of_posting_list,internal_token_ID=self.vocabulary[token]
        except KeyError:
            print( token +" is not present in the vocabulary . No posting list found")
        except:
            print("Unkown error")
        posting_list=self.posting_lists[internal_token_ID]
        i = 0;
        while (i < length_of_posting_list) :
            yield (posting_list[2*i],posting_list[2*i+1])
            i += 1


    def get_external_ID_document(self,internal_document_ID):
        """Function that gets external document ID from the internal document ID"""
        try:
            external_document_ID=self.document_IDs[internal_document_ID]
        except ValueError:
            print( "document not present in the list")
        except:
            print("Unknown Error")
        return external_document_ID

    def get_internal_document_ID(self, external_document_ID):
        """Function that gets the internal_document_ID from the external_document_ID"""
        internal_document_ID=None
        for i in range(len(self.document_IDs)):
             if external_document_ID==self.document_IDs[i]:
                internal_document_ID=i
        if internal_document_ID==None:
            raise ValueError
        else:
            return internal_document_ID

    def compute_idf(self):
        """Function that computes idf for every word in the vocabulary"""
        number_of_documents=len(self.document_IDs)
        #self.vocabulary[token][0] is the length of the posting list. Remember a token in vocabulary has a value [length of posting list,position of posting list or internal token ID]
        self.idf={token: np.log((number_of_documents + 1) / (1 + self.vocabulary[token][0])) for token in
                    self.vocabulary.keys()}

    def compute_collection_frequencies(self):
        """Function that computes frequencies of each word in the vocabulary"""
        self.c_freq={}
        #Calculating the total number of tokens in the collection
        coll_length = sum(self.documents_length)
        #The frequency of a word is the number of occurences of the token in the documents divided by the total number ot tokens in the collection
        for token,value in self.vocabulary.items():
            internal_token_ID=value[1]
            length_posting_list=value[0]
            posting_list=self.posting_lists[internal_token_ID]
            self.c_freq[token]=0
            for i in range(length_posting_list):
                self.c_freq[token]+=posting_list[2*i+1]/coll_length
    def statistics_about_the_structure(self,path_save_plot=None,save_plot=True):
        """Function that computes statistics abour the inverted structure """
        vocab_size=self.get_vocabulary_size()
        print("The vocabulary has ", vocab_size, "tokens",flush=True)
        min_pos_len=99999999999
        max_pos_len=0
        sum_pos_len=0
        list_pos_len=[]
        count=0
        for key,value in self.vocabulary.items():
            length_of_posting_list=value[0]
            list_pos_len.append(length_of_posting_list)
            sum_pos_len+=length_of_posting_list
            if length_of_posting_list>max_pos_len:
                max_pos_len=length_of_posting_list
                max_token=key
            if length_of_posting_list<min_pos_len:
                min_pos_len=length_of_posting_list
            if length_of_posting_list==1 and count <100:
                print("Word = ", key," has post len 1",flush=True)
        print("minimum length of posting list is = ",min_pos_len,flush=True)
        print("maximum length of posting list is = ",max_pos_len,flush=True)
        print("average length of posting list is = ",sum_pos_len/vocab_size,flush=True)
        print("The most used word is = ", max_token,flush=True)
#         plt.hist(list_pos_len,50,density=True,facecolor='g')
#         plt.ylabel=('Number of posting lists')
#         plt.title('Histogram of posting lists length')
#         plt.xlim(min_pos_len,max_pos_len)
#         plt.ylim(0, 0.03)
#         plt.grid(True)
#         plt.show()
#         plt.style.use('ggplot')
        plt.hist(list_pos_len,range=(min_pos_len,100), bins=100,color='yellow',edgecolor='red')
        plt.show()
        if save_plot and path_save_plot!=None:
            plt.savefig(path_save_plot+'/Histogram_of_posting_lists_length.png')