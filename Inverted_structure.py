#packages
import pickle
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords
import array as arr
import numpy as np
import matplotlib.pyplot as plt
import fasttext
import resource
import time
#Files
from Direct_structure import Direct_structure
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
        self.direct_structure = Direct_structure()

    def getTokenId(self,token):
        """
        From a string tocken retuen its id
        If token is unknow, add it to the vocabulary
        Warning: This method creates an empty posting list and adds it to the list of the posting lists too.
        """
        if token not in self.vocabulary:
            internal_token_ID=len(self.vocabulary)
            #updating the vocabulary with the new token.[length of posting list=0,position in the posting file=internal token ID]
            self.vocabulary[token]=[0,internal_token_ID]
            #creating an empty posting list to be filled later
            posting_list=arr.array('I')
            self.posting_lists.append(posting_list)
            return internal_token_ID
        else:
            return self.vocabulary[token][1]


    def inverse_document(self,document_ID,document_text):
        """Function that updates posting lists and vocabulary from a document text,builds a processed document to add to the direct structure and add the document ID to the list of document IDs and document length to the document length's array """
        self.document_IDs.append(document_ID)
        internal_doc_ID=len(self.document_IDs)-1
        internal_token_ID=len(self.vocabulary)
        tmp_dict_freq=Counter()
        processed_document=arr.array('I')
        
        #Preprocessing words, filling up the processed document and adding it to the direct structure, and fills the temporary dictionary to use later for updating the posting lists
        for elem in document_text.split(" "):
            word=elem.lower()
            if word not in self.stop_words:
                token=self.stemmer.stem(word)
                #building the processed document for the document structure
                token_id=self.getTokenId(token)
                processed_document.append(token_id)
                tmp_dict_freq[token]+=1
        self.direct_structure.add_document(processed_document)
        #Computing the document length and adding it to the array of documents' length
        doc_length=sum(tmp_dict_freq.values())
        self.documents_length.append(doc_length)
        #Creating or updating the vocabulary and the posting lists
        for token,frequency in tmp_dict_freq.items():
                #Updating length of posting list corresponding to the token
                self.vocabulary[token][0]+=1
                #Getting the position of the posting list which is the internal token id
                pos=self.vocabulary[token][1]
                #Extending the posting list to include the document ID and the frequency in the document
                self.posting_lists[pos].extend([internal_doc_ID,frequency])
        del(tmp_dict_freq)

    def filter_vocabulary(self,minimum_occurence=5,proportion_of_frequent_words=0.2):
        """Function that filters tokens from vocabulary and posting lists that have an occurence less than minimum_occurence or that are present in more than proportion_of_frequent_words of documents. These tokens will likely be not very useful for the retrieval objective"""
        number_of_documents=self.get_number_of_documents()
        indices_of_posting_lists_to_delete=arr.array('I')
        keys_to_delete=[]
        index_token=0
        print("Memory usage start filter_inverted_structure", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        #Iterating using a generator of tokens
        for token in self.token():
            length_of_posting_list=self.vocabulary[token][0]
            posting_list=self.posting_lists[self.vocabulary[token][1]]

            #Calculating the frequency of the token in the collection and we stop when we exceed the minimum number of occurence
            # If the result is less than the minimum occurence it means that we need to delete that element if not we keep it
            sum_frequency_token=0
            index_post=0
            while (sum_frequency_token<minimum_occurence and index_post<length_of_posting_list):
                sum_frequency_token+=posting_list[2*index_post+1]
                index_post+=1
            #Storing indices of posting lists and tokens to erase from the vocabulary and the posting lists
            if (sum_frequency_token <minimum_occurence) or (length_of_posting_list>proportion_of_frequent_words*number_of_documents):
                #Storing token=key to delete from vocabulary from the vocabulary
                keys_to_delete.append(token)
                #Storing indices of posting lists to delete. We can't delete them one by one because the indices will change when we do that
                indices_of_posting_lists_to_delete.append(index_token)
            index_token+=1
        print("Memory usage after filling indices and tokens to delete", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

 
        
        #It is important to know that since Python 3.7 dictionaries are an ordered structure by insertion order
        #We're filtering the vocabulary and updating the internal token ID in the vocabulary for the whole tokens after filtering

        for token in keys_to_delete:
            del self.vocabulary[token]
        #Iterating using a generator for yielding token from vocabulary
        internal_token_ID=0
        previous_internal_token_ID=arr.array('I')
        for token in self.token():
            previous_internal_token_ID.append(self.vocabulary[token][1])
            self.vocabulary[token][1]=internal_token_ID
            internal_token_ID+=1
        print("Memory usage after deleting tokens from vocab and updating internal token ID", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        del(keys_to_delete)
        
      
        #Going over the processed documents and filtering the tokens and updating the tokens' new internal token ID and updating the documents' length
        print("Size of processed documents ", len(self.direct_structure.processed_documents),flush=True)
        start=time.time()
        processed_document_index_gen=(i for i in range(self.get_number_of_documents()))
        total_number_of_elements_deleted=0
        position=0
        for i in processed_document_index_gen:     
            document_number_of_elements_deleted=0
            for j in range(self.documents_length[i]): 
                if self.direct_structure.processed_documents[j+position-total_number_of_elements_deleted] not in previous_internal_token_ID:
                    self.direct_structure.processed_documents.pop(j+position-total_number_of_elements_deleted)
                    total_number_of_elements_deleted+=1
                    document_number_of_elements_deleted+=1
                else:
                    self.direct_structure.processed_documents[j+position-total_number_of_elements_deleted]=previous_internal_token_ID.index(self.direct_structure.processed_documents[j+position-total_number_of_elements_deleted])
            position+=self.documents_length[i]
            self.documents_length[i]-=document_number_of_elements_deleted
        end=time.time()
        print("Time to filter direct structure =",round(end-start),flush=True)
                    
        print("Memory usage after filtering tokens from processed documents and updating documents's length", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        
        #The indices of posting lists to delete are in an ascending order. But we need to go through it in reverse. So that, deleting elements doesn't modify the indices of the list. To do that we reverse the list of indices to delete using reverse iteration with the reversed() built-in . It neither reverses a list in-place, nor does it create a full copy. Instead we get a reverse iterator we can use to cycle through the elements of the list in reverse order
        index_to_delete_gen=(i for i in reversed(indices_of_posting_lists_to_delete))
        for i in index_to_delete_gen:
            del self.posting_lists[i]

        print("Memory usage after deleting posting lists", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
        del(indices_of_posting_lists_to_delete)
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
        #Saving directed structure
        self.direct_structure.saving_all_documents(file_path)   
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
            #It is important to know that since Python 3.7 dictionaries are an ordered structure by insertion order
            for token in self.vocabulary:
                posting_list=arr.array('I')
                length_of_posting_list=self.vocabulary[token][0]
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
        for key in self.vocabulary:
            yield key

    def existsToken(self,token):
        """
        Fast access to test if a key exists
        Do to use the token() générator for that
        """
        return token in self.vocabulary

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
                    self.vocabulary}

    def compute_collection_frequencies(self):
        """Function that computes frequencies of each word in the vocabulary"""
        self.c_freq={}
        #Calculating the total number of tokens in the collection
        coll_length = sum(self.documents_length)
        #The frequency of a word is the number of occurences of the token in the documents divided by the total number ot tokens in the collection
        for token in self.vocabulary:
            internal_token_ID=self.vocabulary[token][1]
            length_posting_list=self.vocabulary[token][0]
            posting_list=self.posting_lists[internal_token_ID]
            self.c_freq[token]=0
            for i in range(length_posting_list):
                self.c_freq[token]+=posting_list[2*i+1]/coll_length

    def compute_and_save_fasttext_embeddings(self,model_path,save_file_path):
        """Function that computes and saves the fasttext embeddings of every token in the vocabulary : vectos of 300 dimension"""
        model = fasttext.load_model(model_path)
        with open(save_file_path+'/fasttext_embeddings','wb') as f:
            for token in self.vocabulary:
                array_embeddings=arr.array('f',model[token])
                array_embeddings.tofile(f)

    def load_fasttext_embeddings(self,file_path):
        """Function that loads fasttext embeddings vectors of tokens in the vocabulary."""
        vocab_size = self.get_vocabulary_size()
        self.embedding_matrix = np.zeros((vocab_size, 300),dtype=np.float32)
        with open(file_path+'/fasttext_embeddings', 'rb') as f:
            index=0
            for token in self.vocabulary:
                token_embedding=arr.array('f')
                token_embedding.fromfile(f,300)
                self.embedding_matrix[index] = token_embedding
                index+=1
    def statistics_about_the_structure(self,path_save_plot=None,save_plot=True):
        """Function that computes statistics abour the inverted structure """
        vocab_size=self.get_vocabulary_size()
        print("The vocabulary has ", vocab_size, "tokens",flush=True)
        number_docs=self.get_number_of_documents()
        print("The Collection has ", number_docs, "documents",flush=True)
        min_pos_len=99999999999
        max_pos_len=0
        sum_pos_len=0
        list_pos_len=[]
#         count_high_threshold=0
#         sum_high_threshold=0
#         sum_low_threshold=0
#         count_low_threshold=0
#         high_threshold_list=[]
        for token in self.vocabulary:
            length_of_posting_list=self.vocabulary[token][0]
            list_pos_len.append(length_of_posting_list)
            sum_pos_len+=length_of_posting_list
            if length_of_posting_list>max_pos_len:
                max_pos_len=length_of_posting_list
                max_token=token
            if length_of_posting_list<min_pos_len:
                min_pos_len=length_of_posting_list
#             if length_of_posting_list>(0.2*number_docs):
#                 count_high_threshold+=1
#                 sum_high_threshold+=length_of_posting_list
#                 high_threshold_list.append(token)
#             if length_of_posting_list<5:
#                 count_low_threshold+=1
#                 sum_low_threshold+=length_of_posting_list
        print("minimum length of posting list is = ",min_pos_len,flush=True)
        print("maximum length of posting list is = ",max_pos_len,flush=True)
        print("average length of posting list is = ",sum_pos_len/vocab_size,flush=True)
        print("The most used word is = ", max_token,flush=True)
#         print("The number of tokens that appear in more than 20% of documents is = ", count_high_threshold ,flush=True)
#         print("The number of times to modify document lengths after erasing tokens that appear in more than 20% of documents is = ", sum_high_threshold ,flush=True)
#         print("The number of tokens that appear in less than 5 documents is = ", count_low_threshold ,flush=True)
#         print("The number of times to modify document lengths after erasing tokens that appear in more than 20% of documents is = ", sum_low_threshold ,flush=True)

#         for token in high_threshold_list:
#             print(">20% of documents= ",token,flush=True)

        plt.hist(list_pos_len,range=(min_pos_len,100), bins=100,color='yellow',edgecolor='red')
        plt.show()
        if save_plot and path_save_plot!=None:
            plt.savefig(path_save_plot+'/Histogram_of_posting_lists_length.png')
