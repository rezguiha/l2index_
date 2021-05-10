import argparse
import os
import time
import pickle
import numpy
import struct
import array

from nltk.corpus import words
from random import sample
from random import randint
from wikIR_Collection_opt import Inverted_structure

def are_equal_doc(inverse_structure1,inverse_structure2):
    #Equality of Document IDs
    if len(inverse_structure1.document_IDs)!= len(inverse_structure1.document_IDs):
        print("Document_Ids size does not match",flush=True)
        return False
    else:
        for i in range(len(inverse_structure1.document_IDs)):
            if inverse_structure1.document_IDs[i]!=inverse_structure2.document_IDs[i]:
                print( " mismatch at position ", i, " in Doc_IDs",flush=True)
                return False
    return True
def are_equal_voc(inverse_structure1,inverse_structure2):
    #Equaliry of vocabulary
    if len(inverse_structure1.vocabulary)!= len(inverse_structure1.vocabulary):
        print("Vocabulary size does not match",flush=True)
        return False
    else:
        keys_voc_2=list(inverse_structure2.vocabulary.keys())
        for key,value in inverse_structure1.vocabulary.items():
            if (key not in keys_voc_2):
                print(key , " not present in second vocabulary",flush=True) 
                return False
            else:
                value2=inverse_structure2.vocabulary[key]
                if value2[0]!=value[0]:
                    print( "length of posting list does not correspond for ", key,flush=True)
                    return False
                if value2[1]!=value[1]:
                    print("position of " , key, " does not correspond",flush=True)
                    return False
    return True
def are_equal_post(inverse_structure1,inverse_structure2):
    #Equality of posting lists
    if len(inverse_structure1.posting_lists)!= len(inverse_structure1.posting_lists):
        print("Size of posting lists does not match",flush=True)
        return False
    else:
        for i in range (len(inverse_structure1.posting_lists)):
            if len(inverse_structure1.posting_lists[i])!=len(inverse_structure2.posting_lists[i]):
                print("Size of posting list ", i, " does not match",flush=True)
                return False
            else:
                for j in range(len(inverse_structure1.posting_lists[i])):
                    if inverse_structure1.posting_lists[i][j] !=inverse_structure2.posting_lists[i][j]:
                        print("posting list ",i , " does not match at position ",j,flush=True)
                        return False
    return True
def are_equal_doc_length(inverse_structure1,inverse_structure2):
    #Equality of documents length
    if len(inverse_structure1.documents_length)!= len(inverse_structure2.documents_length):
        print("Size of documents_length of the two inverse structures does not match",flush=True)
        return False
    else:
        for i in range(len(inverse_structure1.documents_length)):
            if inverse_structure1.documents_length[i]!=inverse_structure2.documents_length[i]:
                print("document ", i," length does not match in the two structures",flush=True)
                return False
    return True
def test_inverse_document(file_path):
    """ Test inverse_document_method """
    #Building inverted structure
    print("Test inverse_document",flush=True)
    start=time.time()
    
    inverted_structure=Inverted_structure()
    document_ID='1125'
    document_text="The boy is playing football in the street boy boy street boy"

    inverted_structure.inverse_document(document_ID,document_text)


    document_ID='1130'
    document_text="The boy is reading about football reading all the time"
    inverted_structure.inverse_document(document_ID,document_text)
    
    
    document_ID='1240'
    document_text="The boy is watching videos about street football"
    inverted_structure.inverse_document(document_ID,document_text)
    
    end=time.time()
    print("average time to inverse a document ",(end-start)/3,flush=True) 
    print("-------------document IDs------------",flush=True)
    for doc_ID in inverted_structure.document_IDs:
        print('\n'+doc_ID,flush=True)
    print("-------------Vocabulary------------",flush=True)
    for token,value in inverted_structure.vocabulary.items():
        print( token ," length = " ,value[0], " position= ",value[1],flush=True)
    print("-------------Posting lists------------",flush=True)
    i=0
    for posting_list in inverted_structure.posting_lists:
        print("Posting list " ,i,flush=True)
        for elem in posting_list:
            print(elem, ' ',flush=True)
        i+=1 
        print('\n',flush=True)
    print("------------Documents length--------------",flush=True)
    for doc_length in inverted_structure.documents_length:
        print('\n',doc_length,flush=True)
    print("------------ IDF ----------------",flush=True)
    inverted_structure.compute_idf()
    for token,idf in inverted_structure.idf.items():
        print( token ," idf = " ,idf,flush=True)
    print("------------ Collection frequencies ----------------",flush=True)
    inverted_structure.compute_collection_frequencies()
    for token,c_freq in inverted_structure.c_freq.items():
        print( token ," collection frequency = " ,c_freq,flush=True)
    
    
    #Saving the structure
    print("Test save structure",flush=True)
    start=time.time()
    inverted_structure.save(file_path)
    end=time.time()
    print('posting file size   :',os.path.getsize(file_path+'/posting_file'),flush=True)
    print('vocabulary file size:',os.path.getsize(file_path+'/vocabulary'),flush=True)
    print('document_ID file size',os.path.getsize(file_path+'/document_IDs'),flush=True)
    print('documents_length file size',os.path.getsize(file_path+'/documents_length'),flush=True)
    print('Time to save inverted structure ', end-start,flush=True)
    #Loading the structure
    print("Test load structure ",flush=True)      
    start=time.time()
    inverted_structure2=Inverted_structure()
    inverted_structure2.load(file_path)
    end=time.time()
    print('Time to load inverted structure ', end-start,flush=True)
    #Checking that the two structures are equal
    if are_equal_doc(inverted_structure,inverted_structure2):
        print("The save and load doc were successful",flush=True)
    else:
        print("The save and load doc were not successful",flush=True)
    if are_equal_voc(inverted_structure,inverted_structure2):
        print("The save and load voc were successful",flush=True)
    else:
        print("The save and load  voc were not successful",flush=True)
    if are_equal_post(inverted_structure,inverted_structure2):
        print("The save and load post were successful",flush=True)
    else:
        print("The save and load  post were not successful",flush=True)
        
    if are_equal_doc_length(inverted_structure,inverted_structure2):
        print("The save and load doc length were successful",flush=True)
    else:
        print("The save and load doc length  were not successful",flush=True)    
def test_inverse_document_random_text(file_path):
    """ Test inverse_document_method """
    #Building inverted structure
    print("Test inverse_document",flush=True)
    #Generate documents with the same average length of documents of Wikir Collection
    start0=time.time()
    docs_list=[]
    docs_ID_list=[]
    for i in range(100000):
        doc_ID=str(randint(1000,10000))
        doc=' '.join(sample(words.words(), 754))
        docs_ID_list.append(doc_ID)
        docs_list.append(doc)
    print("Time to generate documents" , time.time()-start0,flush=True)
    start=time.time()
    inverted_structure=Inverted_structure()
    
    for i in range (len(docs_ID_list)):
        inverted_structure.inverse_document(docs_ID_list[i],docs_list[i])
    
    end=time.time()
    print("average time to inverse a document ",(end-start)/len(docs_ID_list),flush=True) 

    #Saving the structure
    print("Test save structure",flush=True)
    start=time.time()
    inverted_structure.save(file_path)
    end=time.time()
    print('posting file size   :',os.path.getsize(file_path+'/posting_file'),flush=True)
    print('vocabulary file size:',os.path.getsize(file_path+'/vocabulary'),flush=True)
    print('document_ID file size',os.path.getsize(file_path+'/document_IDs'),flush=True)
    print('documents_length file size',os.path.getsize(file_path+'/documents_length'),flush=True)
    print('Time to save inverted structure ', end-start,flush=True)
    #Loading the structure
    print("Test load structure ",flush=True)      
    start=time.time()
    inverted_structure2=Inverted_structure()
    inverted_structure2.load(file_path)
    end=time.time()
    print('Time to load inverted structure ', end-start,flush=True)
    start=time.time()
    #Checking that the two structures are equal
    if are_equal_doc(inverted_structure,inverted_structure2):
        print("The save and load doc were successful",flush=True)
    else:
        print("The save and load doc were not successful",flush=True)
    if are_equal_voc(inverted_structure,inverted_structure2):
        print("The save and load voc were successful",flush=True)
    else:
        print("The save and load  voc were not successful",flush=True)
    if are_equal_post(inverted_structure,inverted_structure2):
        print("The save and load post were successful",flush=True)
    else:
        print("The save and load  post were not successful",flush=True)
    if are_equal_doc_length(inverted_structure,inverted_structure2):
        print("The save and load doc length were successful",flush=True)
    else:
        print("The save and load doc length  were not successful",flush=True)
    end=time.time()
    print(" Time for checking matching", end-start,flush=True)
    start=time.time()
    inverted_structure.compute_idf()
    end=time.time()
    print("Time to compute IDF ", start-end,flush=True)
    start=time.time()
    inverted_structure.compute_collection_frequencies()
    end=time.time()
    print("Time to compute collection frequencies ", start-end,flush=True)
    print("Total time ", time.time()-start0,flush=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path')
    args = parser.parse_args()

    test_inverse_document_random_text(args.file_path)


if __name__ == "__main__":
    main()
