#Packages
import argparse
import os
import time
import array as arr

from random import sample
from random import choices

#files
from Direct_structure import Direct_structure
from Inverted_structure import Inverted_structure

def test_filter(file_path):
    """ Test inverse_document_method """
    #Building inverted structure
    print("Test inverse_document",flush=True)
    start=time.time()
    
    inverted_structure=Inverted_structure()
    document_ID='1125'
    document_text="The boy is playing football in the street boy boy street boy"

    inverted_structure.inverse_document(document_ID,document_text)


    document_ID='1130'
    document_text="The boy is boy reading about football reading all the time"
    inverted_structure.inverse_document(document_ID,document_text)
    
    
    document_ID='1240'
    document_text="The man is watching street football"
    inverted_structure.inverse_document(document_ID,document_text)
    
    document_ID='1241'
    document_text="The boy is street in love"
    inverted_structure.inverse_document(document_ID,document_text)
    
    document_ID='1242'
    document_text="love boy is in the air air air air air"
    inverted_structure.inverse_document(document_ID,document_text)
    
    document_ID='1243'
    document_text="filter boy street carbon in the car air"
    inverted_structure.inverse_document(document_ID,document_text)
    
    document_ID='1244'
    document_text="street with no parking"
    inverted_structure.inverse_document(document_ID,document_text)
    

    
    end=time.time()
    print("average time to inverse a document ",(end-start)/7,flush=True) 
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
    for doc_length in inverted_structure.direct_structure.doc_length:
        print('\n',doc_length,flush=True)
    print("------------Directed_structure--------------",flush=True)  
    pos=0
    document_number=0
    for doc_length in inverted_structure.direct_structure.doc_length:
        print("Document ", document_number,flush=True)
        for i in range(doc_length):
            print('\n',inverted_structure.direct_structure.processed_documents[pos+i],flush=True)
        pos+=doc_length
        document_number+=1
    #Filtering
    print("----------------------------------------------",flush=True)
    print("----------------------------------------------",flush=True)
    print("----------------------------------------------",flush=True)
    print("-------------------Filtering------------------",flush=True)
    start=time.time()
    inverted_structure.filter_vocabulary(minimum_occurence=2,proportion_of_frequent_words=0.5)
    end=time.time()
    print("average time to filter vocabulary and posting lists and docs length ",(end-start)/7,flush=True)
    
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
        
    print("------------Directed_structure--------------",flush=True)  
    
    pos=0
    document_number=0
    for doc_length in inverted_structure.direct_structure.doc_length:
        print("Document ", document_number,flush=True)
        for i in range(doc_length):
            print('\n',inverted_structure.direct_structure.processed_documents[pos+i],flush=True)
        pos+=doc_length
        document_number+=1
    #Saving the structure
    print("Test save structure",flush=True)
    start=time.time()
    inverted_structure.save(file_path)
    end=time.time()
    print('posting file size   :',os.path.getsize(file_path+'/posting_file'),flush=True)
    print('vocabulary file size:',os.path.getsize(file_path+'/vocabulary'),flush=True)
    print('document_ID file size',os.path.getsize(file_path+'/document_IDs'),flush=True)
    print('documents_length file size',os.path.getsize(file_path+'/documents_length'),flush=True)
    print('Size of directed_structure',os.path.getsize(file_path+'/processed_documents'),flush=True)
    print('Time to save inverted structure ', end-start,flush=True)


def test_filter_vocabulary_random_text():
    """ Test inverse_document_method with random generate text"""
    #Building inverted structure
    print("Test filter vocabulary of Direct_structure class",flush=True)
    #Generate documents with the same average length of documents of Wikir Collection
    start0=time.time()
    old_vocab_size=8400000
    size_of_processed_documents=1000000000
    print("Size of old vocabulary =",old_vocab_size,flush=True)
    print("Size of processed documents =" ,size_of_processed_documents,flush=True)
    #choice function generates k elements with replacement from a list of elements
    processed_documents=arr.array('I',choices([i for i in range(old_vocab_size)],k=size_of_processed_documents))
    new_vocab_size=60000
    print("Size of filtered vocabulary =",new_vocab_size,flush=True)
    vocab_table=arr.array('I',[i for i in range(old_vocab_size)])
    #sample function generates a list of k unique elements from a list of elements
    indices_to_remove_vocab=sample(list(vocab_table),k=old_vocab_size-new_vocab_size)
    for index in indices_to_remove_vocab:
        vocab_table[index]=0xffffffff
    new_index=0
    for i in range(old_vocab_size):
        if vocab_table[i]!=0xffffffff:
            vocab_table[i]=new_index
            new_index+=1
    end=time.time()
    print("Time to generate processed documents and vocabulary table =",round(end-start0) ," s",flush=True)
    direct_structure=Direct_structure()
    direct_structure.processed_documents=processed_documents
    direct_structure.doc_length=arr.array('I',[500 for i in range(size_of_processed_documents//500)])
    start=time.time()
    direct_structure.filter_vocabulary(vocab_table)
    end=time.time()
    print("Length of processed_documents= ",len(direct_structure.processed_documents),flush=True)
    print("Time to filter processed_documents= ", round(end-start),flush=True)
    print("Average time to filter processed_documents= ", round((end-start)/(size_of_processed_documents//500)),flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path')
    args = parser.parse_args()

    test_filter(args.file_path)

if __name__ == "__main__":
    main()
