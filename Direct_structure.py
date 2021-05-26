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
#Defintion of Direct structure class
class Direct_structure:
    """This class defines the direct structure for classical IR system. We speak about documents in this class but it is not restricted to documents it can be queries too."""
    def __init__(self):
        # List of processed documents
        self.processed_documents=arr.array('I')
        #Array of documents'length
        self.doc_length=arr.array('I')

    def add_document(self,processed_document):
        """
        Adding the array processed document to the list of processed documents,Adding document ID to the list of document IDs and adding the document's length to the array of document's length
        """
        self.processed_documents.extend(processed_document)

    def add_document_and_save(self,processed_document,file_path):
        """
        Saving the processed document just after it was processed
        """
        with open(file_path+'/processed_documents','wb') as f:
            processed_document.tofile(f)

    def saving_all_documents(self,file_path):
        """
        Saving all documents that has been processed and added to the overall array
        """
        with open(file_path+'/processed_documents','wb') as f:
            self.processed_documents.tofile(f)
    def load(self,file_path):
        self.doc_length=arr.array('I')
        self.processed_documents=[]
        #Loading documents length
        with open(file_path+'/documents_length', 'rb') as f:
            #We use frombytes because it does not require to enter the number of elements you want to retrieve. fromfile does.
            self.doc_length.frombytes(f.read())
        
        #Loading processed documents
        with open(file_path+'/processed_documents', 'rb') as f:
            for document_length in self.doc_length:
                processed_document=arr.array('I')
                processed_document.fromfile(f,document_length)
                self.processed_documents.append(processed_document)
    def get_number_of_document(self):
        return len(self.doc_length)
    
    
 