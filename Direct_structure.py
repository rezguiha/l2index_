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
    """This class defines the direct structure for classical IR system. We speak about documents in this class but it is not restricted to documents it can be queries too. The directed structure is the inverse of posting lists"""
    def __init__(self):
        # List of processed documents
        self.documents=arr.array('I')
        #Array of documents'length
        self.documents_length=arr.array('I')

    def add_document(self,processed_document):
        """
        Adding the array processed document to the list of processed documents,Adding document ID to the list of document IDs and adding the document's length to the array of document's length
        """
        self.documents.extend(processed_document)

    def add_document_and_save(self,document,file_path):
        """
        Saving the processed document just after it was processed
        """
        with open(file_path+'/documents','wb') as f:
            document.tofile(f)

    def filter_vocabulary(self,new_vocabulary):
        """
        Change the vocabulary of all documents
        new_vocabulary: an array of unsing int of new token id indexed by the previous id
        At the end of this process, the whole collection is represented in the new token id
        To filter the id, some old token id are associated to the max unsigned value: 0xffffffff
        As a consequence of this transformation and filtering, documents may be shorter
        """
        # Position of the old term in current document
        oldPos = 0
        # Position of the new term in current document
        newPos = 0
        # The new size of all document content
        newContentSize = 0;
        # Filter all documents
        for docId in range(len(self.documents_length)):
            # New current document size
            newDocSize = 0
            # for all token of this document
            for tokenNb in range(self.documents_length[docId]):
                # Get current tocken id
                oldId = self.documents[oldPos]
                # Get new tocken
                #The old token ID starts from 1 which corresponds to element oldId-1 in the new_vocabulary table
                newId = new_vocabulary[oldId-1]
                # Test if this tocken is removed
                if newId == 0xffffffff:
                    # this tocken is removed, so we only move to next old tocken
                    oldPos += 1
                else:
                    # tocken not removed, so added to current document
                    self.documents[newPos] = newId
                    # next token and next position in the document
                    oldPos += 1
                    newPos += 1
                    newDocSize += 1
                    newContentSize += 1
            # Store the new document size
            self.documents_length[docId] = newDocSize
        # All document are filtered, remove the extra space if necessary
        if newContentSize < len(self.documents):
            # We hope that python is smart enought for managing memory here
            self.documents = self.documents[:newContentSize]


    def saving_all_documents_and_documents_length(self,file_path):
        """
        Saving all documents that has been processed and added to the overall array
        """
        #Saving the documents length
        with open(file_path+'/documents_length','wb') as f:
            self.documents_length.tofile(f)
        #Saving the direct documents
        with open(file_path+'/direct_documents','wb') as f:
            self.documents.tofile(f)
            
    def load(self,file_path):
        self.documents_length=arr.array('I')
        self.documents=[]
        #Loading documents length
        with open(file_path+'/documents_length', 'rb') as f:
            #We use frombytes because it does not require to enter the number of elements you want to retrieve. fromfile does.
            self.documents_length.frombytes(f.read())

        #Loading processed documents
        with open(file_path+'/direct_documents', 'rb') as f:
            for document_length in self.documents_length:
                processed_document=arr.array('I')
                processed_document.fromfile(f,document_length)
                self.documents.append(processed_document)
    def get_number_of_document(self):
        return len(self.documents_length)
