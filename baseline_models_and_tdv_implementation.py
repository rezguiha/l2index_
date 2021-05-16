
import numpy as np
from collections import Counter

"""Baseline models and their corresponding tdv implementation"""
#queries_struct is an instance of the Queries class
#inverted_struct is an instance of Inverted_structure

class simple_tf:
    def __init__(self,queries_struct, inverted_struct, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
    def runQueries(self):
        # Declare only one objet pour cumulate the answers
        result = Counter()
        # Using the generator of the class Queries to go through the processed queries
        for query in self.queries_struct.query():
            # Empty the ansers list
            result.clear()
            for token in query:
                #Using the generator of the class Inverted_structure to go through the tokens in the vocabulary
                if self.inverted_struct.existsToken(token):
                    #Using the generator of the class Inverted_structure to go through the tuples (document internal id,frequency) generated from the posting list of the token
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += freq
    #         if len(result) == 0:
    #             result[-1] += 0
            # Sort sur la valeur <======
            # et et limite à max resultats
            yield result

class weighted_simple_tf:
    def __init__(self,queries_struct, inverted_struct,weights, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.weights=weights
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.weights[token] * freq
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result


class tf_idf:
    def __init__(self,queries_struct, inverted_struct, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += freq * self.inverted_struct.idf[token]
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result


class dir_language_model:
    def __init__(self,queries_struct, inverted_struct, mu=2500, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.mu=mu
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += np.log(1 + (freq / (self.mu * self.inverted_struct.c_freq[token]))) + np.log(
                        self.mu / (self.inverted_struct.documents_length[document] + self.mu))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result

class Okapi_BM25:
    def __init__(self,queries_struct, inverted_struct,k1=1.2, b=0.75, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.k1=k1
        self.b=b
        self.avg_docs_len = sum(self.inverted_struct.documents_length) / len(self.inverted_struct.documents_length)
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.inverted_struct.idf[token] * ((self.k1 + 1) * freq) / (
                                freq + self.k1 * ((1 - self.b) + self.b * self.inverted_struct.documents_length[document] / self.avg_docs_len))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result


class fast_Okapi_BM25:
    def __init__(self,queries_struct, inverted_struct,avg_docs_len,k1=1.2, b=0.75, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.k1=k1
        self.b=b
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.inverted_struct.idf[token] * ((self.k1 + 1) * freq) / (
                                freq + self.k1 * ((1 - self.b) + self.b * self.inverted_struct.documents_length[document] / avg_docs_len))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result



class weighted_tf_idf:
    def __init__(self,queries_struct, inverted_struct,weights, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.weights=weights
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.weights[token]*freq * inverted_struct.idf[token]
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result




class weighted_dir_language_model:
    def __init__(self,queries_struct, inverted_struct,weights, mu=2500, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.weigths=weights
        self.mu=mu
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.weights[token]*np.log(1 + (freq / (self.mu * self.inverted_struct.c_freq[token]))) + np.log(
                        self.mu / (self.inverted_struct.documents_length[document] + self.mu))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result


class weighted_Okapi_BM25:
    def __init__(self,queries_struct, inverted_struct,weights,k1=1.2, b=0.75, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.k1=k1
        self.b=b
        self.weights=weigths
        self.avg_docs_len = sum(self.inverted_struct.documents_length) / len(self.inverted_struct.documents_length)
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.weights[token]*self.inverted_struct.idf[token] * ((self.k1 + 1) * freq) / (
                                freq + self.k1 * ((1 - self.b) + self.b * self.inverted_struct.documents_length[document] / self.avg_docs_len))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result


class Lemur_tf_idf:
    def __init__(self,queries_struct, inverted_struct,k1=1.2, b=0.75, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.k1=k1
        self.b=b
        self.avg_docs_len = sum(self.inverted_struct.documents_length) / len(self.inverted_struct.documents_length)
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        Robertson_tf = self.k1 * freq / (freq + self.k1 * (1 - self.b + self.b * self.inverted_struct.docs_length[document] / self.avg_docs_len))
                    result[document] += Robertson_tf * np.power(self.inverted_struct.idf[token], 2)
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result

        
class JM_language_model:
    def __init__(self,queries_struct, inverted_struct,Lambda=0.15, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.Lambda=Lambda
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += np.log(
                        1 + ((1 / (self.inverted_struct.c_freq[token])) * (self.Lambda * freq) / ((1 - self.Lambda) * self.inverted_struct.documents_length[document])))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result

class weighted_JM_language_model:
    def __init__(self,queries_struct, inverted_struct,weights,Lambda=0.15, max=1000):
        self.queries_struct = queries_struct
        self.inverted_struct=inverted_struct
        self.Lambda=Lambda
        self.weights=weights
    def runQueries(self):
        result = Counter()
        for query in self.queries_struct.query():
            result.clear()
            for token in query:
                if self.inverted_struct.existsToken(token):
                    for document, freq in self.inverted_struct.posting_list(token):
                        result[document] += self.weights[token]*np.log(
                        1 + ((1 / (self.inverted_struct.c_freq[token])) * (self.Lambda * freq) / ((1 - self.Lambda) * self.inverted_struct.documents_length[document])))
    #         if len(result) == 0:
    #             result[-1] += 0
            yield result

