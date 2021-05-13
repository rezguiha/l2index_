
import numpy as np
from collections import Counter

"""Baseline models and their corresponding tdv implementation"""
#queries_struct is an instance of the Queries class
#inverted_struct is an instance of Inverted_structure

class simple_tf:
    def __init__(queries_struct, inverted_struct, max=1000):
        self.queries_struct = queries_struct

    def runQueries():
        # Declare only one objet pour cumulate the answers
        result = Counter()
        # Using the generator of the class Queries to go through the processed queries
        for query in sef.queries_struct.query():
            # Empty the ansers list
            result.clear()
            for token in query:
                #Using the generator of the class Inverted_structure to go through the tokens in the vocabulary
                if inverted_struct.existsToken(token):
                    #Using the generator of the class Inverted_structure to go through the tuples (document internal id,frequency) generated from the posting list of the token
                    for document, freq in inverted_struct.posting_list(token):
                        result[document] += freq
    #         if len(result) == 0:
    #             result[-1] += 0
            # Sort sur la valeur <======
            # et et limite à max resultats
            yield result

def weighted_simple_tf(queries_struct, inverted_struct, weights):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += weights[token] * freq
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def tf_idf(queries_struct, inverted_struct):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += freq * inverted_struct.idf[token]
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def dir_language_model(queries_struct, inverted_struct, mu=2500):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += np.log(1 + (freq / (mu * inverted_struct.c_freq[token]))) + np.log(
                        mu / (inverted_struct.documents_length[document] + mu))
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def Okapi_BM25(queries_struct, inverted_struct, k1=1.2, b=0.75):
    results = []

    avg_docs_len = sum(inverted_struct.documents_length) / len(inverted_struct.documents_length)

    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += inverted_struct.idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * inverted_struct.documents_length[document] / avg_docs_len))
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def fast_Okapi_BM25(queries_struct, inverted_struct, avg_docs_len, k1=1.2, b=0.75):
    results = []

    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += inverted_struct.idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * inverted_struct.documents_length[document] / avg_docs_len))
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def weighted_tf_idf(queries_struct, inverted_struct, weights):
    results = []

    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += weights[token] * freq * inverted_struct.idf[token]
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def weighted_dir_language_model(queries_struct, inverted_struct, weights, mu=2500):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += weights[token] * (
                                np.log(1 + (freq / (mu * inverted_struct.c_freq[token]))) + np.log(mu / (inverted_struct.documents_length[document] + mu)))
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def weighted_Okapi_BM25(queries_struct, inverted_struct, weights, k1=1.2, b=0.75):
    results = []

    avg_docs_len = sum(inverted_struct.documents_length) / len(inverted_struct.documents_length)

    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += weights[token] * inverted_struct.idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * inverted_struct.documents_length[document] / avg_docs_len))
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def Lemur_tf_idf(queries_struct, inverted_struct, docs_length, idf, k1=1.2, b=0.75):
    avg_docs_len = sum([value for key, value in docs_length.items()]) / len(docs_length)

    results = []

    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    Robertson_tf = k1 * freq / (freq + k1 * (1 - b + b * docs_length[document] / avg_docs_len))
                    result[document] += Robertson_tf * np.power(idf[token], 2)
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def JM_language_model(queries_struct, inverted_struct, docs_length, c_freq, Lambda=0.15):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += np.log(
                        1 + ((1 / (inverted_struct.c_freq[token])) * (Lambda * freq) / ((1 - Lambda) * inverted_struct.documents_length[document])))

#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results

def weighted_JM_language_model(queries_struct, inverted_struct, docs_length, c_freq,weights, Lambda=0.15):
    results = []
    for query in queries_struct.query():
        result = Counter()
        for token in query:
            if token in inverted_struct.token():
                for document, freq in inverted_struct.posting_list(token):
                    result[document] += weights[token]*np.log(
                        1 + ((1 / (inverted_struct.c_freq[token])) * (Lambda * freq) / ((1 - Lambda) * inverted_struct.documents_length[document])))

        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results
