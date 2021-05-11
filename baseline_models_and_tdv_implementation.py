
import numpy as np
from collections import Counter

"""Baseline models and their corresponding tdv implementation"""


def simple_tf(indexed_queries, inverted_index):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += freq
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def weighted_simple_tf(indexed_queries, inverted_index, weights):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += weights[token] * freq
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def tf_idf(indexed_queries, inverted_index, idf):
    results = []

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += freq * idf[token]
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def dir_language_model(indexed_queries, inverted_index, docs_length, c_freq, mu=2500):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += np.log(1 + (freq / (mu * c_freq[token]))) + np.log(
                        mu / (docs_length[document] + mu))
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def Okapi_BM25(indexed_queries, inverted_index, docs_length, idf, k1=1.2, b=0.75):
    results = []

    avg_docs_len = sum(docs_length) / len(docs_length)

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * docs_length[document] / avg_docs_len))
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def fast_Okapi_BM25(indexed_queries, inverted_index, docs_length, idf, avg_docs_len, k1=1.2, b=0.75):
    results = []

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * docs_length[document] / avg_docs_len))
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def weighted_tf_idf(indexed_queries, inverted_index, weights, idf):
    results = []

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += weights[token] * freq * idf[token]
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def weighted_dir_language_model(indexed_queries, inverted_index, weights, docs_length, c_freq, mu=2500):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += weights[token] * (
                                np.log(1 + (freq / (mu * c_freq[token]))) + np.log(mu / (docs_length[document] + mu)))
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def weighted_Okapi_BM25(indexed_queries, inverted_index, weights, docs_length, idf, k1=1.2, b=0.75):
    results = []

    avg_docs_len = sum(docs_length) / len(docs_length)

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += weights[token] * idf[token] * ((k1 + 1) * freq) / (
                                freq + k1 * ((1 - b) + b * docs_length[document] / avg_docs_len))
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def Lemur_tf_idf(indexed_queries, inverted_index, docs_length, idf, k1=1.2, b=0.75):
    avg_docs_len = sum([value for key, value in docs_length.items()]) / len(docs_length)

    results = []

    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    Robertson_tf = k1 * freq / (freq + k1 * (1 - b + b * docs_length[document] / avg_docs_len))
                    result[document] += Robertson_tf * np.power(idf[token], 2)
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def JM_language_model(indexed_queries, inverted_index, docs_length, c_freq, Lambda=0.15):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += np.log(
                        1 + ((1 / (c_freq[token])) * (Lambda * freq) / ((1 - Lambda) * docs_length[document])))

        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results

def weighted_JM_language_model(indexed_queries, inverted_index, docs_length, c_freq,weights, Lambda=0.15):
    results = []
    for indexed_query in indexed_queries:
        result = Counter()
        for token in indexed_query:
            if token in inverted_index.token():
                for document, freq in inverted_index.posting_list(token):
                    result[document] += weights[token]*np.log(
                        1 + ((1 / (c_freq[token])) * (Lambda * freq) / ((1 - Lambda) * docs_length[document])))

        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results
