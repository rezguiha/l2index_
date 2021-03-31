# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 10 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 16 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of useful functions to build vocabulary and index 
# queries and documents of collections. 
# HR modified the name of the functions by adding prefix std_tokenizer because
# there are functions of the same name in other code files
# =============================================================================
from collections import Counter
from nltk.corpus import stopwords


def std_tokenizer_build_standard_vocabulary(queries, documents, min_occ=2, limit_docs=None, limit_queries=None):
    """Function that builds the standard vocabulary from a list of queries and a list of documents and
    with a limit on the number of documents and queries to manipulate""" #HR
    vocabulary = Counter()

    count = 0
    for _, document in documents.iterrows():
        for word in document[0].split(" "):
            vocabulary[word] += 1
        count += 1
        if count == limit_docs:
            break

    count = 0
    for _, query in queries.iterrows():
        for word in query[0].split(" "):
            vocabulary[word] += 1
        count += 1
        if count == limit_queries:
            break

    vocabulary = {i: elem[0] for i, elem in enumerate(vocabulary.most_common()) if elem[1] >= min_occ}

    for key in list(vocabulary):
        vocabulary[vocabulary[key]] = key

    return vocabulary


def std_tokenizer_index(pdDataFrame, vocabulary, stemmer=None):
    """Function that indexes a dataframe either documents or queries for example according to a vocabulary.
    While doing that it can perform a stemmerization if the vocabulary was built on words that got stemmerized""" #HR
    indexed_elements = []
    index = dict()
    count = 0
    if stemmer is None:
        for key, element in pdDataFrame.iterrows():
            indexed_elements.append(
                [vocabulary[elem.lower()] for elem in element[0].split(" ") if elem.lower() in vocabulary])
            index[str(key)] = count
            index[count] = str(key)
            count += 1

    else:
        for key, element in pdDataFrame.iterrows():
            indexed_elements.append([vocabulary[stemmer.stem(elem.lower())] for elem in element[0].split(" ") if
                                     stemmer.stem(elem.lower()) in vocabulary])
            index[str(key)] = count
            index[count] = str(key)
            count += 1

    return index, indexed_elements


def std_tokenizer_index_dict(pdDataFrame, vocabulary):
    """Function that indexes a dict that could be documents or queries for example according to a vocabulary""" #HR
    indexed_elements = []
    index = dict()
    count = 0
    for key, element in pdDataFrame.items():
        indexed_elements.append([vocabulary[elem] for elem in element.split(" ") if elem in vocabulary])
        index[str(key)] = count
        index[count] = str(key)
        count += 1
    return index, indexed_elements


def std_tokenizer_preprocess(queries, documents, min_occ=5): 
    """Function that preprocesses queries and documents. It builds the standard vocabulary and indexes both
    the documents and the queries and returns the vocabulary , the query and doc index and the indexed elements of
    both doc and  query""" #HR
    vocabulary = std_tokenizer_build_standard_vocabulary(queries,
                                                         documents,
                                                         min_occ=min_occ)

    doc_index, indexed_docs = std_tokenizer_index(documents, vocabulary)

    query_index, indexed_queries = std_tokenizer_index(queries, vocabulary)

    return vocabulary, query_index, indexed_queries, doc_index, indexed_docs
