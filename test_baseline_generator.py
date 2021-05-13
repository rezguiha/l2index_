# Packages
import argparse
import time
import pytrec_eval
import os
# files
from Inverted_structure import Inverted_structure
from Queries import Queries
from collections import Counter

"""This file contains two parts: first part to test the use of generator in the baseline models and a second part to test the new implementation of compute metrics function from utils"""

# Testing the generators in a baseline model
def simple_tf(queries, inverted_index):
    results = []
    for query in queries.query():
        result = Counter()
        for token in query:
            if token in inverted_index.token():
                print("------token ", token, " is being evaluated", flush=True)
                for document, freq in inverted_index.posting_list(token):
                    print("document ", document, " has been generated with freq =", freq, flush=True)
                    result[document] += freq
        if len(result) == 0:
            result[-1] += 0
        results.append(result)

    return results


def tf_idf(queries, inverted_index, idf):
    results = []

    for query in queries.query():
        result = Counter()
        for token in query:
            if token in inverted_index.token():
                print("------token ", token, " is being evaluated", flush=True)
                for document, freq in inverted_index.posting_list(token):
                    print("document ", document, " has been generated with freq =", freq, flush=True)
                    result[document] += freq * idf[token]
#         if len(result) == 0:
#             result[-1] += 0
        results.append(result)

    return results


def test_generator_baseline_model(file_path):
    print("Test inverse_document", flush=True)
    start = time.time()

    inverted_structure = Inverted_structure()
    document_ID = '1125'
    document_text = "The boy is playing football in the street boy boy street boy"

    inverted_structure.inverse_document(document_ID, document_text)

    document_ID = '1130'
    document_text = "The boy is reading about football reading all the time"
    inverted_structure.inverse_document(document_ID, document_text)

    document_ID = '1240'
    document_text = "The boy is watching videos about street football"
    inverted_structure.inverse_document(document_ID, document_text)

    end = time.time()
    print("average time to inverse a document ", (end - start) / 3, flush=True)
    print("-------------document IDs------------", flush=True)
    for doc_ID in inverted_structure.document_IDs:
        print('\n' + doc_ID, flush=True)
    print("-------------Vocabulary------------", flush=True)
    for token, value in inverted_structure.vocabulary.items():
        print(token, " length = ", value[0], " position= ", value[1], flush=True)
    print("-------------Posting lists------------", flush=True)
    i = 0
    for posting_list in inverted_structure.posting_lists:
        print("Posting list ", i, flush=True)
        for elem in posting_list:
            print(elem, ' ', flush=True)
        i += 1
        print('\n', flush=True)
    print("------------Documents length--------------", flush=True)
    for doc_length in inverted_structure.documents_length:
        print('\n', doc_length, flush=True)
    print("------------ IDF ----------------", flush=True)
    inverted_structure.compute_idf()
    for token, idf in inverted_structure.idf.items():
        print(token, " idf = ", idf, flush=True)
    print("------------ Collection frequencies ----------------", flush=True)
    inverted_structure.compute_collection_frequencies()
    for token, c_freq in inverted_structure.c_freq.items():
        print(token, " collection frequency = ", c_freq, flush=True)

    queries=Queries()
    query1_ID='10200'
    query1 = "is the boy football street"
    queries.process_query_and_get_ID(query1_ID,query1)
    query2_ID='10205'
    query2 = "footbal video is he watching"
    queries.process_query_and_get_ID(query2_ID,query2)
    query3_ID='10209'
    query3 = "sleep or match cards"
    queries.process_query_and_get_ID(query3_ID, query3)
    results = tf_idf(queries, inverted_structure, inverted_structure.idf)
    i = 0
    for res in results:
        print("result for query ", i, flush=True)
        for key, value in res.items():
            print("document ", key, " = ", value, flush=True)
        i+=1
    
#Testing computing metrics 

def compute_metrics(queries_ID,documents_ID, qrel, results,score_file_path,top_k=1000, save_res=False):
    """Function that saves the results of retrieval: the top_k documents according to their score in a format suitable for the pytrec_eval library . Then, it computes different metrics for IR using the pytrec_eval package""" #HR
    # queries_ID is the array of queries IDs from a Queries instance
    #documents_ID is the array of document IDs from the Inverted_structure instance
    #qrel is the loaded query document relavance file
    #Score_file_path is the full path to  where to store the top_k results in the format of pytrec_eval with the file name (number of epoch)
    #results are the list of counter objects that you get from the baseline models
    
    #Writing the top k results in the format for pytrec_eval
    result_generator=(result for result in results) 
    with open(score_file_path+'/relavance_results', 'w') as f:
            for internal_query_ID, counter_doc_relavance_score in enumerate(result_generator):
                top_docs_generator=(x for x in counter_doc_relavance_score.most_common(top_k))
                for i, scores in enumerate(top_docs_generator):
                    internal_document_ID=int(scores[0])
                    relavance_score=scores[1]
                    f.write(queries_ID[internal_query_ID] + ' Q0 ' + documents_ID[internal_document_ID] + ' ' + str(i) + ' ' + str(relavance_score) + ' 0\n')
    
    #Loading result score file using pytrec_eval
    with open(score_file_path+'/relavance_results', 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    if not save_res:
        os.remove(score_file_path+'/relavance_results')
        
    #Evaluating metrics for all queries that have a query document relavance
    measures = {"map", "ndcg_cut", "recall", "P"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)

    all_metrics = evaluator.evaluate(run)
    #Aggregating metrics and computing the average
    metrics = {'P_5': 0,
               'P_10': 0,
               'P_20': 0,
               'ndcg_cut_5': 0,
               'ndcg_cut_10': 0,
               'ndcg_cut_20': 0,
               'ndcg_cut_1000': 0,
               'map': 0,
               'recall_1000': 0}

    nb_queries = len(all_metrics)
    for key, values in all_metrics.items():
        for metric in metrics:
            metrics[metric] += values[metric] / nb_queries

    return metrics
def test_compute_metrics(score_file_path,qrel_file_path):
    print("Test inverse_document", flush=True)
    start = time.time()

    inverted_structure = Inverted_structure()
    document_ID = '1125'
    document_text = "The boy is playing football in the street boy boy street boy"

    inverted_structure.inverse_document(document_ID, document_text)

    document_ID = '1130'
    document_text = "The boy is reading about football reading all the time"
    inverted_structure.inverse_document(document_ID, document_text)

    document_ID = '1240'
    document_text = "The boy is watching videos about street football"
    inverted_structure.inverse_document(document_ID, document_text)

    end = time.time()
    print("average time to inverse a document ", (end - start) / 3, flush=True)
    print("-------------document IDs------------", flush=True)
    for doc_ID in inverted_structure.document_IDs:
        print('\n' + doc_ID, flush=True)
    print("-------------Vocabulary------------", flush=True)
    for token, value in inverted_structure.vocabulary.items():
        print(token, " length = ", value[0], " position= ", value[1], flush=True)
    print("-------------Posting lists------------", flush=True)
    i = 0
    for posting_list in inverted_structure.posting_lists:
        print("Posting list ", i, flush=True)
        for elem in posting_list:
            print(elem, ' ', flush=True)
        i += 1
        print('\n', flush=True)
    print("------------Documents length--------------", flush=True)
    for doc_length in inverted_structure.documents_length:
        print('\n', doc_length, flush=True)
    print("------------ IDF ----------------", flush=True)
    inverted_structure.compute_idf()
    for token, idf in inverted_structure.idf.items():
        print(token, " idf = ", idf, flush=True)
    print("------------ Collection frequencies ----------------", flush=True)
    inverted_structure.compute_collection_frequencies()
    for token, c_freq in inverted_structure.c_freq.items():
        print(token, " collection frequency = ", c_freq, flush=True)

    queries=Queries()
    query1_ID='10200'
    query1 = "is the boy football street"
    queries.process_query_and_get_ID(query1_ID,query1)
    query2_ID='10205'
    query2 = "footbal video is he watching"
    queries.process_query_and_get_ID(query2_ID,query2)
    query3_ID='10209'
    query3 = "sleep or match cards"
    queries.process_query_and_get_ID(query3_ID, query3)
    results = tf_idf(queries, inverted_structure, inverted_structure.idf)
    i = 0
    for res in results:
        print("result for query ", i, flush=True)
        for key, value in res.items():
            print("document ", key, " = ", value, flush=True)
        i+=1
    
    
#     #creating a result scores file  in the format that pytrec_eval understands for the simple test we are doing
#     with open(score_file_path+'/relavance_results', 'w') as f:
#             for query, counter_doc_relavance_score in enumerate(results):
#                 for i, scores in enumerate(counter_doc_relavance_score.most_common(3)):
#                     internal_document_ID=int(scores[0])
#                     relavance_score=scores[1]
#                     f.write(queries.queries_IDs[query] + ' Q0 ' + inverted_structure.document_IDs[internal_document_ID] + ' ' + str(i) + ' ' + str(relavance_score) + ' 0\n')
                    
#     #Loading query relavance values #HR
#     with open(file_path + '/qrels', 'r') as f_qrel:
#         qrel = pytrec_eval.parse_qrel(f_qrel)
#     #Loading the result scores file
#     with open(score_file_path+'/relavance_results', 'r') as f_run:
#         run = pytrec_eval.parse_run(f_run)
        
#     #Computing metrics
#     measures = {"map", "ndcg_cut", "recall", "P"}

#     evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)

#     all_metrics = evaluator.evaluate(run)

#     metrics = {'P_5': 0,
#                'P_10': 0,
#                'P_20': 0,
#                'ndcg_cut_5': 0,
#                'ndcg_cut_10': 0,
#                'ndcg_cut_20': 0,
#                'ndcg_cut_1000': 0,
#                'map': 0,
#                'recall_1000': 0}

#     nb_queries = len(all_metrics)
#     for key, values in all_metrics.items():
#         for metric in metrics:
#             metrics[metric] += values[metric] / nb_queries

    #Loading query relavance values #HR
    with open(qrel_file_path + '/qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    metrics=compute_metrics(queries.queries_IDs,inverted_structure.document_IDs,qrel,results,score_file_path,top_k=3)

    print("------------metrics------------",flush=True)
    print(metrics.items(),flush=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path')
    parser.add_argument('-s', '--score_file_path')
    parser.add_argument('-q','--qrel_file_path')
    args = parser.parse_args()
    test_generator_baseline_model(args.file_path)
    test_compute_metrics(args.score_file_path,args.qrel_file_path)


if __name__ == "__main__":
    main()
