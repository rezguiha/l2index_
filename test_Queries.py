#Packages 
import argparse
import os
import time
import pickle
from nltk.corpus import words
from random import sample
from random import randint
#Files
from Queries import Queries
def are_equal_query_IDs(queries1,queries2):
    #Equality of queries IDs
    if len(queries1.queries_IDs)!= len(queries2.queries_IDs):
        print("query_Ids size does not match",flush=True)
        return False
    else:
        for i in range(len(queries1.queries_IDs)):
            if queries1.queries_IDs[i]!=queries2.queries_IDs[i]:
                print( " mismatch at position ", i, " in Doc_IDs",flush=True)
                return False
    return True
def are_equal_processed_queries(queries1,queries2):
    #Equality of processed queries
    if len(queries1.processed_queries)!= len(queries2.processed_queries):
        print("Processed queries size does not match",flush=True)
        return False
    else:
        for i in range(len(queries1.processed_queries)):
            if len(queries1.processed_queries[i])!= len(queries2.processed_queries[i]):
                print("processed query ",i," size does not match",flush=True)
                return False
            else:
                for j in range(len(queries1.processed_queries[i])):
                    if queries1.processed_queries[i][j]!=queries2.processed_queries[i][j]:
                        print( " mismatch at query ", i, " in position ",j,flush=True)
                        return False
    return True

def test_Queries_class(file_path):
    print("Test Queries",flush=True)
    start=time.time()
    queries=Queries()
    query1= "Is the boy fond of street football"
    query1_ID='1000'
    queries.process_query_and_get_ID(query1_ID,query1)
    query2= "Where can i watch football games"
    query2_ID='1003'
    queries.process_query_and_get_ID(query2_ID,query2)
    query3="Books about street football"
    query3_ID='1005'
    queries.process_query_and_get_ID(query3_ID,query3)
    query4="recipes with avocados"
    query4_ID='1008'
    queries.process_query_and_get_ID(query4_ID,query4)
    query5="how are you"
    query5_ID='1010'
    queries.process_query_and_get_ID(query5_ID,query5)
    end=time.time()
    print("average time to process query ",(end-start)/5,flush=True)
    print("-------------queries IDs------------",flush=True)
    for query_ID in queries.queries_IDs:
        print('\n'+query_ID,flush=True)
    print("-----------processed queries-----------", flush=True)
    i=0
    for query in queries.processed_queries:
        print ("----query ",i,flush=True)
        for token in query:
            print('\n'+ token,flush=True)
        i+=1
     #Saving the structure
    print("Test save Queries",flush=True)
    start=time.time()
    queries.save(file_path,"uni_test")
    end=time.time()
    print("Time for save ", end-start,flush=True)
    print('query IDs file size   :',os.path.getsize(file_path+'/'+'uni_test'+'_queries_IDs'),flush=True)
    print('processed queries file size:',os.path.getsize(file_path+'/'+'uni_test'+'_queries'),flush=True)       
     #Loading the queries
    print("Test load queries",flush=True)      
    start=time.time()
    queries2=Queries()
    queries2.load(file_path,"uni_test")                                             
    end=time.time()
    print('Time to load queries ', end-start,flush=True) 
    
    print("Cheking for load and save of queries")                                                     
    if are_equal_query_IDs(queries,queries2):
        print("The save and load query IDs were successful",flush=True)
    else:
        print("The save and load query IDs were not successful",flush=True)
    if are_equal_processed_queries(queries,queries2):
        print("The save and load processed queries were successful",flush=True)
    else:
        print("The save and load  processed queries were not successful",flush=True)
def test_Queries_class_random_queries(file_path,average_length_query):
    print("-----------------Test Queries randomly generated------------",flush=True)
    #Building queries instance

    #Generate queries with the same average length of queries of Wikir Collection or WikirS collection
    start0=time.time()
    query_list=[]
    query_ID_list=[]
    for i in range(20000):
        query_ID=str(randint(1000,10000))
        query=' '.join(sample(words.words(), average_length_query))
        query_ID_list.append(query_ID)
        query_list.append(query)
    print("Time to generate documents" , time.time()-start0,flush=True)
    start=time.time()
    queries=Queries()
    
    for i in range (len(query_ID_list)):
        queries.process_query_and_get_ID(query_ID_list[i],query_list[i])
    
    end=time.time()
    print("average time to process queries ",((end-start)/len(query_ID_list))*1000,' ms',flush=True) 
    
    
     #Saving the structure
    print("Test save Queries",flush=True)
    start=time.time()
    queries.save(file_path,"uni_test")
    end=time.time()
    print("Time for save ", end-start,'s',flush=True)
    print('query IDs file size   :',os.path.getsize(file_path+'/'+'uni_test'+'_queries_IDs'),flush=True)
    print('processed queries file size:',os.path.getsize(file_path+'/'+'uni_test'+'_queries'),flush=True)       
     #Loading the queries
    print("Test load queries",flush=True)      
    start=time.time()
    queries2=Queries()
    queries2.load(file_path,"uni_test")                                             
    end=time.time()
    print('Time to load queries ', end-start,' s',flush=True) 
    
    print("Cheking for load and save of queries",flush=True) 
    start=time.time()
    if are_equal_query_IDs(queries,queries2):
        print("The save and load query IDs were successful",flush=True)
    else:
        print("The save and load query IDs were not successful",flush=True)
    if are_equal_processed_queries(queries,queries2):
        print("The save and load processed queries were successful",flush=True)
    else:
        print("The save and load  processed queries were not successful",flush=True)
    end=time.time()
    print("Time to do checking of save and load ",end-start,' s',flush=True)
    print("Total time ", end-start0, ' s',flush=True)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path')
    args = parser.parse_args()

    test_Queries_class_random_queries(args.file_path,10)


if __name__ == "__main__":
    main()