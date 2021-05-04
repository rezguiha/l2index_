import argparse
import os
import time
import pickle
import numpy

def pickle_test():
    """
    Test larrge object save and load
    """
    print("Pickle test")
    max = 1000000
    filename='bigfile.pkl'
    # Create large list
    list = []
    tic = time.time()
    for i in range(max):
        list.append(numpy.int32(i))
    toc = time.time()
    print('create list in',round(toc-tic),'s')
    tic = time.time()
    with open(filename, 'wb') as f:
        pickle.dump(list,f)
    toc = time.time()
    print('Min file size   :',max*4)
    print('Actual file size:',os.path.getsize(filename))
    print('pickle dump in',round(toc-tic),'s')
    tic = time.time()
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
    toc = time.time()
    print('pickle load in',round(toc-tic),'s')
    # Check
    tic = time.time()
    for i in range(max):
        if loaded[i] != i:
            print("Error",i)
    toc = time.time()
    print('check in',round(toc-tic),'s')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pickle_test',action='store_true',help='run pickle test')
    args = parser.parse_args()

    if args.pickle_test is not None:
        pickle_test()


if __name__ == "__main__":
    main()
