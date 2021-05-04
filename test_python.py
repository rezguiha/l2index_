import argparse
import os
import time
import pickle
import numpy
import struct

max=50000000

def pickle_test():
    """
    Test large object save and load with pickle
    """
    print("Pickle test")
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

def file_test():
    """
    Test large object save and load with binary file
    """
    print("File test")
    filename='bigfile.bin'
    # Create large list
    list = []
    tic = time.time()
    for i in range(max):
        list.append(numpy.int32(i))
    toc = time.time()
    print('create list in',round(toc-tic),'s')
    tic = time.time()
    with open(filename, 'wb') as f:
        for x in list:
            # Transforme en binaire
            bin = struct.pack("@i",x)
            f.write(bin)
    toc = time.time()
    print('Min file size   :',max*4)
    print('Actual file size:',os.path.getsize(filename))
    print('Bin dump in',round(toc-tic),'s')
    tic = time.time()
    loaded = []
    with open(filename, 'rb') as f:
        bin = f.read(4)
        while bin:
            i = struct.unpack("@i",bin)[0]
            loaded.append(i)
            bin = f.read(4)
    toc = time.time()
    print('Bin load in',round(toc-tic),'s')
    # Check
    tic = time.time()
    for i in range(max):
        if loaded[i] != i:
            print("Error",i)
            print(loaded)
            exit(1)
    toc = time.time()
    print('check in',round(toc-tic),'s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pickle_test',action='store_true',help='run pickle test')
    parser.add_argument('-f', '--file_test',action='store_true',help='run file test')
    args = parser.parse_args()

    if args.pickle_test:
        pickle_test()
    if args.file_test:
        file_test()


if __name__ == "__main__":
    main()
