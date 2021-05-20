from Inverted_structure import Inverted_structure
import argparse
import time
def test_filter_inverted_structure(indexed_path,path_to_save_plot):
    inverted_structure=Inverted_structure()
    inverted_structure.load(indexed_path)
    start=time.time()
    inverted_structure.filter_vocabulary(minimum_occurence=5,proportion_of_frequent_words=0.2)
    end=time.time()
          
    number_of_documents=inverted_structure.get_number_of_documents()
    print("Total time to filter vocabulary and posting lists= ",round(end-start)," s",flush=True)
    print("Average time to filter vocabulary and posting lists= ",round(((end-start)/number_of_documents)*1000)," ms",flush=True)
    inverted_structure.statistics_about_the_structure(path_to_save_plot)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indexed_path')
    parser.add_argument('-s','--save_plot_path')
    args = parser.parse_args()

    test_filter_inverted_structure(args.indexed_path,args.save_plot_path)


if __name__ == "__main__":
    main()