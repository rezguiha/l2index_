from Inverted_structure import Inverted_structure
import argparse
def test_statistics_inverted_structure(indexed_path,path_to_save_plot):
    inverted_structure=Inverted_structure()
    inverted_structure.load(indexed_path)
    inverted_structure.statistics_about_the_structure(path_to_save_plot)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indexed_path')
    parser.add_argument('-s','--save_plot_path')
    args = parser.parse_args()

    test_statistics_inverted_structure(args.indexed_path,args.save_plot_path)


if __name__ == "__main__":
    main()