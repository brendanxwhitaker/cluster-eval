from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import numpy as np
import sklearn
import sys
import csv
import os

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# source directory
dataset_path = sys.argv[1]

def check_valid_dir(some_dir):
    if not os.path.isdir(dataset_path):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

check_valid_dir(dataset_path)

#=========1=========2=========3=========4=========5=========6=========7=

#DOES: prints all csv files in "csv_dir" to a single text document
# called "all_csvs_in_one_file.txt". 
def print_files(csv_dir):
    # get a list of all files in the directory
    dir_list = os.listdir(csv_dir)
    # number of files with no valid header
    bad_files = 0
    text_file = open("all_csvs_in_one_file.txt", "w")
    for filename in tqdm(dir_list):
        text_file.write("\n\n\n\n\n\n============================================================================================")
        # get the path of the current file
        path = os.path.join(csv_dir, filename) 
        with open(path, "r") as f:
            # read csv and get the header as a list
            reader = csv.reader(f)
            try:
                header_list = next(reader)
                while True:
                    # if there's only one nonempty attribute, it's
                    # probably just a descriptor of the table, so try the
                    # next line. 
                    header_list = next(reader)
                    text_file.write(str(header_list))
                    text_file.write("\n")
            except StopIteration:
                bad_files = bad_files + 1
                continue
            # throw a key value pair in the dict, with filename as key
    print("Throwing out this number of files, all have less than 10% nonempty cells in every row: ", bad_files)    
    text_file.close()
    return 

def main():
    # MAIN PROGRAM: 
    print_files(dataset_path)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 
