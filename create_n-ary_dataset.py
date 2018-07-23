from tqdm import tqdm
import random
import csv
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

# Generates a new dataset in "dest" called "n-ary_toy_dataset" which
# is a directory hierarchy with n-ary tree structure. The leaf nodes 
# contain 100 text files each, where each text file contains 100 words
# per line and 100 lines. Each word in every text file in a given leaf
# node is the same, taken from a list of seed words.

# A .csv file called "seed_words.csv" should be in the same directory
# as this script for proper functionality.  

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# the directory in which to place the toy dataset
dest = sys.argv[1]
try:
    n = int(sys.argv[2])
    depth = int(sys.argv[3])
except TypeError:
    print("You probably tried to pass a non-integer argument. These"
          + " are supposed to be natural numbers. ")
    exit()

# DOES: checks whether or not a directory argument is valid 
def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
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

# check if destination is valid, get its absolute path
check_valid_dir(dest)
dest = os.path.abspath(dest)
# generate path to the new root of our toy dataset
dataset_name = str(n) + "-ary_toy_dataset"
dataset_path = os.path.join(dest, dataset_name)
# make sure a directory with the same name doesn't already exist
if os.path.isdir(dataset_path):
    print("This directory already exists, change the existing "
          + "directory's name, or try a different destination. ")
    exit()
# create the directory
os.system("mkdir " + dataset_path)

#=========1=========2=========3=========4=========5=========6=========7=

''' DOES: reads from "seed_words.csv" and gets a list of all words in it
    RETURNS: a list of words from the file '''
def read_seed():
    word_list = []
    try:
        with open('../seed_words.csv', 'r') as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rows: 
                for word in row:
                    word_list.append(word)
    except FileNotFoundError:
        print("Make sure you have a seed_words.csv file in the parent"
              + " directory of this folder. ")
        exit() 
    return word_list

#=========1=========2=========3=========4=========5=========6=========7=

def generate_dataset_txt(dataset_path, num_children, depth, word_list):   
 
    if (depth == 1):
        word = random.choice(word_list)
        row_list = []
        for k in tqdm(range(10)):
            row_list.append(word)
        word_string = ' '.join(token for token in row_list) + " "
        text_file_path = os.path.join(dataset_path, word + "_1" + ".txt")
        with open(text_file_path, 'a') as the_file: 
            for j in tqdm(range(100)):
                the_file.write(word_string + "\n")
        for i in tqdm(range(20)):
            if i != 1:
                next_file_path = os.path.join(dataset_path, 
                                              word + "_" 
                                              + str(i) + ".txt")
                os.system("cp " + text_file_path + " " + next_file_path)
    else:
        for i in tqdm(range(num_children)):
            child_path = os.path.join(dataset_path, str(i))
            if not os.path.isdir(child_path):
                os.system("mkdir " + child_path)
            generate_dataset_txt(child_path, 
                                 num_children, depth - 1, word_list)            
            
#=========1=========2=========3=========4=========5=========6=========7=

def generate_dataset_csv(dataset_path, num_children, depth, word_list):
    
    if (depth == 1):
        word = random.choice(word_list)
        row_list = []
        for k in tqdm(range(8)):
            row_list.append(word)
        row_list.append(random.choice(word_list))
        row_list.append(random.choice(word_list))
        word_string = ','.join(token for token in row_list)
        text_file_path = os.path.join(dataset_path, word + "_1" + ".csv")
        with open(text_file_path, 'a') as the_file: 
            for j in tqdm(range(100)):
                the_file.write(word_string + "\n")
        for i in tqdm(range(20)):
            if i != 1:
                next_file_path = os.path.join(dataset_path, 
                                              word + "_" 
                                              + str(i) + ".csv")
                os.system("cp " + text_file_path + " " + next_file_path)
    else:
        for i in tqdm(range(num_children)):
            child_path = os.path.join(dataset_path, str(i))
            if not os.path.isdir(child_path):
                os.system("mkdir " + child_path)
            generate_dataset_csv(child_path, 
                                 num_children, depth - 1, word_list)            
            
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    print("Just the main function. ")
    word_list = read_seed()
    generate_dataset_txt(dataset_path, n, depth, word_list)
    generate_dataset_csv(dataset_path, n, depth, word_list)
    print("Should have worked. If you see a bunch of copy errors, "
          + "make sure there are no spaces between words in your "
          + "seed_words.csv file. ")

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 



