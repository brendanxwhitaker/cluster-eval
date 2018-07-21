from calculate_file_distances import get_dir_from_path
import numpy as np
import DFS
import sys
import os

# DOESN'T LIKE FILES WITH SPACES IN THE FILENAME, will skip. 

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
dataset_path = sys.argv[1]

#=========1=========2=========3=========4=========5=========6=========7=

#FUNCTIONS
# DOES: randomly shuffles the location of the files in 
#       "dataset_path". 
def shuffle(dataset_path):
    if (confirm(prompt="Warning, this will scramble the directory "
                + "structure of all files and folders in " + dataset_path
                + ". Are you sure you want to do this? ")):
        print("Ok.")
    else: exit()    
    if (confirm(prompt="Really sure, though?")):
        print("Ok.")
    else: exit()
    if (confirm(prompt="Super duper sure???")):
        print("Ok.")
    else: exit()
    # get a list of the paths to every file in the dataset
    # rooted at "dataset_path"
    filepaths = DFS.DFS(dataset_path)
    num_files = len(filepaths)
    # list of the parent directories of every file in 
    # "filepaths". 
    directory_list = []
    # for each file
    for filepath in filepaths:
        # get its parent directory
        directory = get_dir_from_path(filepath)
        # and add it to our list of parent directories
        directory_list.append(directory)
    # generate a permutation of the number of files
    perm = np.random.permutation(num_files)
    # for each index
    for i in range(num_files):
        # get the image of the index under our permutation
        permuted_index = perm[i]
        # get the file we're moving
        next_file = filepaths[i]
        # get the randomly chosen destination directory
        dest_dir = directory_list[permuted_index]
        # move the file
        print(next_file)
        os.system("mv " + next_file + " " + dest_dir)

#=========1=========2=========3=========4=========5=========6=========7=

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n: 
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: 
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """
    
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
        
    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print ('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

#=========1=========2=========3=========4=========5=========6=========7=

#MAIN FUNCTION
def main():
    print("Just the main function. ")
    shuffle(dataset_path)
  
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 
