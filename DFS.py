import path_utilities
import numpy as np
import tokenizer
import path_utilities
import sys
import os

# FIRST ARGUMENT: the root directory path (the path of the dataset)

#=========1=========2=========3=========4=========5=========6=========7=

''' RETURNS: a list of the paths of every file in the directory "path". '''
def DFS(path):
    stack = []
    all_paths = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack) - 1)
        # if this is a valid path.
        if(os.path.isdir(tmp)):
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                # for each "item".
                stack.append(os.path.join(tmp, item))
        # if it's not a valid path but it IS a valid file.
        elif(os.path.isfile(tmp)):
            # "tmp" format: '/home/ljung/shit/shitposting/'
            all_paths.append(tmp)
    return all_paths

#=========1=========2=========3=========4=========5=========6=========7=

''' RETURNS: a dictionary which maps extension names of the form "csv"
             to lists of the full paths of files with those extensions in 
             the dataset.'''
def extension_indexer(dataset_path, n, write_path):  
    allpaths = DFS(dataset_path)
    
    # a list of all the filenames (without paths)
    filenames = []
    for path in allpaths:
        filenames.append(path_utilities.get_fname_from_path(path))
    filenames_no_ext, exts = tokenizer.remove_all_extensions(filenames)
    
    sorted_tuple = tokenizer.count_and_sort_exts(exts, n, 
                                                 write_path, dataset_path)
    sorted_exts, sorted_counts = sorted_tuple
    top_n_exts = sorted_exts[:n]
     
    # makes a dictionary key for each of the top extensions
    ext_locations = {}
    for extension in top_n_exts:    
        ext_locations.update({extension:[]})     

    # checks every file and saves the paths of those with the top extensions
    # in a dict called "ext_locations"
    for fp in allpaths:
        fn = path_utilities.get_fname_from_path(fp)
        if fn[:2]!="._":
            ext = path_utilities.get_single_extension(fn)
            if ext in top_n_exts:
                ext_list = ext_locations.get(ext)
                ext_list.append(fp)
                ext_locations.update({ext:ext_list})
  
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)
    ext_write_path = os.path.join(write_path, "extension_index_" 
                                  + dataset_name + ".npy")  
    np.save(ext_write_path, ext_locations)
    
    return ext_locations

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    dataset_path = sys.argv[1]
    num_slices = sys.argv[2]

    # allpaths = DFS(dataset_path)
    
    '''os.chdir(dataset_path)
    f1 = open(dataset_path + "transformed_paths.txt","w")

    f1.write(str(allpaths))
    f1.close()'''

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 
