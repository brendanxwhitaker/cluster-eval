from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

from path_utilities import get_last_dir_from_path 
from attributes import get_cluster_attributes
from generate_results import generate_results
from plot_dendrogram import plot_dendrogram 
from plot_clusters import plot_clusters
from path_utilities import str_encode

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from tqdm import tqdm

import pandas as pd
import numpy as np
import sklearn
import pickle
import sys
import csv
import os
import re

np.set_printoptions(threshold=np.nan)

# Converts all the .xls or .xlsx files in a directory to .csv files. 
# Then it clusters the schemas of these .csv files using agglomerative
# clustering. 

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():

    print("Parsing arguments. ")
    # ARGUMENTS    
    # source directory and output directory
    dataset_path = sys.argv[1]          # directory of dataset
    num_clusters = int(sys.argv[2])     # number of clusters to generate
    fill_threshold = float(sys.argv[3]) # ignore rows filled less
    overwrite = sys.argv[3]             # overwrite the distance matrix
    overwrite_plot = sys.argv[4]        # overwrite plot cache 
    # overwrite is a string, should be "0" for don't overwrite, and "1"
    # for do
    arg_list = [
                dataset_path, 
                num_clusters, 
                overwrite, 
                overwrite_plot, 
                fill_threshold,
               ]
    print("Arguments parsed. ")
    return arg_list

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

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict(csv_dir, csv_path_list, 
                    fill_threshold, converted_status):
    
    # maps filenames to csv header lists
    print("Generating structured header dictionary. ")
    header_dict = {}
    
    # number of files with no valid header
    bad_files = 0
    
    # number of decoding errors while reading csvs
    decode_probs = 0

    # This code is rather confusing because I wanted the function to 
    # be able to handle both types of inputs (lists of paths in names)
    # and just directory locations. 

    # CASE 1:
    # If we're reading in converted files, we only need the csv_dir
    # argument, so we get a list of the filenames from that directory. 
    # These filenames are in the form:
    # "@home@ljung@pub8@oceans@some_file.csv"
    if (converted_status):
        dir_list = os.listdir(csv_dir)

    # CASE 2:
    # Otherwise, we are reading in a list of the true, original 
    # locations of files that were csvs to begin with in the dataset.
    else:
        dir_list = csv_path_list
  
    # CASE 1: "path" looks like:"@home@ljung@pub8@oceans@some_file.csv" 
    # CASE 2: "path" is literally the path of that file in the original
    # dataset as a string. 
    for path in tqdm(dir_list):
        if (converted_status): 
            
            # get the new location of the current file in "csv_dir", 
            # i.e. not in original dataset. 
            filename = path
            path = os.path.join(csv_dir, path) 
        else:
            
            # convert to "@home@ljung@pub8@oceans@some_file.csv" form. 
            filename = str_encode(path)

        # So now in both cases, filename has the "@"s, and path is
        # the location of some copy of the file. 
        with open(path, "r") as f:
            
            # read csv and get the header as a list
            reader = csv.reader(f)
            try:
                header_list = next(reader)
                
                # if the header is empty, try the next line
                if (len(header_list) == 0):
                    header_list = next(reader)
                 
                # number of nonempty attribute strings
                num_nonempty = 0
                for attribute in header_list:
                    if not (attribute == ""):
                        num_nonempty = num_nonempty + 1
                fill_ratio = num_nonempty / len(header_list)                

                # keep checking lines until you get one where there
                # are enough nonempty attributes
                while (fill_ratio <= fill_threshold):
                    
                    # if there's only one nonempty attribute, it's
                    # probably just a descriptor of the table, so try
                    # the next line. 
                    header_list = next(reader)
                    num_nonempty = 0
                    for attribute in header_list:
                        if not (attribute == ""):
                            num_nonempty = num_nonempty + 1
                    if (len(header_list) == 0):
                        fill_ratio = -1
                    else:
                        fill_ratio = num_nonempty / len(header_list)
                    
                    #===================================================
                    # Here we've hardcoded some information about 
                    # scientific data to work better with CDIAC. 
                    # feel free to remove it. 
                    
                    # people seem to denote pre-header stuff with a *
                    for attribute in header_list:
                        if (attribute != "" and attribute[-1] == "*"):
                            fill_ratio = -1
                    if (len(header_list) > 3):
                        if (header_list[0] == "Year" 
                            and header_list[2] != ""):
                            break
                        if (header_list[0] == "Citation"):
                            fill_ratio = -1
                    #===================================================
    
            except UnicodeDecodeError:
                decode_probs = decode_probs + 1                    
            except StopIteration:
                bad_files = bad_files + 1
                #os.system("cp " + path + " ~/bad_csvs/")
                continue
            
            # throw a key value pair in the dict, with filename as key
            header_dict.update({filename:header_list})
    print("Throwing out this number of files, all have less than ", 
          fill_threshold*100, 
          "% nonempty cells in every row: ", bad_files)    
    print("Number of UnicodeDecodeErrors: ", decode_probs)
    print("Dictionary generated. ")
    return header_dict

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: Jaccard distance between two lists of strings. 
def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if (union == 0):
        union = 1
    return 1 - float(intersection / union)
    
#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: computes the jaccard distance matrix of the headers in 
# header_dict. 
# RETURNS: a tuple with the first element being an array of all the 
# headers in numpy array form, the second being the jaccard dist
# matrix, and the third being a list of 2-tuples (filename, header_list)
def dist_mat_generator(header_dict, 
                       write_path, overwrite, dataset_name):

    #===================================================================
    #=#BLOCK#=#: Get paths, read from header_dict, and initialize stuff. 
    #===================================================================
    print("Getting write paths for the distance matrix. ")   
 
    # Define the names for the files we write distance matrix and the
    # filename_header_pairs list to.  
    dist_mat_path = os.path.join(write_path, "dist_" 
                                 + dataset_name + ".npy")
    headpairs_path = os.path.join(write_path, 
                                  "headpairs_" + dataset_name + ".pkl")

    # list of all header_lists
    header_lists = []
    
    # list of tuples, first element is filename, second is header_list
    filename_header_pairs = []

    for filename, header_list in header_dict.items():
        header_lists.append(header_list)
        filename_header_pairs.append([filename, header_list])
   
    # we just need an empty numpy array 
    jacc_matrix = np.zeros((2,1))
    
    #===================================================================
    #=#BLOCK#=#: Regenerate and overwrite the Jaccard distance matrix 
    #            and save to a file, or else, read old one from file.  
    #===================================================================

    if (not os.path.isfile(dist_mat_path) or 
        not os.path.isfile(headpairs_path) or overwrite == "1"):
        
        print("No existing cached files for this directory. ")
        print("Generating distance matrix using jaccard similarity. ")
        print("This could take a while... ")
        
        # we generate the distance matrix as a list
        dist_mat_list = []
        #j = 0
        
        # iterating over the header array once...
        for header_a in tqdm(header_lists):
            #===========================
            #print(header_a)
            #print(filename_header_pairs[j][0])
            #j = j + 1
            #===========================
            
            # storing distances for a single header
            single_row = []
            
            # iterating again...
            for header_b in header_lists:
                jacc = jaccard_distance(header_a, header_b)
                single_row.append(jacc)
            
            # add one row to the list
            dist_mat_list.append(np.array(single_row))
        
        # convert list to numpy array
        jacc_matrix = np.array(dist_mat_list)
        jacc_matrix = np.stack(jacc_matrix, axis=0)
        print(jacc_matrix.shape)
        
        # save on disk, because computation is expensive
        print("Saving file to: ", dist_mat_path)
        np.save(dist_mat_path, jacc_matrix)
        with open(headpairs_path, 'wb') as f:
            pickle.dump(filename_header_pairs, f)

    else:
        print("There is an existing distance matrix for this dataset. ")
        print("Loading file from: ", dist_mat_path)
        jacc_matrix = np.load(dist_mat_path)
        with open(headpairs_path, 'rb') as f:
            filename_header_pairs = pickle.load(f)
                
    return jacc_matrix, filename_header_pairs

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: the labels from the agglomerative clustering. 
def agglomerative(jacc_matrix, 
                  num_clusters, 
                  filename_header_pairs, 
                  overwrite,
                  write_path,
                  dataset_name):

    print("Initializing sklearn agglomerative clustering object. ")
    clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                         affinity='precomputed', 
                                         linkage='complete')

    print("Fitting model to the distance matrix. ")
    clustering.fit(jacc_matrix)
    labels = clustering.labels_

    if (overwrite == 1):
        print("Replotting dendrogram. ")
        plt.figure(figsize=(17,9))
        plot_dendrogram(clustering, labels = clustering.labels_)
        dend_path = os.path.join(write_path, 
                                 "dendrogram_" + dataset_name 
                                 + "_k=" + str(num_clusters))
        plt.savefig(dend_path, dpi=300)
        print("Dendrogram written to \"dendrogram.png\"")
    
    return labels

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM:
def runflow(dataset_path, num_clusters, 
            overwrite, overwrite_plot, fill_threshold):
   
    #===================================================================
    #=#BLOCK#=#: Get read and write paths for cluster functions 
    #===================================================================
    print("Getting read and write paths for cluster functions. ")  
 
    if overwrite == 'y' or overwrite == 'Y':
        overwrite = "1"
    if overwrite_plot == 'y' or overwrite_plot == 'Y':
        overwrite_plot = "1"
 
    # check if the dataset location is a valid directory 
    check_valid_dir(dataset_path)
   
    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    dataset_name = get_last_dir_from_path(dataset_path)
    
    # get converted file location and output location
    out_dir = os.path.join(dataset_path, 
                           "../" + "converted-" + dataset_name)
    
    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute paths 
    out_dir = os.path.abspath(out_dir)
    write_path = os.path.abspath(write_path)
    
    # get the location of the extension index file
    print("Finding extension index file. ")
    ext_dict_file_loc = os.path.join(write_path, "extension_index_"
                                     + dataset_name + ".npy")
    # check if the above paths are valid
    check_valid_dir(out_dir)
    check_valid_file(ext_dict_file_loc)
    
    # load the extension to path dict
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    
    # location of files converted to csv format
    csv_dir = os.path.join(out_dir, "csv/")

    #===================================================================
    #=#BLOCK#=#: Generates the files needed for clustering, clusters,
    #            and and prints various results. 
    #===================================================================
    
    # if csvs have less than fill_threshold*100% nonempty cells in 
    # every row then we throw them out of our clustering. 
    
    # we have two dicts, one made up of files which were converted to
    # csv format, and the other made up of files that were in csv
    # format originally. we concatenate both dicts into "header_dict".

    # Get the combined header dict
    header_dict_converted = get_header_dict(csv_dir, [],  
                                            fill_threshold, True)
    header_dict_csv = get_header_dict("", csv_path_list,
                                      fill_threshold, False) 
    header_dict = dict(header_dict_converted)
    header_dict.update(header_dict_csv)

    # Get the file/header array, distance matrix
    dist_tuple = dist_mat_generator(header_dict, write_path, 
                                    overwrite, dataset_name)
    jacc_matrix, filename_header_pairs = dist_tuple
    
    # cluster, generate labels
    labels = agglomerative(jacc_matrix, num_clusters, 
                           filename_header_pairs, 
                           overwrite_plot, write_path, dataset_name)

    # plot in 3D
    print("Plotting clusters in R^3. ")
    plot_clusters(jacc_matrix, labels, write_path, 
                  overwrite_plot, dataset_name, num_clusters)
 
    # generate results in pdf and text files
    print("Generating results. ")
    list_cluster_lists = generate_results(filename_header_pairs, 
                                          labels, 
                                          num_clusters, 
                                          dataset_path, 
                                          write_path, 
                                          dataset_name)

    # get a table of the most common attributes in each cluster
    print("Getting cluster attributes. ")
    clust_attr_lists = get_cluster_attributes(filename_header_pairs, 
                                              labels, 
                                              num_clusters,
                                              write_path, 
                                              dataset_name) 
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    arg_list = parse_args()
    dataset_path = arg_list[0]
    num_clusters = arg_list[1]
    overwrite = arg_list[2]
    overwrite_plot = arg_list[3]
    fill_threshold = arg_list[4]

    print("Don't run this file standalone. ")
    runflow(dataset_path, num_clusters, 
                    overwrite, overwrite_plot, fill_threshold)    
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

