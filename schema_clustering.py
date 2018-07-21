import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from plot_dendrogram import plot_dendrogram 
from silhouette import compute_silhouette
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
import get_cluster_stats as get_stats
from matplotlib import pyplot as plt
import calculate_file_distances
from collections import Counter
from sklearn import manifold
from tqdm import tqdm
import path_utilities
import pylatex as pl
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
            filename = path_utilities.str_encode(path)

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
    clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                         affinity='precomputed', 
                                         linkage='complete')
    clustering.fit(jacc_matrix)
    labels = clustering.labels_
    #print(labels)

    if (overwrite == 1):
        plt.figure(figsize=(17,9))
        plot_dendrogram(clustering, labels = clustering.labels_)
        dend_path = os.path.join(write_path, 
                                 "dendrogram_" + dataset_name 
                                 + "_k=" + str(num_clusters))
        plt.savefig(dend_path, dpi=300)
        print("dendrogram written to \"dendrogram.png\"")
    
    return labels

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: plots the schema_clusters for the csv files. 
def plot_clusters(jacc_matrix, labels, write_path, 
                  overwrite_plot, dataset_name, num_clusters):
 
    plot_mat_path = os.path.join(write_path, 
                                 "plot_" + dataset_name 
                                 + "_k=" + str(num_clusters) + ".npy")
    if not os.path.isfile(plot_mat_path) or overwrite_plot == "1":
        
        # multidimensional scaling to convert distance matrix into 3D
        mds = manifold.MDS(n_components=3, n_jobs=4, 
                           dissimilarity="precomputed", 
                           random_state=1, verbose=2)
        print("Fitting to the distance matrix. ")
        
        # shape (n_components, n_samples)
        pos = mds.fit_transform(jacc_matrix)
        np.save(plot_mat_path,pos)
    else:
        pos = np.load(plot_mat_path)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    print("Setting up plot. ")
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=labels)) 
    
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    for name, group in tqdm(groups):
            
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], 
                       group.y.iloc[t], 
                       group.z.iloc[t], 
                       c=color, marker='o')
            ax.set_aspect('auto')

    plot_3D_path = os.path.join(write_path, "3D_schema_cluster_" 
                                + dataset_name 
                                + "_k=" + str(num_clusters))
    plt.savefig(plot_3D_path, dpi=300)
    print("scatter plot written to \"3D_schema_cluster.png\"")
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: generates barcharts which show the distribution of unique
#       filepaths in a cluster, prints these as well as stats to a pdf,
#       and also prints the info to a text file as well.  
def generate_results(filename_header_pairs, labels, num_clusters, 
                       dataset_path, write_path, dataset_name):

    #===================================================================
    #=#BLOCK#=#: Generates two data structures: 
    #            "list_cluster_lists": list of lists, each list contains 
    #            the filepaths for one cluster.
    #            "cluster_directories": list of dicts, one per cluster, 
    #            keys are unique directories, values are counts
    #===================================================================
 
    # create a dict mapping cluster indices to lists of filepaths
    cluster_filepath_dict = {}
    
    # list of lists, each list is full of the filepaths for one cluster.
    list_cluster_lists = []
    
    # list of dicts, keys are unique directories, values are counts
    # each list corresponds to a cluster
    cluster_directories = []
    
    # initialize each child list. 
    for k in range(num_clusters):
        list_cluster_lists.append([])
        
        # add k empty dicts
        cluster_directories.append({})    

    # for each label in labels
    for i in tqdm(range(len(labels))):
        
        # get the corresponding filename
        filename_header_pair = filename_header_pairs[i]
        filename = filename_header_pair[0]
        
        # transform "@" delimiters to "/"
        filename = path_utilities.str_decode(filename)
        
        # remove the actual filename to get its directory
        decoded_filepath = path_utilities.remove_path_end(filename)
        
        # get common prefix of top level dataset directory
        common_prefix = path_utilities.remove_path_end(dataset_path)
        
        # remove the common prefix for display on barchart. The " - 1"
        # is so that we include the leading "/". 
        len_pre = len(common_prefix)
        len_decod = len(decoded_filepath)
        decoded_filepath_trunc = decoded_filepath[len_pre - 1:len_decod]
        
        # add it to the appropriate list based on the label
        list_cluster_lists[labels[i]].append(decoded_filepath_trunc)   

    # create a list of dicts, one for each cluster, which map dirs to 
    # counts. 
    for k in range(num_clusters):
        for directory in list_cluster_lists[k]:
            if directory in cluster_directories[k]:
                old_count = cluster_directories[k].get(directory)
                new_count = old_count + 1
                cluster_directories[k].update({directory:new_count})
            else:
                cluster_directories[k].update({directory:1})
    
    #===================================================================
    #=#BLOCK#=#: Prints cluster information to .pdf and .txt files.  
    #===================================================================
    
    # get a list of the cluster statistic for printing to pdf
    cluster_stats = get_stats.get_cluster_stats(cluster_directories)
    
    # compute silhouette coefficients for each cluster (sil_list)
    # and for the entire clustering (sil)
    sil, sil_list = compute_silhouette(cluster_directories,dataset_path)
    l = 0
    for coeff in sil_list:
        print("Silhouette score for cluster " + str(l)+": "+str(coeff))
        l += 1
    print("Total silhouette for entire clustering: ", sil)
    
    # just make font a bit smaller
    matplotlib.rcParams.update({'font.size': 4})
    print("\n\nGenerating barcharts...")
    
    # open the pdf and text files for writing 
    pdf_path = os.path.join(write_path, "tabular_stats_" + dataset_name 
                            + "_k=" + str(num_clusters) + ".pdf")
    txt_path = os.path.join(write_path, "tabular_stats_" + dataset_name 
                            + "_k=" + str(num_clusters) + ".txt")
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    f = open(txt_path,'w')
    
    # for each cluster
    for k in range(num_clusters):
        single_cluster_stats = cluster_stats[k]
        
        #fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 20))
        #plt.figure(k) 
        plt.clf()

        # get frequencies of the paths
        path_counts = Counter(list_cluster_lists[k])
        
        # Create a dataframe from path_counts        
        df = pd.DataFrame.from_dict(path_counts, orient='index')
        
        # rename the frequency axis
        df = df.rename(columns={ df.columns[0]: "freqs" })
        
        # sort it with highest freqs on top
        sorted_df = df.sort_values("freqs",ascending=False)
        top_10_slice = sorted_df.head(10)
        top_10_slice.plot(kind='bar')
        
        # leave enough space for x-axis labels
        # fig.subplots_adjust(hspace=7)

        fig_title = ("Directory distribution for cluster "+str(k)+"\n"
        +"Number of unique directories: " 
        +str(single_cluster_stats[0])+"\n"
        +"Mean frequency: "+str(single_cluster_stats[1])+"\n"
        +"Median frequency: "+str(single_cluster_stats[3])+"\n"
        +"Standard deviation of frequencies: " 
        +str(single_cluster_stats[2])+"\n"
        +"Closest common ancestor of all directories: " 
        +single_cluster_stats[4] + "\n"
        +"Silhouette score: " + str(sil_list[k]))
        plt.title(fig_title)
        plt.xlabel('Directory')
        plt.ylabel('Quantity of files in directory')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, top=0.87)
        pdf.savefig(plt.gcf())

        # print to .txt file as well
        f.write(fig_title)
        f.write("\n\n")
    
    f.write("total_silhouette: " + str(sil))
    f.close()
    pdf.close()
    return
 
#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of lists, one for each cluster, which contain
#          attribute, count pairs.  
def get_cluster_attributes(filename_header_pairs, labels, 
                           num_clusters, write_path, dataset_name):

    #===================================================================
    #=#BLOCK#=#: Creates "attr_dicts", a list of dicts, one per cluster,
    #            which map unique header attributes to their counts in 
    #            that cluster.   
    #===================================================================
    
    # list of dicts, keys are unique attributes, values are counts
    # each list corresponds to a cluster
    attr_dicts = []
    
    # initialize each child list. 
    for k in range(num_clusters):
        
        # add k empty dicts
        attr_dicts.append({})    

    # for each label in labels
    for i in tqdm(range(len(labels))):
        
        # get the corresponding header
        filename_header_pair = filename_header_pairs[i]
        header = filename_header_pair[1]
        
        # for each attribute in this header
        for attribute in header:
            
            # if it's already in this cluster's dict
            if attribute in attr_dicts[labels[i]]:
                old_count = attr_dicts[labels[i]].get(attribute)
                new_count = old_count + 1
                
                # increment the frequency count
                attr_dicts[labels[i]].update({attribute:new_count})
            
            # otherwise, add it to the dict with a count of 1
            else:
                attr_dicts[labels[i]].update({attribute:1})    

    #===================================================================
    #=#BLOCK#=#: Creates "array_list", a list of numpy arrays, each 
    #            array consists of tuples of attributes and frequencies
    #            for that cluster, sorted in descending order.  
    #===================================================================

    # create a list of lists, one for each cluster, containing
    # 2-tuples where the first element is a unique attribute and the 
    # second element is an integer representing its frequency in this
    # cluster
    clust_attr_lists = []
    array_list = []
    max_length = 0
    
    # for every attribute dict created above
    for attr_dict in attr_dicts:
        
        # the list of tuples for this cluster
        clust_attr_list = []
        
        # for each key value pair in this dict
        for attribute, count in attr_dict.items():
            
            # add the corresponding tuple to our list
            clust_attr_list.append([attribute,count])
        
        # sort the list in ascending order by frequency
        clust_attr_list = sorted(clust_attr_list, key=lambda x: x[1])
        
        # find the max length list
        if (max_length < len(clust_attr_list)):
            max_length = len(clust_attr_list)
        
        # add each list to our list of lists
        clust_attr_lists.append(clust_attr_list)
        
        # convert each list to a dataframe
        attr_df = pd.DataFrame(clust_attr_list)
        
        # make it descending order
        sorted_attr_df = attr_df.iloc[::-1]
        
        # convert to numpy array
        sorted_array = sorted_attr_df.values 
        
        # add to list of numpy arrays
        array_list.append(sorted_array)


    #===================================================================
    #=#BLOCK#=#: Turns "array_list" into one big numpy array, with a set
    #            of columns for each cluster. Then prints to csv. 
    #===================================================================

    # this block just adds 0s to each array so they all have the same
    # length, so that we can put them all in a single array called
    # "concat". 
    new_array_list = []
    for array in array_list:
        diff = max_length - array.shape[0]
        if (diff > 0):
            arr = np.zeros(shape=(diff, 2))
            array = np.append(array, arr, axis=0)    
        new_array_list.append(array)

    # create one big array for all clusters, joining all columns
    concat = np.concatenate(new_array_list, axis=1)
    
    # take only the 50 most frequent attributes
    concat = concat[0:50]
    concat_df = pd.DataFrame(concat)
    attribute_path = os.path.join(write_path, "top_50_attributes_" 
                                  + dataset_name + "_k=" 
                                  + str(num_clusters) + ".csv") 
    concat_df.to_csv(attribute_path)  
    return clust_attr_lists 

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM:
def runflow(dataset_path, num_clusters, 
            overwrite, overwrite_plot, fill_threshold):
   
    #===================================================================
    #=#BLOCK#=#: Get read and write paths for cluster functions 
    #===================================================================
    
    # check if the dataset location is a valid directory 
    print("Checking if " + dataset_path + " is a valid directory. ")
    check_valid_dir(dataset_path)
   
    # get its absolute path
    print("Getting the absolute path to the dataset. ") 
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    print("Getting the name of the top-level directory of the "
          + "dataset. ")
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)
    
    # Get converted file location and output location
    print("Getting the location of any and all converted tabular" 
          + "files. ")
    out_dir = os.path.join(dataset_path, 
                           "../" + "converted-" + dataset_name)
    print("All results printing to " + write_path)
    write_path = "../outputs/" + dataset_name + "--output/"
    
    # Get absolute paths 
    out_dir = os.path.abspath(out_dir)
    write_path = os.path.abspath(write_path)
    
    # get the location of the extension index file
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
    plot_clusters(jacc_matrix, labels, write_path, 
                  overwrite_plot, dataset_name, num_clusters)

    # generate results in pdf and text files
    generate_results(filename_header_pairs, labels, num_clusters, 
                     dataset_path, write_path, dataset_name)

    # get a table of the most common attributes in each cluster
    clust_attr_lists = get_cluster_attributes(filename_header_pairs, 
                                              labels, 
                                              num_clusters,
                                              write_path, 
                                              dataset_name) 
    return

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

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

