import numpy as np
import os

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS: a list of all the paths
# RETURNS: the closest common ancestor of all paths in "filepaths"
def closest_common_ancestor(filepaths):
    path1 = filepaths[0]
    # list of all folders in the first path
    path1_folders = path1.split("/")
    # list of the folders that make up the path to the closest common
    # ancestor so far, we initialize it to the path to the first dir
    # in the list, for the case where "filepaths" has length 1. 
    common_ancestor_list = path1_folders
    # for every other path, call them path2
    for i in range(1,len(filepaths)):
        path2 = filepaths[i]
        # list all folders in path2
        path2_folders = path2.split("/")
        # the smaller of the two depths
        min_folderlist_len = min(len(path1_folders), len(path2_folders))
        for j in range(min_folderlist_len):
            # if the folder matches
            if path1_folders[j]==path2_folders[j]:
                # save the shared path, goes to j + 1 since we want 
                # the deepest folder which is a common ancestor, and 
                # this is the jth folder (indexing not inclusive on 
                # the right). 
                common_ancestor_list = path1_folders[0:j + 1]
            # once they are no longer equal, stop iterating. 
            else:
                break
    # print("Returning: ", common_ancestor_list)
    return common_ancestor_list

#=========1=========2=========3=========4=========5=========6=========7=

#"cluster_directories" is a list of lists, the ith list is a list of the
# directories from the ith cluster. 
# RETURNS: cluster stats
def get_cluster_stats(cluster_directories):
    cluster_stats = []
    # print("Number of unique directories in:")
    for i in range(len(cluster_directories)):
        # will be a list containing in order: cluster size, avg freq,
        # median freq, std-dev, closest common ancestor as string.
        single_cluster_stats = []
        sum = 0
        # print("Cluster", i, ":", len(cluster_directories[i]))
        dir_counts = np.array(list(cluster_directories[i].values()))
        # for key, value in cluster_directories[i].items():
        # sum = sum + value
        # print("Avg dir frequency:", sum/len(cluster_directories[i]))
        # print("Avg dir frequency:", np.mean(dir_counts))
        # print("Std-dev dir frequency:", np.std(dir_counts))
        # print("Median frequency:", np.median(dir_counts))
        # Note that closest_common_ancestor returns a list of the 
        # individual directories in the path, so "/".join concatenates
        # them with "/"s. 
        closest_com_ancest = "/".join(closest_common_ancestor(list(cluster_directories[i].keys())))
        # print("closest common ancestor directory:", closest_com_ancest)
        single_cluster_stats.append(len(cluster_directories[i]))
        single_cluster_stats.append(np.mean(dir_counts))
        single_cluster_stats.append(np.std(dir_counts))
        single_cluster_stats.append(np.median(dir_counts))
        single_cluster_stats.append(closest_com_ancest)
        
        cluster_stats.append(single_cluster_stats)
    return cluster_stats        

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    print("This file is just for importing functions, don't run it. ")    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main() 

