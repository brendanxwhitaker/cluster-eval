from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# RETURNS: a list of lists, one for each cluster, which contain
#          attribute, count pairs.  
def get_cluster_attributes(filename_header_pairs, labels, 
                           num_clusters, write_path, dataset_name):

    #===================================================================
    #=#BLOCK#=#: Creates "attr_dicts", a list of dicts, one per cluster,
    #            which map unique header attributes to their counts in 
    #            that cluster.   
    #===================================================================
    print("Printing list of dicts which map header attributes to their"
          + " frequencies in the corresponding cluster. ")
    
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
    print("Printing attributes to a .csv file. ")
    concat = concat[0:50]
    concat_df = pd.DataFrame(concat)
    attribute_path = os.path.join(write_path, "top_50_attributes_" 
                                  + dataset_name + "_k=" 
                                  + str(num_clusters) + ".csv") 
    concat_df.to_csv(attribute_path)  
    return clust_attr_lists 
