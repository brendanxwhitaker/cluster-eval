from collections import Counter

import pandas as pd
import numpy as np

import sys
import os

# THIS FUNCTION PROVIDES AN ALTERNATE DISTANCE METRIC TO SILHOUETTE
# SCORE. 

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of the frequencydrop scores of each of the clusters,
#          and a total, the average of all these scores. 
def compute_freqdrop_score(list_cluster_lists):

    # list of scores, total score
    freqdrop_scores = []
    freqdrop_total = 0 

    j = 0
    
    # iterate over directory, frequency lists 
    for cluster_list in list_cluster_lists:
         
        freqdrop_score = 0

        # get frequencies of the paths
        path_counts = Counter(cluster_list)
        
        # Create a dataframe from path_counts        
        df = pd.DataFrame.from_dict(path_counts, orient='index')
        
        # rename the frequency axis
        df = df.rename(columns={ df.columns[0]: "freqs" })
        
        # sort it with highest freqs on top
        sorted_df = df.sort_values("freqs",ascending=False)

        # make a new index, use the directory names as a new column
        sorted_df = sorted_df.reset_index()

        # get just the frequencies column
        freq_df = sorted_df.loc[:,'freqs']

        # list of frequency drop values
        diffs = []

        # list of frequencies in descending order
        freq_list = freq_df.values

        # add all frequency differences to the list
        for i in range(len(freq_list) - 1):
            diff = freq_list[i] - freq_list[i + 1]
            diffs.append(diff)

        # find the largest drop in frequency
        max_diff = 0
        max_diff_index = 0
        for i in range(len(diffs)):
            if diffs[i] > max_diff:
                max_diff = diffs[i]
                max_diff_index = i

        # if the largest drop is the first
        if max_diff_index == 0:
            
            # assign worst score
            freqdrop_score = -1        
        else:

            # otherwise, weight by its frequency, we want a small
            # number of frequent clusters 
            freqdrop_score = 1 / (2 ** (max_diff_index - 1))
        freqdrop_scores.append(freqdrop_score)
        print("Frequency drop score for cluster", 
              j, "is: ", freqdrop_score)
        j += 1

    freqdrop_total = sum(freqdrop_scores)/len(freqdrop_scores)
    print("Total frequency drop score: ", freqdrop_total)

    return freqdrop_score

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

