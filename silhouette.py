import calculate_file_distances

# NOT NORMALIZED YET, SHOULD NORMALIZE IT
def tree_dist(path1, path2, max_dist):
    return calculate_file_distances.path_dist(path1, path2) / max_dist

# RETURNS:  Silhouette coefficient of clustering, and list of the
#           silhouette coeffs for each of the clusters.
def compute_silhouette(cluster_directories, dataset_path):

    if len(cluster_directories)<2:
        print("Need at least 2 clusters to computer silhouette score.")
        return 0

    max_dist = calculate_file_distances.naive_max_dist(dataset_path)
    # print("max_dist: ", max_dist)
    silhouette_list = []
    total_num_files = 0
    total_silhouette_sum = 0
    # print("Calculating silhouette scores...")
    for cluster_i in cluster_directories:
        num_files_i = 0
        silhouette_sum_i = 0
        for dir_x, count_x in cluster_i.items():
            # COMPUTE INTERCLUSTER DISTANCE
            # for each point
            set_of_means_x = []
            for cluster_j in cluster_directories:
                if (cluster_j != cluster_i):
                    total_count_j = 0
                    total_dist_j = 0
                    for dir_y, count_y in cluster_j.items():
                        # Make sure this function normalizes
                        # get the distance between the two dirs
                        xy_dist = tree_dist(dir_x, dir_y, max_dist)
                        # print("distance between ", dir_x," and ", dir_y," :", xy_dist)
                        # multiply by the y count to add to the sum
                        # since we have "count_y" files in this dir
                        # to take dist with. 
                        xy_dist *= count_y
                        # add this quantity to the running sum so we
                        # can compute the mean for x,j.
                        total_dist_j += xy_dist
                        # keep track of total count so we can divide 
                        # by it when computing mean.  
                        total_count_j += count_y
                    if (total_count_j == 0):
                        total_count_j = 1
                    mean_x_j = total_dist_j / total_count_j
                    set_of_means_x.append(mean_x_j)
            min_mean_x = min(set_of_means_x)
            # print("min_mean_x: ", min_mean_x)
            
            # COMPUTE INTRACLUSTER DISTANCE
            total_count_i = 0
            total_dist_i = 0
            for dir_y, count_y in cluster_i.items():
                if (dir_y != dir_x):
                    xy_dist = tree_dist(dir_x, dir_y, max_dist)
                    # print("distance between ", dir_x," and ", dir_y," :", xy_dist)
                    # multiply by the y count to add to the sum
                    # since we have "count_y" files in this dir
                    # to take dist with. 
                    xy_dist *= count_y
                    # add this quantity to the running sum so we
                    # can compute the mean for x,j.
                    total_dist_i += xy_dist
                    # keep track of total count so we can divide 
                    # by it when computing mean.  
                    total_count_i += count_y
            # get the mean of the distances between x and the other 
            # files in cluster i
            if (total_count_i == 0):
                total_count_i = 1
            mean_x_i = total_dist_i / total_count_i
            # print("mean_x_i: ", mean_x_i)
            # compute the denominator of silhouette for x
            silhouette_x_denom = max([min_mean_x, mean_x_i])
            # print("denom: ", silhouette_x_denom)
            # compute silhouette of x
            if (silhouette_x_denom == 0):
                silhouette_x_denom = 1
            silhouette_x = (min_mean_x - mean_x_i) / silhouette_x_denom
            # print("silhouette_x: ", silhouette_x)
            # add the silhouette for x to our sum of all silhouettes in
            # cluster i 
            silhouette_sum_i += silhouette_x * count_x
            # count number of files in cluster i
            num_files_i += count_x
        # compute average silhouette for cluster i
        if (num_files_i == 0):
            num_files_i = 1
        silhouette_i = silhouette_sum_i / num_files_i
        # print("silhouette_sum_i: ", silhouette_sum_i)
        # print("num_files_i: ", num_files_i)
        # keep track of the total number of files in the dataset
        total_num_files += num_files_i
        # keep track of the total silhouette sum for all clusters
        total_silhouette_sum += silhouette_sum_i
        # add the silhouette coeff for cluster i to our list of coeffs
        silhouette_list.append(silhouette_i)
    # compute the total silhouette coeff for whole clustering
    if (total_num_files == 0):
        total_num_files = 1
    total_silhouette = total_silhouette_sum / total_num_files
  
    return total_silhouette, silhouette_list
