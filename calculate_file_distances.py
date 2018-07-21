from tqdm import tqdm
import sys
import os
import path_utilities

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: two DIRECTORY paths (not file paths)
    RETURNS: distance between the files in the specified directories '''
def path_dist(path1, path2):
    path1_folders = path1.split("/")
    path2_folders = path2.split("/")
    # remove empty strings
    path1_folders = list(filter(None, path1_folders))
    path2_folders = list(filter(None, path2_folders))
    
    # "shared" contains the shared directory closest to the bottom
    # "common_index" contains the tree depth of "shared", e.g.
    # the common_index of the top level directory is 0. Its 
    # grandchildren have common index 2. 
    shared = ""
    common_index = 0
    min_folderlist_len = min(len(path1_folders), len(path2_folders))
    # iterate over the smaller of the lengths of the folderlists
    for i in range(min_folderlist_len):
        # if the directories match
        if path1_folders[i] == path2_folders[i]:
            # save the directory name in shared, save common index
            shared = path1_folders[i]
            common_index = i
        # once they are no longer equal, stop iterating. 
        else:
            break
    
    # compute distances to closest common ancestor
    p1_dist_to_shared = len(path1_folders)-1-common_index
    p2_dist_to_shared = len(path2_folders)-1-common_index

    total_dist = p1_dist_to_shared + p2_dist_to_shared
    return total_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a path to a directory
    DOES: checks if the directory alone (not its children) has any FILES '''
def has_files(directory):
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            return True
    return False

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a path to a directory
    RETURNS: a list of the folders with files in them in the directory "path" '''
def DFS(path):
    stack = []
    all_dirs = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack) - 1)
        # if this is a directory and has at least one file in it
        if os.path.isdir(tmp):
            if has_files(tmp)==True:
                all_dirs.append(tmp)
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                stack.append(os.path.join(tmp, item))
    return all_dirs

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a directory path
    RETURNS: Naive max distance according to the above distance metric. '''
def naive_max_dist(root):
    max_dist = 0
    all_paths = DFS(root)
    for path_a in tqdm(all_paths):
        for path_b in all_paths:
            dist = path_dist(path_a, path_b) 
            if (max_dist < dist):
                max_dist = dist
    return max_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' DEPRECATED
    PARAMETER: all the paths in a cluster
    RETURNS: the average intracluster distance '''
def intracluster_dist(cluster_paths):
    distances = []
    for i in range(len(cluster_paths)-1):
        path1 = cluster_paths[i]
        path2 = cluster_paths[i+1]
        dirOf_path1 = path_utilities.remove_path_end(path1)
        dirOf_path2 = path_utilities.remove_path_end(path2)
        distances.append(path_dist(dirOf_path1, dirOf_path2))
    dists = np.array(distances)
    return np.mean(dists)

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]
    max_dist = naive_max_dist(root_path)
    print("The max_dist is: ", max_dist)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 










