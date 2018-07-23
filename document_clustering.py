import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from glob import glob
from unipath import Path
from six import string_types
import nltk, re, os, codecs, mpld3, sys, random
from nltk.stem.snowball import SnowballStemmer

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

import silhouette
import path_utilities

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAM: "directory" - a parent directory containing pdf/ doc/ etc
           "dataset_path" - a directory leading to the dataset
    RETURNS: a list of filenames and a list of the contents of the files 
    DOES: gets all the filenames and their contents of a directory '''   
def get_document_contents(directory, dataset_path):
    dataset_name, file_place = initialize_output_location(dataset_path)
    
    # load in the extension index file from the output folder
    ext_dict_file_loc = os.path.join(file_place,"extension_index_" + dataset_name + ".npy") 
    ext_paths = np.load(ext_dict_file_loc).item()
    
    # "filenames" list of the paths of files
    # "data" list of the contents of files 
    filenames = []
    data = []
    i = 1
    
    # get contents of txt files still in original dataset
    txt_paths = ext_paths.get("txt")
    print("Getting .txt contents from " + dataset_path)
    for path in tqdm(txt_paths):
        if os.path.isfile(path):
            i = i+1
            # add the path of the file to "filenames" 
            filenames.append(path)

            # read the contents of the file and remove newlines
            fread = open(path, "r", errors='backslashreplace')
            contents = fread.read()
            fread.close()
            contents = contents.replace("\n","")
            # add the string of the contents of the file to "data"
            data.append(contents)
    
    # get contents of converted files in the other directory
    conv_folders = path_utilities.get_immediate_subdirectories(directory)
    # for each folder in the directory (e.g. pdf/ doc/)
    for folder in conv_folders:
        filetype = path_utilities.get_last_dir_from_path(folder)
        if filetype in ["pdf", "doc", "docx"]: #, "xml", "html"]:
            print("Getting ."+folder+" contents")
            for filename in tqdm(os.listdir(folder)):
                cur_file = os.path.join(folder,filename)
                if os.path.isfile(cur_file):
                    i = i + 1
                    # add the non-converted filename to "filenames" 
                    new_name = path_utilities.str_decode(path_utilities.remove_extension(filename))
                    filenames.append(new_name)

                    # read the contents of the file and remove newlines
                    fread = open(cur_file, "r", errors='backslashreplace')
                    contents = fread.read()
                    fread.close()
                    contents = contents.replace("\n","")
                    # add the string of the file contents to "data"
                    data.append(contents)
    
    print("Num total files: ", i)
    print("All directory contents retrieved")
    return filenames, data

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: the text of a document
    RETURN: list of filtered tokens and a list of stems
    DOES: splits a document into a list of tokens & stems each token '''
def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    # tokenize by sentence, then by word so punctuation is its own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # filter out tokens without letters (e.g., numbers, punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            # removes tokens with length 1
            if len(token)>1:
               filtered_tokens.append(token)
    
    # stems the tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    
    return filtered_tokens, stems

''' PARAMETER: the text of a document
    RETURN: a list of stems
    DOES: for use with Tfidf Vectorizer '''
def tokenize_and_stem_call(text):
    ft, stems = tokenize_and_stem(text)
    return stems

''' PARAMETER: the path leading to the dataset
    RETURNS: the name of the dataset and the output path '''
def initialize_output_location(dataset_path):
    # setup output path as "file_place" outside the repo
    p = Path(Path(os.getcwd()).parent).parent
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)
    file_place = os.path.join(p, "cluster-datalake-outputs/", dataset_name + "--output")    
    
    if not os.path.isdir(file_place):
        os.mkdir(file_place)
    return dataset_name, file_place
    
#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: parameters explained in main_function
    RETURNS: a list of filenames and a list of file data
    DOES: retokenizes and catches errors '''
def to_retokenize(retokenize, corpusdir, dataset_path): 
    dataset_name, file_place = initialize_output_location(dataset_path)
    
    # "fnames" is a list of the paths of each file in the  source dataset
    # "dataset" is a list of strings, each containing a document's text
    fnames, dataset = get_document_contents(corpusdir, dataset_path)
    
    if retokenize == "0":
        print("\nAttempting to bypass retokenizing...")
        vf = os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl")
        tms = os.path.join(file_place, "terms_" + dataset_name + ".npy")
        dm = os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy")
        tf = os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy")

        if not os.path.isfile(vf) or not os.path.isfile(tms) or not os.path.isfile(dm) or not os.path.isfile(tf):
            print("One or more dependencies is missing... \nRetokenizing...")
            retokenize = "1"
            
    if retokenize == "1":     
        print("\nTokenizing", len(dataset), "documents...")
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in tqdm(dataset):
            # for each item in the dataset, tokenize and stem
            allwords_tokenized, allwords_stemmed = tokenize_and_stem(i)
            # extend "totalvocab_stemmed" and "totalvocab_tokenized"
            totalvocab_stemmed.extend(allwords_stemmed) 
            totalvocab_tokenized.extend(allwords_tokenized)

        # create vocab_frame with "totalvocab_stemmed" or "totalvocab_tokenized"
        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, 
                    index = totalvocab_stemmed)
        vocab_frame.to_pickle(os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl"))
        print('There are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                              min_df=1, stop_words='english', use_idf=True, 
                              tokenizer=tokenize_and_stem_call, ngram_range=(1,3))
        
        #fits the vectorizer to the dataset
        print("\nFitting vectorizer to data...")
        tfidf_matrix = tfidf_vectorizer.fit_transform(dataset) 
        print("matrix", tfidf_matrix)
        np.save(os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy"), tfidf_matrix)
        terms = tfidf_vectorizer.get_feature_names()
        print("length ofh t erms", len(terms))
        np.save(os.path.join(file_place, "terms_" + dataset_name + ".npy"), terms)
        dist = 1 - cosine_similarity(tfidf_matrix)
        np.save(os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy"), dist)
        print("Vectorizer fitted to data")
        
    return fnames, dataset
      
#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: parameters explained in main_function
    DOES: reclusters and catches errors '''
def to_recluster(num_clusters, retokenize, recluster, tfidf_matrix, dataset_path, minibatch):
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
   
    if recluster == "0":
        print("\nAttempting to bypass reclustering...") 
        dc = os.path.join(file_place, "doc_cluster_" + trailer_text + ".pkl")
        
        if not os.path.isfile(dc):
            print("\"doc_cluster_" + trailer_text + ".pkl\" is missing... \nReclustering...")
            recluster = "1"

        if retokenize == "0":
            print("\nNo output if not retokenizing or reclustering. Please try again. \nExiting...")
            exit()
    
    if retokenize == "1":
        recluster = "1"

    if recluster == "1":
        clustert0 = time()
        # cluster using KMeans on the tfidf matrix
        print("\nClustering using kmeans with k = " + str(num_clusters) + "...")
        
        if minibatch == "1":
            km = MiniBatchKMeans(n_clusters=num_clusters)
        else:
            km = KMeans(n_clusters=num_clusters, n_jobs=-1)
        
        km.fit(tfidf_matrix)
        print("Kmeans clustering complete")
        print("Time to cluster:", time()-clustert0)    
    
        # pickle the model, reload the model/reassign the labels as the clusters
        joblib.dump(km, os.path.join(file_place, 'doc_cluster_' + trailer_text + '.pkl'))

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: "num_clusters" - the number of clusters to cluster into
                "retokenize" - 1 if to retokenize, 0 to not
                "corpusdir" - the directory of converted files
                "dataset_path" - directory of original dataset
                "n_words" - number of words/cluster to print out 
    RETURNS: "frame" - dataframe containing clusters and file paths
             "all_cluster_words" - list of lists of top words in cluster
             "distinct_cluster_labels" - list of distinct cluster labels '''
def main_function(num_clusters, retokenize, recluster, corpusdir, dataset_path, n_words, minibatch):
    #stopwords = nltk.download('stopwords')

    #nltk.download('punkt')
    stemmer = SnowballStemmer("english")

    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    
    print("\nAll outputs generated will be in \"~\\cluster-datalake-outputs\\" + dataset_name + "--output\"")

    #=========1=========2=========3=========4=========5=========6=======
    
    fnames, dataset = to_retokenize(retokenize, corpusdir, dataset_path)
    tfidf_matrix = np.load(os.path.join(file_place, "tfidf_matrix_" + dataset_name + ".npy")).item()
    to_recluster(num_clusters, retokenize, recluster, tfidf_matrix, dataset_path, minibatch)
    
    km = joblib.load(os.path.join(file_place, 'doc_cluster_' + trailer_text + '.pkl'))
    vocab_frame = pd.read_pickle(os.path.join(file_place, "vocab_frame_" + dataset_name + ".pkl"))
    terms = np.load(os.path.join(file_place, "terms_" + dataset_name + ".npy")).tolist()
    dist = np.load(os.path.join(file_place, "distance_matrix_" + dataset_name + ".npy"))
    print("\nLoaded in existing cluster profile...\n")

    clusters = km.labels_.tolist()
    # get the actual number of clusters in the dataframe, in case some
    # are omitted for some reason
    distinct_cluster_labels = []
    for label in clusters:
        if label not in distinct_cluster_labels:
            distinct_cluster_labels.append(label)
    
    # create a dictionary "db" of filenames, contents, and clusters
    db = {'filename': fnames, 'content': dataset, 'cluster': clusters}
    # convert "db" to a pandas dataframe
    frame = pd.DataFrame(db, index=[clusters], columns=['filename','cluster'])
    # print the number of files in each cluster
    #print("Number of files in each cluster: ")
    #print(frame['cluster'].value_counts())

    #=========1=========2=========3=========4=========5=========6=======
   
    # open file writer for result output
    fwriter = open(os.path.join(file_place, "doc_clusters_" + trailer_text + ".txt"), "w")
    fwriter.write("Clusters from text files in: " + corpusdir)

    fwriter.write("\nTop terms per cluster: \n\n")
    print("Top terms per cluster: \n")

    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
   
    #print("Printing the index. ")
    #print(frame.index)
    #print(vocab_frame.to_string())   
    #print("true number of clusters: ", true_num_clusters)

    all_cluster_words = {}
    # for each cluster

    ''' terms contains all the feature labels of the clustering
        vocab_frame contains all the tokens mapped to their stemmed counterparts
        you're finding the token version of the stem you get
        the stem you get is at the ind position of the terms list
        ind is from order_centroids
        order_centroids is a sorted array with num_clusters rows and len(terms) features
        order_centroids[i] is the coordinates of cluster i
        order_centroids[i,:] is the coordinates from ALL features for cluster i'''
         
    distinct_cluster_labels = sorted(distinct_cluster_labels)
    for i in distinct_cluster_labels:
        fwriter.write("Cluster " + str(i) + " words: ")
        print("Cluster %d words:" % i, end='')
        
        cluster_words = []
        for ind in order_centroids[i, :]: 
            test_var = vocab_frame.ix[terms[ind].split(" ")].values.tolist()[0]
            cluster_words.append(test_var[0])
        cluster_words = unique(cluster_words)
        
        # print the first "n_words" words in a cluster
        #for ind in order_centroids[i, : n_words]:
        for ind in range(min(n_words, len(cluster_words))):
            #print("fuck mylife -  ")
            #test_var = vocab_frame.ix[terms[ind].split(" ")].values.tolist()[0]
            print(' %s' % cluster_words[ind], end=",")  
            fwriter.write(cluster_words[ind].rstrip("\n") + ", ")
            #print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=",")
            #fwriter.write(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].rstrip('\n') + ", ")
            #cluster_words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0])
        print()
        fwriter.write("\n")
         
        all_cluster_words.update({i:cluster_words})

        # print out the filenames in the cluster
        print("Cluster %d filenames:" % i, end='')
        fwriter.write("Cluster " + str(i) + " filenames: ")
        for filename in frame.loc[i]['filename'].values.tolist():
            print(' %s,' % filename, end='')
            fwriter.write(filename.rstrip('\n') + ", ")
        print("\n")
        fwriter.write("\n\n") 

    fwriter.close()
    print("Output written to \"doc_clusters_" + trailer_text + ".txt\"")

    #=========1=========2=========3=========4=========5=========6========
    
    # multidimensional scaling: convert distance matrix into 3-dimensions
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
    print("\nFitting the distance matrix into 3 dimensions...")
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, and filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=clusters, filename=fnames)) 
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    print("\n\nPlotting scatterplot of cluster points...")
    for name, group in tqdm(groups):
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], group.y.iloc[t], group.z.iloc[t], 
                c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig(os.path.join(file_place, "3D_document_cluster_" + trailer_text + ".png"), dpi=300)
    print("Scatter plot written to \"3D_document_cluster_" + trailer_text + ".png\"")
           
    return frame, all_cluster_words, distinct_cluster_labels

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETERS: "frame" - a dataframe containing which files are in which clusters
                "num_clusters" - the number of clusters
                "home_dir" - the parent directory of the dataset (e.g. /home/ljung/)
                "dataset_path" - the directory of the dataset (e.g. /home/ljung/pub8/)
    DOES: plots a pdf with all the bar charts on it
          each bar chart shows which directories the files in a cluster come from '''
def bar_clusters(frame, distinct_cluster_labels, num_clusters, home_dir, dataset_path):
    plt.figure("bar")
  
    matplotlib.rcParams.update({'font.size': 4})
    
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    
    print("\n\nGenerating barcharts...")    
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(file_place, "text_barcharts_" + trailer_text + ".pdf"))
    cluster_directories = []
    # for each cluster, generate a bar chart 
    for i in tqdm(distinct_cluster_labels):
        # We allow for the case where some clusters are missing 
        plt.clf()
        paths_in_cluster = {}
        # get the files associated with the current cluster
        cluster_files = frame.loc[frame['cluster']==i]
        count = 0 
        for index, row in cluster_files.iterrows():
            if count>1:
                path = path_utilities.remove_path_end(row['filename'])
                # if the path is already in the cluster, add to count
                if path in paths_in_cluster:
                    paths_in_cluster.update({path:paths_in_cluster.get(path)+1})
                else:
                    paths_in_cluster.update({path:1})
            count+=1
        cluster_directories.append(paths_in_cluster)
        sorted_names = []
        sorted_counts = []
        # sort the paths in ascending order based on # of occurrences
        for e in sorted(paths_in_cluster, key=paths_in_cluster.get, reverse=True):
            trimmed_name = e[len(home_dir):]
            sorted_names.append(trimmed_name)
            sorted_counts.append(paths_in_cluster[e])

        cluster_stats = get_cluster_stats(paths_in_cluster)

        y_pos = np.arange(len(sorted_names))
        plt.bar(y_pos, sorted_counts, align='center', alpha=0.5)
        plt.xticks(y_pos, sorted_names, rotation=90)
        plt.rc('xtick')
        plt.ylabel('Number of files')
        plt.xlabel('Directory')
        plt.title('Directories in Cluster ' + str(i) + "\n" + cluster_stats)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, top=0.87)
        
        pdf.savefig(plt.gcf())
    pdf.close()
    np.save(os.path.join(file_place, "cluster_directories_" + trailer_text + ".npy"), cluster_directories)
    print("Bar charts written to \"text_barcharts_" + trailer_text + ".pdf\"")

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a list of filepaths
    RETURNS: nearest shared parent directory of all the files given '''
def nearest_shared_parent(filepaths):
    path1 = filepaths[0]
    path1_folders = path1.split("/")
    shared_index = len(path1_folders)
    for i in range(1,len(filepaths)):
        path2 = filepaths[i]
        path2_folders = path2.split("/")
        min_folderlist_len = min(len(path1_folders), len(path2_folders))
        for j in range(min_folderlist_len):
            # if the folder matches
            if path1_folders[j]==path2_folders[j]:
                # save the shared path
                shared_index = j+1
            # once they are no longer equal, stop iterating. 
            else:
                break
    path1 = path1_folders[:shared_index]
    return path1

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a dictionary of directories and their counts from ONE cluster
    DOES: returns a string of mean, std-dev, and nearest shared parent for that cluster '''
def get_cluster_stats(one_cluster_directories):
    dir_counts = np.array(list(one_cluster_directories.values()))
    unique = str(len(one_cluster_directories)) + " unique directories"
    avg = "Avg dir frequency: " + str( np.mean(dir_counts))
    med = "Median dir frequency: " + str( np.median(dir_counts))
    std = "Std-dev dir frequency: " + str( np.std(dir_counts))
    nsd = "Nearest shared directory: " + "/".join(nearest_shared_parent(list(one_cluster_directories.keys())))
    return unique+"\n"+avg+"\n"+med+"\n"+std+"\n"+nsd

def print_cluster_stats(frame, top_words, dataset_path, num_clusters):
    dataset_name, file_place = initialize_output_location(dataset_path)
    trailer_text = dataset_name + "_k=" + str(num_clusters)
    cluster_directories = np.load(os.path.join(file_place, "cluster_directories_" + trailer_text + ".npy"))
    
    fr = open(os.path.join(file_place, "cluster_stats_" + trailer_text + ".txt"), "w")
    total_silhouette, scores = silhouette.compute_silhouette(cluster_directories, dataset_path)
    num_files_per_cluster = frame['cluster'].value_counts().sort_index().tolist()

    print("\n\nComputing cluster statistics...")
    for clust_num in tqdm(range(len(cluster_directories))):
        c_stats = "Cluster " + str(clust_num) + "\n" + str(num_files_per_cluster[clust_num]) + " files \n"
        c_stats = c_stats + get_cluster_stats(cluster_directories[clust_num])
        c_stats = c_stats + "\nSilhouette score: " + str(scores[clust_num]) + "\nTop 10 words: " + ", ".join(top_words.get(clust_num))
        fr.write(c_stats + "\n\n")
    fr.write("\nTotal silhouette score: " + str(total_silhouette))
    fr.close()
    print("Cluster stats written to \"cluster_stats_" + trailer_text + ".txt\"")

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM

def runflow(num_clusters, retokenize, recluster, dataset_path, minibatch):
    
    if retokenize.lower() == 'y':
        retokenize = "1"
    else:
        retokenize = "0"

    if recluster.lower() == 'y':
        recluster = "1"
    else:
        recluster = "0"

    if minibatch.lower() == 'y':
        minibatch = "1"
 
    home_dir = path_utilities.remove_path_end(dataset_path)
    corpusdir = os.path.join(home_dir, "converted-" +  path_utilities.get_last_dir_from_path(dataset_path)) #eg. /home/ljung/converted/
 
    # record initial time that program started
    t0 = time()
    
    fr, all_cluster_words, distinct_cluster_labels = main_function(num_clusters, retokenize, recluster, corpusdir, dataset_path, 10, minibatch)
    bar_clusters(fr, distinct_cluster_labels, num_clusters, home_dir, dataset_path)    
    print_cluster_stats(fr, all_cluster_words, dataset_path, num_clusters)
    
    # print total time taken to run program
    print("\nTime taken: ", time()-t0, " seconds\n")

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    num_clusters = int(sys.argv[1])
    retokenize = sys.argv[2]
    recluster = sys.argv[3]
    # the directory containing the files not in pub8 you want to cluster
    dataset_path = sys.argv[4] #eg. /home/ljung/pub8
    print("Don't run this file standalone.")
    #runflow(num_clusters, retokenize, recluster, dataset_path) 

