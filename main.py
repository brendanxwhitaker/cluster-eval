from options import load_arguments
import converting_utilities
import document_clustering
import schema_clustering
import data_composition
import sys
import os

# EVERY OTHER FUNCTION WILL BE CALLED FROM HERE.

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    print("ARGUMENTS: ")
    args = load_arguments()
    print("Arguments loaded. ")

    if args.num_clusters_end < args.num_clusters_start:
        args.num_clusters_end = args.num_clusters_start
   
    if args.plot_extensions.lower() == 'y':
        data_composition.plot_extensions(args.dataset_path,
                                  args.num_extensions)

    if args.convert.lower() == 'y':
        converting_utilities.convert(args.dataset_path, 
                                     args.num_extensions)

    start = args.num_clusters_start
    end = args.num_clusters_end
    num_clusters = start

    print("Clustering for all k: " + str(start) + "<=k<=" + str(end) + "...\n")
    while num_clusters <= end:
        if args.cluster_struct.lower() == 'y':
            schema_clustering.runflow(args.dataset_path, 
                                      num_clusters, 
                                      args.overwrite_distmat_struct, 
                                      args.overwrite_plot_struct,
                                      args.fill_threshold)

        if args.cluster_text.lower() == 'y':
            document_clustering.runflow(num_clusters, 
                                        args.overwrite_tokens_text,
                                        args.overwrite_clusters_text,
                                        args.dataset_path,
                                        args.minibatch_kmeans)
        num_clusters += 1
 
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
